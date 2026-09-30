"""
burnin_pipeline.py - self-contained, end-to-end reference implementation (SIH 26170, ISRO)
AI-Driven Anomaly Detection in Component Burn-In & Screening.

Run:  python burnin_pipeline.py [--csv your_data.csv] [--cost-fn 50] [--cost-fp 1] [--dpat-k 6]
                                [--t-field 55] [--mission-years 15] [--ea 0.7] [--top 5]

Everything is in this one file (no imports from src/), in reading order:
  1. Config & schema   Lot_ID, Component_ID, Param_Name, Value_0h, Value_24h, Value_96h, Value_168h
                       + datasheet limits, delta limits (MIL-STD-883 / ESCC), two-sided flags
  2. Synthetic data    log-normal lot/part hierarchy, power-law drift, small & large lots, latent
                       defects (accelerating, late-bloomer after 96h, step, erratic), tester glitches
  3. Features          lot-localized robust z-scores of values and interval drifts
  4. Rules             SPEC_FAIL, DELTA_FAIL (|dP| > max(pct*|P0|, floor)) - never waived
  5. Module A          DPAT (log scale for currents) + per-lot MinCovDet Mahalanobis + Isolation Forest
  6. Module B          physics power law dP = A t^n (n per parameter), hybrid physics + XGBoost residual,
                       pure XGBoost, Ridge, naive; CQR conformal bounds; safety slope =
                       MIN(spec delta / 168h, lot median + k sigma, mission-life Arrhenius slope);
                       two-sided drift where configured
  7. Cost-sensitive    cost = C_FN*FN + C_FP*FP, thresholds tuned on validation lots; overkill,
                       Clopper-Pearson escape-rate bound, recall/overkill vs DPAT k
  8. Explainability    SHAP (TreeExplainer), Mahalanobis decomposition, binding safety limit
  9. Main              split by lot -> fit -> calibrate/tune -> evaluate on held-out lots -> explain
The modular package in src/ (used by train.py / app.py) implements the same algorithms.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ============================================================== 1. CONFIG & SCHEMA
TIMES = [0, 24, 96, 168]
H = 168.0
VCOLS = [f"Value_{t}h" for t in TIMES]
REQUIRED = ["Lot_ID", "Component_ID", "Param_Name"] + VCOLS
GROUP = ["Lot_ID", "Param_Name"]
#            unit, min, max, delta_pct, delta_floor, two_sided, log_scale
DATASHEET = {"Iddq": ("µA", 0.0, 25.0, 0.50, 1.0, False, True),
             "Leakage": ("µA", 0.0, 50.0, 1.00, 0.5, False, True),
             "Prop_Delay": ("ns", 0.0, 20.0, 0.08, 0.2, True, False)}
DEFAULT_SPEC = ("", -np.inf, np.inf, np.inf, np.inf, True, False)
SPEC_COLS = ["Unit", "Spec_Min", "Spec_Max", "Delta_Pct", "Delta_Floor", "Two_Sided", "Log_Scale"]
DPAT_K = 6.0           # AEC-Q001 Dynamic PAT: robust mean +/- k robust sigma (overridable: --dpat-k)
SLOPE_K = 6.0          # lot safety slope = median + k robust sigma of the lot's bound slopes
PI_Q = 0.9             # conformalized P90 upper / P10 lower bound
MAHA_Q = 0.999         # chi-square quantile that maps to Mahalanobis score 1
REVIEW_FRAC = 0.85     # REVIEW band: 0.85 <= combined score < 1
SEED = 42
PHYS = dict(ea=0.7, t_bi=125.0, t_field=55.0, mission_h=15 * 8760.0)
LIMITS = np.array(["spec_delta", "lot", "mission"])
LIMIT_TEXT = {"spec_delta": "spec delta limit/168h", "lot": "lot median + k·σ slope",
              "mission": "mission-life (Arrhenius) slope"}


def arrhenius_af(ea, t_stress_c, t_use_c):
    return float(np.exp(ea / 8.617e-5 * (1 / (t_use_c + 273.15) - 1 / (t_stress_c + 273.15))))


def robust_sigma(x):
    x = np.asarray(x, float)
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    if s <= 0:
        s = np.subtract(*np.percentile(x, [75, 25])) / 1.349
    return float(max(s, 1e-9 + 1e-6 * abs(med)))


def robust_z(s):
    return (s - s.median()) / robust_sigma(s.values)


# ============================================================== 2. SYNTHETIC DATA
#       median, lot_sd, part_sd, rel_noise, abs_floor, drift_A, drift_n, corner_sign
BASE = {"Iddq": (5.0, .25, .25, .015, .05, .04, .30, 1.0),
        "Leakage": (3.0, .35, .45, .02, .08, .05, .40, 1.0),
        "Prop_Delay": (12.0, .04, .025, .003, .01, .012, .20, -1.0)}
DEFECTS = {"gross_outlier": .07, "lot_outlier": .20, "accel_drift": .23, "late_bloomer": .18,
           "step_jump": .12, "erratic": .20}
T = np.array(TIMES, float)
TN = T / H


def defect_log_path(kind, rng, part_sd, corner):
    sc = 0.25 if part_sd < 0.05 else 1.0          # delay moves by %, currents by x-fold
    if kind == "accel_drift":                     # super-linear runaway, subtle at 24h
        sign = -1.0 if (corner < 0 and rng.random() < 0.3) else 1.0
        return sign * rng.uniform(.08, .8) * sc * TN ** rng.uniform(1.3, 2.5)
    if kind == "late_bloomer":                    # normal until 40-90h, separates after 96h
        t_on = rng.uniform(40, 90)
        return rng.uniform(.06, .5) * sc * np.clip((T - t_on) / (H - t_on), 0, None) ** 1.5
    if kind == "step_jump":
        return rng.uniform(.08, .6) * (.3 if sc < 1 else 1) * (T >= 96)
    return rng.normal(0, rng.uniform(.05, .25) * (.3 if sc < 1 else 1), 4) * np.r_[0, 1, 1, 1]


def generate(n_lots=20, defect_rate=0.035, glitch_rate=0.003, seed=7, small_frac=0.5):
    rng = np.random.default_rng(seed)
    params, kinds = list(BASE), list(DEFECTS)
    probs = np.array(list(DEFECTS.values())); probs /= probs.sum()
    rows = []
    for li in range(1, n_lots + 1):
        lot = f"L{li:02d}"
        n_parts = int(rng.integers(30, 81) if rng.random() < small_frac else rng.integers(250, 501))
        shift = {p: rng.normal(0, BASE[p][1]) for p in params}
        lot_A = {p: BASE[p][5] * rng.uniform(.5, 1.6) for p in params}
        lot_n = {p: BASE[p][6] * np.exp(rng.normal(0, .1)) for p in params}
        for ci in range(1, n_parts + 1):
            corner = rng.normal()
            bad = rng.random() < defect_rate
            kind = rng.choice(kinds, p=probs) if bad else None
            aff = set(rng.choice(params, rng.choice([1, 2], p=[.7, .3]), replace=False)) if bad else set()
            for p in params:
                med, _, psd, noise, floor, _, _, csign = BASE[p]
                unit, smin, smax = DATASHEET[p][:3]
                lot_med = med * np.exp(shift[p])
                v0 = lot_med * np.exp(psd * (.6 * csign * corner + .8 * rng.normal()))
                dlog = lot_A[p] * np.exp(rng.normal(0, .3)) * TN ** (lot_n[p] * np.exp(rng.normal(0, .15)))
                is_def = p in aff
                if is_def and kind == "gross_outlier":
                    true = smax * rng.uniform(1.05, 1.6) * np.exp(dlog)
                elif is_def and kind == "lot_outlier":   # inside datasheet, far from the lot
                    true = min(lot_med * np.exp(rng.uniform(4.5, 9) * psd), smax * .95) * np.exp(dlog)
                else:
                    extra = defect_log_path(kind, rng, psd, csign * corner) if is_def else 0.0
                    true = v0 * np.exp(dlog + extra)
                meas = true * np.exp(rng.normal(0, noise, 4)) + rng.normal(0, floor, 4)
                if not is_def and rng.random() < glitch_rate:      # tester glitch on a good part
                    meas[rng.choice([1, 2])] *= rng.uniform(1.2, 1.6)
                meas = np.maximum(meas, floor / 2)
                rows.append([lot, f"{lot}-{ci:04d}", p, *np.round(meas, 4), int(is_def),
                             kind if is_def else "none"])
    return fill_specs(pd.DataFrame(rows, columns=REQUIRED + ["Is_Defective", "Defect_Type"]))


def fill_specs(df):
    for i, col in enumerate(SPEC_COLS):
        if col not in df:
            df[col] = df["Param_Name"].map(lambda p: DATASHEET.get(p, DEFAULT_SPEC)[i])
    df[VCOLS] = df[VCOLS].astype(float).clip(lower=1e-6)
    df["Lot_ID"] = df["Lot_ID"].astype(str)
    return df


def load(path):
    df = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in df]
    if miss:
        sys.exit(f"CSV missing columns {miss}; expected {REQUIRED}")
    return fill_specs(df)


# ============================================================== 3. FEATURES (lot-localized)
A_FEATS = ["z_0h", "z_24h", "z_96h", "z_168h", "zd_0_24", "zd_24_96", "zd_96_168", "zd_0_168", "z_accel"]
B_FEATS = ["log_v0", "log_v24", "r24", "lot_med_log_v0", "lot_med_log_v24", "lot_med_r24",
           "rel_v0", "rel_r24", "is_Iddq", "is_Leakage", "is_Prop_Delay"]
H_FEATS = B_FEATS + ["phys"]
LABEL = {"z_0h": "Value_0h lot robust z-score", "z_24h": "Value_24h lot robust z-score",
         "z_96h": "Value_96h lot robust z-score", "z_168h": "Value_168h lot robust z-score",
         "zd_0_24": "0h→24h drift (lot z-score)", "zd_24_96": "24h→96h drift (lot z-score)",
         "zd_96_168": "96h→168h drift (lot z-score)", "zd_0_168": "total 0h→168h drift (lot z-score)",
         "z_accel": "drift acceleration (lot z-score)", "log_v0": "log Value_0h", "log_v24": "log Value_24h",
         "r24": "log ratio 24h/0h", "lot_med_log_v0": "lot median log Value_0h",
         "lot_med_log_v24": "lot median log Value_24h", "lot_med_r24": "lot median 24h/0h ratio",
         "rel_v0": "Value_0h vs lot median", "rel_r24": "0h→24h drift vs lot median drift",
         "is_Iddq": "param=Iddq", "is_Leakage": "param=Leakage", "is_Prop_Delay": "param=Prop_Delay",
         "phys": "physics baseline r24·7^n"}


def build_features(df):
    f = df.copy().reset_index(drop=True)
    lg = {t: np.log(f[f"Value_{t}h"]) for t in TIMES}
    tmp = pd.DataFrame({"d_0_24": lg[24] - lg[0], "d_24_96": lg[96] - lg[24],
                        "d_96_168": lg[168] - lg[96], "d_0_168": lg[168] - lg[0]})
    tmp["accel"] = tmp["d_96_168"] / 72 - tmp["d_0_24"] / 24
    for t in TIMES:
        tmp[f"log_{t}"] = lg[t]
    tmp[GROUP] = f[GROUP]
    g = tmp.groupby(GROUP)
    new = {f"z_{t}h": g[f"log_{t}"].transform(robust_z) for t in TIMES}
    new.update({f"zd_{k}": g[f"d_{k}"].transform(robust_z) for k in ["0_24", "24_96", "96_168", "0_168"]})
    new["z_accel"] = g["accel"].transform(robust_z)
    # Module B: only 0h/24h information + lot context
    new.update(log_v0=lg[0], log_v24=lg[24], r24=tmp["d_0_24"],
               lot_med_log_v0=g["log_0"].transform("median"), lot_med_log_v24=g["log_24"].transform("median"),
               lot_med_r24=g["d_0_24"].transform("median"))
    new["rel_v0"], new["rel_r24"] = new["log_v0"] - new["lot_med_log_v0"], new["r24"] - new["lot_med_r24"]
    for p in ["Iddq", "Leakage", "Prop_Delay"]:
        new[f"is_{p}"] = (f["Param_Name"] == p).astype(float)
    return pd.concat([f, pd.DataFrame(new)], axis=1)


# ============================================================== 4. RULES (never waived)
def rules(f):
    v = f[VCOLS].to_numpy(float)
    r = pd.DataFrame(index=f.index)
    r["spec_fail"] = ((v > f["Spec_Max"].to_numpy()[:, None]) | (v < f["Spec_Min"].to_numpy()[:, None])).any(1)
    allow = np.maximum(f["Delta_Pct"].to_numpy() * np.abs(v[:, 0]), f["Delta_Floor"].to_numpy())
    ratio = np.abs(v[:, 1:] - v[:, :1]) / allow[:, None]
    r["delta_allow"], r["delta_ratio"] = allow, ratio.max(1)
    r["delta_worst_t"] = np.array(["24h", "96h", "168h"])[ratio.argmax(1)]
    r["delta_fail"] = r["delta_ratio"] > 1
    r["rule_fail"] = r["spec_fail"] | r["delta_fail"]
    return r


# ============================================================== 5. MODULE A - contextual outliers
def dpat_z(f):
    """Robust z per (lot, param) of each value and the 0->168h drift; log scale for log-normal currents."""
    ls = f["Log_Scale"].to_numpy(bool)
    meas = {c: np.where(ls, np.log(f[c]), f[c]) for c in VCOLS}
    meas["Drift"] = np.where(ls, np.log(f["Value_168h"] / f["Value_0h"]), f["Value_168h"] - f["Value_0h"])
    tmp = f[GROUP].copy()
    return pd.DataFrame({c: tmp.assign(_v=m).groupby(GROUP)["_v"].transform(robust_z).to_numpy()
                         for c, m in meas.items()})


def mahalanobis_per_lot(f):
    """Robust (MinCovDet) Mahalanobis per (lot, param); contrib_i = d_i*(P d)_i sums to D^2.
    Lots with < 5p parts use a diagonal metric (features are already robust z-scores)."""
    p = len(A_FEATS)
    contrib = np.zeros((len(f), p))
    for _, idx in f.groupby(GROUP).indices.items():
        X = f.iloc[idx][A_FEATS].to_numpy(float)
        if len(X) >= 5 * p:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = MinCovDet(random_state=SEED).fit(X)
            mu, P = m.location_, m.get_precision()
        else:
            mu, P = np.zeros(p), np.eye(p)
        d = X - mu
        contrib[idx] = d * (d @ P)
    return np.maximum(contrib.sum(1), 0), contrib


class ModuleA:
    def fit(self, f):
        self.ifs, self.norm = {}, {}
        for p, g in f.groupby("Param_Name"):
            X = g[A_FEATS].to_numpy(float)
            m = IsolationForest(n_estimators=300, random_state=SEED).fit(X)
            a = -m.score_samples(X)
            self.ifs[p], self.norm[p] = m, (np.median(a), np.quantile(a, .99))
        return self

    def score(self, f):
        r = pd.DataFrame(index=f.index)
        z = dpat_z(f)
        for c in z:
            r[f"dz_{c}"] = z[c].values
        r["dpat_maxz"] = z.abs().max(axis=1).values
        r["dpat_score"] = r["dpat_maxz"] / DPAT_K
        d2, contrib = mahalanobis_per_lot(f)
        r["maha_d2"], r["maha_score"] = d2, np.sqrt(d2 / chi2.ppf(MAHA_Q, len(A_FEATS)))
        r[[f"mc_{a}" for a in A_FEATS]] = contrib
        r["if_score"] = 0.0
        for p, idx in f.groupby("Param_Name").indices.items():
            if p in self.ifs:
                a = -self.ifs[p].score_samples(f.iloc[idx][A_FEATS].to_numpy(float))
                med, q = self.norm[p]
                r.iloc[idx, r.columns.get_loc("if_score")] = (a - med) / max(q - med, 1e-9)
        r["A_score"] = r[["dpat_score", "maha_score", "if_score"]].max(axis=1)   # recall-first OR
        return r


# ============================================================== 6. MODULE B - physics-informed drift
def make_gbm(q=None):
    if HAS_XGB:
        kw = dict(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.9,
                  colsample_bytree=.9, random_state=SEED, n_jobs=4)
        if q is not None:
            kw.update(objective="reg:quantileerror", quantile_alpha=q)
        return xgb.XGBRegressor(**kw)
    if q is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=SEED)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=.05, random_state=SEED)


def conformal_q(scores, level):
    n = len(scores)
    return 0.0 if n == 0 else float(np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n), method="higher"))


class ModuleB:
    """y = log(V168/V0). physics: y = r24 * 7^n (power law A t^n, n per Param_Name);
    hybrid: physics + XGBoost(residual); xgboost; ridge; naive V0 + 7 (V24 - V0)."""
    MODELS = ["physics", "hybrid", "xgboost", "ridge", "naive"]

    def fit(self, f):
        y = np.log(f["Value_168h"] / f["Value_0h"]).to_numpy()
        meds = f[GROUP].assign(y=y, r24=f["r24"].values).groupby(GROUP)[["y", "r24"]].median().reset_index()
        self.n_exp = {}
        for p, g in meds.groupby("Param_Name"):
            ok = (np.sign(g["y"]) == np.sign(g["r24"])) & (g["r24"].abs() > 1e-6)
            ns = np.log(g.loc[ok, "y"] / g.loc[ok, "r24"]) / np.log(7.0)
            self.n_exp[p] = float(np.clip(np.median(ns), .05, 1.5)) if len(ns) else .5
        self.y_scale = {p: max(robust_sigma(y[i]), 1e-6) for p, i in f.groupby("Param_Name").indices.items()}
        X, Xh, phys, s = self._design(f)
        ys, rs = y / s, (y - phys) / s
        self.m_xgb, self.m_hyb = make_gbm().fit(X, ys), make_gbm().fit(Xh, rs)
        self.m_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
        self.q_hyb = (make_gbm(1 - PI_Q).fit(Xh, rs), make_gbm(PI_Q).fit(Xh, rs))
        return self

    def _design(self, f):
        n = f["Param_Name"].map(lambda p: self.n_exp.get(p, .5)).to_numpy(float)
        dflt = float(np.median(list(self.y_scale.values())))
        s = f["Param_Name"].map(lambda p: self.y_scale.get(p, dflt)).to_numpy(float)
        X = f[B_FEATS].to_numpy(float)
        phys = f["r24"].to_numpy() * 7.0 ** n
        return X, np.column_stack([X, phys]), phys, s

    def _preds(self, f):
        X, Xh, phys, s = self._design(f)
        v0, v24 = f["Value_0h"].to_numpy(), f["Value_24h"].to_numpy()
        out = {"physics": v0 * np.exp(phys), "hybrid": v0 * np.exp(phys + s * self.m_hyb.predict(Xh)),
               "xgboost": v0 * np.exp(s * self.m_xgb.predict(X)), "ridge": v0 * np.exp(self.m_ridge.predict(X)),
               "naive": v0 + 7.0 * (v24 - v0)}
        lo = phys + s * self.q_hyb[0].predict(Xh)
        hi = phys + s * self.q_hyb[1].predict(Xh)
        return out, lo, hi

    def calibrate(self, fv):
        """Default model = lowest validation nMAE; CQR offsets per parameter on validation lots."""
        preds, lo, hi = self._preds(fv)
        act = fv["Value_168h"].to_numpy()
        self.val_nmae = {m: float(np.mean(np.abs(p - act) / act)) for m, p in preds.items()}
        self.default = min((m for m in preds if m != "naive"), key=self.val_nmae.get)
        y = np.log(act / fv["Value_0h"].to_numpy())
        self.cqr = {p: (conformal_q(lo[i] - y[i], PI_Q), conformal_q(y[i] - hi[i], PI_Q))
                    for p, i in fv.groupby("Param_Name").indices.items()}
        return self

    def predict(self, f):
        preds, lo, hi = self._preds(f)
        v0 = f["Value_0h"].to_numpy()
        r = pd.DataFrame({f"pred_168_{m}": p for m, p in preds.items()}, index=f.index)
        r["pred_168"] = r[f"pred_168_{self.default}"]
        c_lo = f["Param_Name"].map(lambda p: self.cqr.get(p, (0, 0))[0]).to_numpy()
        c_hi = f["Param_Name"].map(lambda p: self.cqr.get(p, (0, 0))[1]).to_numpy()
        r["pred_168_upper"] = np.maximum(v0 * np.exp(hi + c_hi), r["pred_168"])
        r["pred_168_lower"] = np.minimum(v0 * np.exp(lo - c_lo), r["pred_168"])
        r["n_exp"] = f["Param_Name"].map(lambda p: self.n_exp.get(p, .5)).to_numpy()
        r["slope_upper"], r["slope_lower"] = (r["pred_168_upper"] - v0) / H, (r["pred_168_lower"] - v0) / H
        # (a) spec delta limit: max(pct * |V0|, floor) over the burn-in duration
        s_spec = np.maximum(f["Delta_Pct"].to_numpy() * np.abs(v0), f["Delta_Floor"].to_numpy()) / H
        # (b) lot robust slope (DPAT applied to the lot's bound slopes)
        tmp = pd.concat([f[GROUP], r[["slope_upper", "slope_lower"]]], axis=1)
        gu, gl = tmp.groupby(GROUP)["slope_upper"], tmp.groupby(GROUP)["slope_lower"]
        med_u, sig_u = gu.transform("median"), gu.transform(lambda s: robust_sigma(s.values))
        med_l, sig_l = gl.transform("median"), gl.transform(lambda s: robust_sigma(s.values))
        s_lot_u, s_lot_l = med_u + SLOPE_K * sig_u, med_l - SLOPE_K * sig_l
        # (c) mission life: V0 + A (T_mission/AF)^n <= limit with A = delta168 / 168^n
        factor = (H * AF / PHYS["mission_h"]) ** r["n_exp"].to_numpy()
        head_u, head_l = f["Spec_Max"].to_numpy() - v0, v0 - f["Spec_Min"].to_numpy()
        s_mis_u = np.where(np.isfinite(head_u), np.maximum(head_u, 0) * factor / H, np.inf)
        s_mis_l = np.where(np.isfinite(head_l), -np.maximum(head_l, 0) * factor / H, -np.inf)
        up = np.column_stack([s_spec, s_lot_u, s_mis_u])
        dn = np.column_stack([-s_spec, s_lot_l, s_mis_l])
        r["safe_spec"], r["safe_lot"], r["safe_mission"] = up.T
        r["safety_slope"], r["binding_up"] = up.min(1), LIMITS[up.argmin(1)]
        r["safety_slope_lower"], r["binding_lo"] = dn.max(1), LIMITS[dn.argmax(1)]
        # B = 1 exactly at the binding limit; datasheet term = 0 at the lot median bound, 1 at the limit
        d_u = np.maximum.reduce([r["safety_slope"] - med_u, sig_u, np.full(len(r), 1e-12)])
        d_l = np.maximum.reduce([med_l - r["safety_slope_lower"], sig_l, np.full(len(r), 1e-12)])
        r["B_up"] = 1 + (r["slope_upper"] - r["safety_slope"]) / d_u
        r["B_down"] = np.where(f["Two_Sided"].to_numpy(bool),
                               1 + (r["safety_slope_lower"] - r["slope_lower"]) / d_l, -np.inf)
        smax = f["Spec_Max"].to_numpy()
        m_hi = pd.concat([f[GROUP], r["pred_168_upper"]], axis=1).groupby(GROUP)["pred_168_upper"].transform("median")
        with np.errstate(divide="ignore", invalid="ignore"):
            b_spec = (r["pred_168_upper"] - m_hi) / np.maximum(smax - m_hi, 1e-12)
        r["B_spec"] = np.where(np.isfinite(smax), b_spec, -np.inf)
        r["B_score"] = r[["B_up", "B_down", "B_spec"]].max(axis=1)
        r["B_flag"] = r["B_score"] >= 1
        return r


# ============================================================== 7. COST-SENSITIVE THRESHOLDS & METRICS
def best_threshold(s, y, c_fn, c_fp):
    """Threshold (midpoint between distinct scores) minimising c_fn*FN + c_fp*FP; ties -> lower."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    u = np.unique(s)
    cands = np.r_[u[0] - 1e-9, (u[:-1] + u[1:]) / 2, u[-1] + 1e-9]
    o = np.argsort(s)
    pos, neg = np.r_[0, np.cumsum(y[o])], np.r_[0, np.cumsum(~y[o])]
    k = np.searchsorted(s[o], cands)
    cost = c_fn * pos[k] + c_fp * (neg[-1] - neg[k])
    i = int(np.argmin(cost))
    return float(cands[i]), float(cost[i])


def best_threshold_pair(a, b, y, c_fn, c_fp):
    """Joint (tau_a, tau_b) for `A >= tau_a OR B >= tau_b` (rule failures are already caught)."""
    a, b, y = np.asarray(a, float), np.asarray(b, float), np.asarray(y, bool)
    grid = np.unique(np.quantile(b, np.linspace(.5, 1, 200)))
    best = (np.inf, None, None)
    for tb in np.r_[grid[grid > .05] + 1e-9, np.inf]:
        caught = b >= tb
        if (~caught).sum() == 0:
            continue
        ta, c = best_threshold(a[~caught], y[~caught], c_fn, c_fp)
        c += c_fp * (caught & ~y).sum()
        if c < best[0]:
            best = (c, max(ta, 1e-3), tb)
    return best[1], best[2], best[0]


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound on a binomial rate (0 escapes -> ~3/n)."""
    return 1.0 if n == 0 or k >= n else float(beta_dist.ppf(conf, k + 1, n - k))


def report(y, pred, c_fn, c_fp, b=2.0):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn, tn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum(), (~y & ~pred).sum()
    rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(recall=rec, precision=prec, f2=(1 + b * b) * prec * rec / max(b * b * prec + rec, 1e-12),
                overkill=fp / max(fp + tn, 1), escapes=int(fn), escape_ub95=cp_upper(int(fn), int(tp + fn)),
                false_rejects=int(fp), cost=c_fn * fn + c_fp * fp)


# ============================================================== 8. EXPLAINABILITY
def if_shap(model, X):
    if HAS_SHAP:
        try:
            return -np.asarray(shap.TreeExplainer(model).shap_values(X))   # + = more anomalous
        except Exception:
            pass
    return np.zeros_like(X)


def gbm_shap(model, X):
    if HAS_SHAP:
        try:
            return np.asarray(shap.TreeExplainer(model).shap_values(X))
        except Exception:
            pass
    if HAS_XGB:
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    return np.zeros_like(X)


def fmt(v):
    return f"{v:.3g}" if abs(v) < 100 else f"{v:.1f}"


def explain_row(r, lot_rows, ma, mb):
    u, p = r["Unit"], r["Param_Name"]
    zc = {c: r[f"dz_{c}"] for c in VCOLS + ["Drift"]}
    worst = max(zc, key=lambda c: abs(zc[c]))
    txt = [f"{p}: {'0h→168h drift' if worst == 'Drift' else worst} is {zc[worst]:+.1f} robust-σ from the lot "
           f"median (DPAT k={DPAT_K:g}{', log scale' if r['Log_Scale'] else ''})."]
    peak = max(VCOLS, key=lambda c: r[c])
    if r["spec_fail"]:
        txt.append(f"SPEC_FAIL: datasheet limit exceeded (peak {fmt(r[peak])} {u} vs max {fmt(r['Spec_Max'])}).")
    else:
        txt.append(f"All readings within the datasheet max {fmt(r['Spec_Max'])} {u} (peak {fmt(r[peak])}).")
    if r["delta_fail"]:
        txt.append(f"DELTA_FAIL: |Δ| at {r['delta_worst_t']} = {r['delta_ratio']:.1f}× the allowed "
                   f"±{fmt(r['delta_allow'])} {u} (max(pct·|V0|, floor)).")
    mc = np.array([r[f"mc_{a}"] for a in A_FEATS])
    j = int(np.argmax(mc))
    txt.append(f"Top Mahalanobis contributor: {LABEL[A_FEATS[j]]} ({100 * mc[j] / max(r['maha_d2'], 1e-12):.0f}% of D²).")
    if p in ma.ifs:
        sv = if_shap(ma.ifs[p], r[A_FEATS].to_numpy(float)[None])[0]
        txt.append(f"Top Isolation-Forest SHAP driver: {LABEL[A_FEATS[int(np.argmax(sv))]]}.")
    xh = np.r_[r[B_FEATS].to_numpy(float), r["r24"] * 7.0 ** r["n_exp"]][None]
    sb = np.abs(gbm_shap(mb.m_hyb, xh)[0])
    verdict = "EXCEEDS" if r["slope_upper"] > r["safety_slope"] else "within"
    txt.append(f"Drift from 0h+24h only ({mb.default}, n={r['n_exp']:.2f}): predicted 168h = {fmt(r['pred_168'])} {u} "
               f"(P{int(PI_Q * 100)} conformal band {fmt(r['pred_168_lower'])}–{fmt(r['pred_168_upper'])}; actual "
               f"{fmt(r['Value_168h'])}); upper slope {r['slope_upper']:.4g} {u}/h {verdict} safety slope "
               f"{r['safety_slope']:.4g} {u}/h = min(spec-delta {r['safe_spec']:.3g}, lot {r['safe_lot']:.3g}, "
               f"mission-life {r['safe_mission']:.3g}) → binding limit: {LIMIT_TEXT[r['binding_up']]}.")
    if r["Two_Sided"] and r["B_down"] >= 1:
        txt.append(f"Downward drift: lower slope {r['slope_lower']:.4g} below {r['safety_slope_lower']:.4g} {u}/h "
                   f"(binding: {LIMIT_TEXT[r['binding_lo']]}).")
    txt.append(f"Main hybrid-residual SHAP driver: {LABEL[H_FEATS[int(np.argmax(sb))]]}.")
    return " ".join(txt)


# ============================================================== 9. MAIN
def score(df, ma, mb, ta, tb):
    f = build_features(df)
    rows = pd.concat([f, rules(f), ma.score(f), mb.predict(f)], axis=1)
    rows["A_eff"] = np.where(rows["rule_fail"], 1e6, rows["A_score"])
    rows["C_score"] = np.maximum(rows["A_eff"] / ta, rows["B_score"] / tb)
    g = rows.groupby(["Lot_ID", "Component_ID"])
    comp = pd.DataFrame({"A": g["A_eff"].max(), "B": g["B_score"].max(), "C": g["C_score"].max(),
                         "rule_fail": g["rule_fail"].any(), "B_flag": g["B_flag"].any()}).reset_index()
    if "Is_Defective" in rows:
        comp = comp.merge(g["Is_Defective"].max().reset_index(), on=["Lot_ID", "Component_ID"])
        comp = comp.merge(g["Defect_Type"].agg(lambda s: ",".join(sorted(set(s) - {"none"})) or "none")
                          .reset_index(), on=["Lot_ID", "Component_ID"])
    comp["Decision"] = np.select([comp["rule_fail"] | (comp["C"] >= 1), comp["C"] >= REVIEW_FRAC],
                                 ["REJECT", "REVIEW"], "ACCEPT")
    return rows, comp


def main():
    global DPAT_K, AF
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="labelled CSV in the wide schema (default: synthetic)")
    ap.add_argument("--cost-fn", type=float, default=50.0)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--dpat-k", type=float, default=DPAT_K)
    ap.add_argument("--ea", type=float, default=PHYS["ea"])
    ap.add_argument("--t-field", type=float, default=PHYS["t_field"])
    ap.add_argument("--mission-years", type=float, default=PHYS["mission_h"] / 8760)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    DPAT_K = a.dpat_k
    PHYS.update(ea=a.ea, t_field=a.t_field, mission_h=a.mission_years * 8760)
    AF = arrhenius_af(a.ea, PHYS["t_bi"], a.t_field)

    df = load(a.csv) if a.csv else generate()
    sizes = df.groupby("Lot_ID")["Component_ID"].nunique()
    print(f"Data: {len(df)} rows, {df['Component_ID'].nunique()} components, {len(sizes)} lots "
          f"(sizes {sizes.min()}-{sizes.max()}); backend={'xgboost' if HAS_XGB else 'sklearn-hgb'}, shap={HAS_SHAP}")
    print(f"Arrhenius AF({a.ea:g} eV, {PHYS['t_bi']:g}→{a.t_field:g} °C) = {AF:.1f}; 168h burn-in ≈ "
          f"{168 * AF / 8760:.1f} field-years; mission {a.mission_years:g} years")

    # Split by LOT (60/20/20) so no lot leaks between train / validation / test.
    lots = np.array(sorted(df["Lot_ID"].unique()), dtype=object)
    np.random.default_rng(SEED).shuffle(lots)
    n = len(lots); nt = max(1, round(.2 * n)); nv = max(1, round(.2 * n))
    test, val, train = (df[df["Lot_ID"].isin(s)].reset_index(drop=True)
                        for s in (lots[:nt], lots[nt:nt + nv], lots[nt + nv:]))
    print("Test lots:", sorted(lots[:nt]), "| Validation lots:", sorted(lots[nt:nt + nv]))

    ftrain = build_features(train)
    ma, mb = ModuleA().fit(ftrain), ModuleB().fit(ftrain)
    mb.calibrate(build_features(val))
    print(f"Power-law n per parameter: " + ", ".join(f"{p}={v:.2f}" for p, v in mb.n_exp.items()))
    print(f"Validation nMAE: " + ", ".join(f"{m}={100 * v:.2f}%" for m, v in mb.val_nmae.items())
          + f"  → default forecast model: {mb.default}")

    _, cv = score(val, ma, mb, 1.0, 1.0)
    ta, tb, cost_c = best_threshold_pair(cv["A"], cv["B"], cv["Is_Defective"], a.cost_fn, a.cost_fp)
    print(f"\nCost = {a.cost_fn:g}*FN + {a.cost_fp:g}*FP (validation lots): REJECT if rule fails OR "
          f"A >= {ta:.3f} OR B >= {tb:.3f} (val cost {cost_c:.0f})")

    rows, comp = score(test, ma, mb, ta, tb)
    y = comp["Is_Defective"].to_numpy()
    res = {"Rules only (spec+delta)": comp["rule_fail"],
           "Module B early @24h": comp["B_flag"],
           "Combined REJECT": comp["Decision"] == "REJECT",
           "Combined REJECT+REVIEW": comp["Decision"] != "ACCEPT"}
    print(f"\n=== Detection on held-out test lots ({len(comp)} components, {int(y.sum())} defective) ===")
    for k, pred in res.items():
        r = report(y, pred, a.cost_fn, a.cost_fp)
        print(f"  {k:24s} recall={r['recall']:.3f} overkill={100 * r['overkill']:.2f}% F2={r['f2']:.3f} "
              f"escapes={r['escapes']} (95% UB escape rate {100 * r['escape_ub95']:.1f}%) cost={r['cost']:.0f}")

    print("\n=== Recall / overkill vs DPAT k (rules + DPAT only, no tuning) ===")
    dmax = rows.groupby("Component_ID")["dpat_maxz"].max().reindex(comp["Component_ID"]).to_numpy()
    for k in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        r = report(y, comp["rule_fail"].to_numpy() | (dmax >= k), a.cost_fn, a.cost_fp)
        print(f"  k={k:<4g} recall={r['recall']:.3f} overkill={100 * r['overkill']:.2f}% escapes={r['escapes']}")

    print("\n=== Module B: Value_168h MAE on test lots (all models; * = default) ===")
    for m in ModuleB.MODELS:
        err = (rows[f"pred_168_{m}"] - rows["Value_168h"]).abs()
        per = ", ".join(f"{p}={e:.4f}" for p, e in err.groupby(rows["Param_Name"]).mean().items())
        star = "*" if m == mb.default else " "
        print(f"  {m:8s}{star} overall={err.mean():.4f}  nMAE={100 * (err / rows['Value_168h']).mean():.2f}%  ({per})")
    cov = ((rows["Value_168h"] <= rows["pred_168_upper"]) & (rows["Value_168h"] >= rows["pred_168_lower"])).mean()
    print(f"  CQR P{int(PI_Q * 100)} upper coverage {(rows['Value_168h'] <= rows['pred_168_upper']).mean():.3f}, "
          f"band coverage {cov:.3f}")
    print("  Binding safety-slope limit (rows): " +
          ", ".join(f"{k}={v}" for k, v in rows["binding_up"].value_counts().items()))

    print(f"\n=== Top {a.top} rejected components (QA justifications) ===")
    rej = comp[comp["Decision"] == "REJECT"].sort_values(["rule_fail", "C"], ascending=[True, False])
    for _, c in rej.head(a.top).iterrows():
        r = rows[rows["Component_ID"] == c["Component_ID"]].sort_values("C_score", ascending=False).iloc[0]
        lot_rows = rows[(rows["Lot_ID"] == r["Lot_ID"]) & (rows["Param_Name"] == r["Param_Name"])]
        truth = f" [ground truth: {c['Defect_Type']}]" if "Defect_Type" in c else ""
        print(f"\n- Component {c['Component_ID']} REJECTED (A={min(c['A'], 999):.2f} vs τA={ta:.2f}, "
              f"B={c['B']:.2f} vs τB={tb:.2f}){truth}\n  {explain_row(r, lot_rows, ma, mb)}")


AF = arrhenius_af(PHYS["ea"], PHYS["t_bi"], PHYS["t_field"])

if __name__ == "__main__":
    main()

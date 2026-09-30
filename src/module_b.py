"""Module B - physics-informed early (24h) drift predictor.

Inputs : Value_0h, Value_24h (+ lot context at 0h/24h and parameter type). No 96h/168h leakage.
Space  : y = log(V168/V0) (multiplicative drift, scale-free across parameters and lots).
Models :
  physics  power law  dlog(t) = A * t^n  =>  y_phys = r24 * 7^n,  n learned per Param_Name from
           training lots as median_lots[ ln(median y / median r24) / ln 7 ]
  hybrid   y_phys + XGBoost(residual) with asymmetric loss      (docs/RESEARCH.md §5.2)
  xgboost  XGBoost on the same features, no physics
  ridge    linear baseline;  naive  V0 + 7 * (V24 - V0)
  The default model is the one with the lowest validation-lot error.
Bounds : conformalized quantile regression (CQR): XGBoost P10/P90 quantile models, corrected on
         validation lots per Param_Name so the empirical coverage matches the nominal level.
Safety slope (per part, native units / hour) = MIN of
  (a) spec delta limit / 168h, delta = max(pct * |V0|, floor)       (MIL-STD-883 / ESCC style)
  (b) lot robust slope: median + k * robust sigma of the lot's bound slopes (DPAT on slopes)
  (c) mission-life slope: slope that breaches the datasheet limit within mission life, with field
      hours = AF(Arrhenius) x burn-in hours and power-law extrapolation with the learned n
Flag if the upper-bound slope exceeds the upward safety slope, or (two-sided parameters) the
lower-bound slope is below the downward safety slope, or the bound breaches the datasheet limit.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import BURNIN_HOURS, Config
from .features import B_FEATURES, GROUP, robust_sigma
from .custom_loss import asymmetric_huber_grad

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:  # pragma: no cover - fallback path
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor

MODELS = ["physics", "hybrid", "xgboost", "ridge", "naive"]
H_FEATURES = B_FEATURES + ["phys"]
LIMIT_NAMES = {"spec_delta": "spec delta limit / 168h", "lot": "lot robust slope (median + k·σ)",
               "mission": "mission-life (Arrhenius) slope"}


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


class ModuleB:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.backend = "xgboost" if HAS_XGB else "sklearn-hgb"
        self.default = "hybrid"
        self.n_exp, self.cqr = {}, {}
        self.val_mae = {}

    def _gbm(self, quantile=None):
        rs = self.cfg.random_state
        if HAS_XGB:
            kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, random_state=rs, n_jobs=4,
                      # Monotonic constraints: +1 means higher input -> higher output (worse drift)
                      # Applied to B_FEATURES: log_v0, log_v24, r24, lot_med_log_v0, lot_med_log_v24,
                      # lot_med_r24, rel_v0, rel_r24, is_Iddq/Leakage/Prop_Delay
                      monotone_constraints=(1, 1, 1, 0, 0, 0, 1, 0, 0))
            if quantile is not None:
                kw.update(objective="reg:quantileerror", quantile_alpha=quantile)
            return xgb.XGBRegressor(**kw)
        if quantile is not None:
            return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=rs)
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=rs)

    # ------------------------------------------------------------------ physics
    @staticmethod
    def fit_power_law(feat: pd.DataFrame) -> dict:
        """Exponent n per Param_Name from lot-median drifts (robust to per-part noise)."""
        y = np.log(feat["Value_168h"] / feat["Value_0h"])
        tmp = feat[GROUP].assign(y=y.values, r24=feat["r24"].values)
        out = {}
        for p, g in tmp.groupby("Param_Name"):
            meds = g.groupby("Lot_ID")[["y", "r24"]].median()
            ok = (np.sign(meds["y"]) == np.sign(meds["r24"])) & (meds["r24"].abs() > 1e-6)
            ns = np.log(meds.loc[ok, "y"] / meds.loc[ok, "r24"]) / np.log(7.0)
            out[p] = float(np.clip(np.median(ns), 0.05, 1.5)) if len(ns) else 0.5
        return out

    def _n_for(self, params: pd.Series) -> np.ndarray:
        return params.map(lambda p: self.n_exp.get(p, 0.5)).to_numpy(float)

    def _design(self, feat):
        X = feat[B_FEATURES].to_numpy(float)
        phys = feat["r24"].to_numpy() * 7.0 ** self._n_for(feat["Param_Name"])
        return X, np.column_stack([X, phys]), phys

    # ------------------------------------------------------------------ training
    def _scale(self, params: pd.Series) -> np.ndarray:
        """Per-parameter target scale, so pooled trees resolve small-drift parameters (e.g. delay)."""
        default = float(np.median(list(self.y_scale.values()))) if self.y_scale else 1.0
        return params.map(lambda p: self.y_scale.get(p, default)).to_numpy(float)

    def fit(self, feat: pd.DataFrame):
        self.n_exp = self.fit_power_law(feat)
        X, Xh, phys = self._design(feat)
        y = np.log(feat["Value_168h"] / feat["Value_0h"]).to_numpy()
        self.y_scale = {p: max(robust_sigma(y[idx]), 1e-6)
                        for p, idx in feat.groupby("Param_Name").indices.items()}
        s = self._scale(feat["Param_Name"])
        ys, rs = y / s, (y - phys) / s
        self.m_xgb = self._gbm().fit(X, ys)

        # Hybrid model with asymmetric loss (penalize under-prediction 10× more)
        if HAS_XGB:
            def xgb_asymmetric_obj(y_pred, dtrain):
                y_true = dtrain.get_label()
                return asymmetric_huber_grad(y_true, y_pred, alpha=10.0)

            dtrain = xgb.DMatrix(Xh, rs)
            self.m_hyb = xgb.train(
                params={
                    'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05,
                    'subsample': 0.9, 'colsample_bytree': 0.9, 'random_state': self.cfg.random_state,
                    'monotone_constraints': (1, 1, 1, 0, 0, 0, 1, 0, 0, 0),
                },
                dtrain=dtrain,
                obj=xgb_asymmetric_obj,
                num_boost_round=400
            )
        else:
            self.m_hyb = self._gbm().fit(Xh, rs)

        self.m_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
        q = self.cfg.pi_quantile
        self.q_models = {"xgboost": (self._gbm(1 - q).fit(X, ys), self._gbm(q).fit(X, ys)),
                         "hybrid": (self._gbm(1 - q).fit(Xh, rs), self._gbm(q).fit(Xh, rs))}
        return self

    def _logratio_preds(self, feat):
        X, Xh, phys = self._design(feat)
        s = self._scale(feat["Param_Name"])
        return {"physics": phys, "hybrid": phys + s * self.m_hyb.predict(Xh), "xgboost": s * self.m_xgb.predict(X),
                "ridge": self.m_ridge.predict(X)}, X, Xh, phys

    def _raw_bounds(self, feat, X, Xh, phys):
        fam = self.default if self.default in self.q_models else "hybrid"
        lo_m, hi_m = self.q_models[fam]
        s = self._scale(feat["Param_Name"])
        Xq, off = (Xh, phys) if fam == "hybrid" else (X, 0.0)
        return s * lo_m.predict(Xq) + off, s * hi_m.predict(Xq) + off

    def calibrate(self, feat_val: pd.DataFrame):
        """Pick the default model and conformalize bounds on validation lots (never on test lots)."""
        preds, X, Xh, phys = self._logratio_preds(feat_val)
        v0, v24 = feat_val["Value_0h"].to_numpy(), feat_val["Value_24h"].to_numpy()
        actual = feat_val["Value_168h"].to_numpy()
        native = {m: v0 * np.exp(p) for m, p in preds.items()}
        native["naive"] = v0 + 7.0 * (v24 - v0)
        self.val_mae = {m: float(np.mean(np.abs(native[m] - actual) / actual)) for m in native}
        self.default = min((m for m in native if m != "naive"), key=self.val_mae.get)
        y = np.log(actual / v0)
        lo, hi = self._raw_bounds(feat_val, X, Xh, phys)
        level = self.cfg.pi_quantile
        self.cqr = {}
        for p, idx in feat_val.groupby("Param_Name").indices.items():
            self.cqr[p] = (conformal_quantile(lo[idx] - y[idx], level),
                           conformal_quantile(y[idx] - hi[idx], level))
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        preds, X, Xh, phys = self._logratio_preds(feat)
        v0, v24 = feat["Value_0h"].to_numpy(), feat["Value_24h"].to_numpy()
        res = feat[["Lot_ID", "Component_ID", "Param_Name"]].copy()
        for m, p in preds.items():
            res[f"pred_168_{m}"] = v0 * np.exp(p)
        res["pred_168_naive"] = v0 + 7.0 * (v24 - v0)
        res["pred_168"] = res[f"pred_168_{self.default}"]
        lo, hi = self._raw_bounds(feat, X, Xh, phys)
        c_lo = feat["Param_Name"].map(lambda p: self.cqr.get(p, (0.0, 0.0))[0]).to_numpy()
        c_hi = feat["Param_Name"].map(lambda p: self.cqr.get(p, (0.0, 0.0))[1]).to_numpy()
        res["pred_168_upper"] = np.maximum(v0 * np.exp(hi + c_hi), res["pred_168"])
        res["pred_168_lower"] = np.minimum(v0 * np.exp(lo - c_lo), res["pred_168"])
        res["n_exp"] = self._n_for(feat["Param_Name"])

        H = BURNIN_HOURS
        res["slope_pred"] = (res["pred_168"] - v0) / H
        res["slope_upper"] = (res["pred_168_upper"] - v0) / H
        res["slope_lower"] = (res["pred_168_lower"] - v0) / H

        # (a) spec delta limit
        d_allow = np.maximum(feat["Delta_Pct"].to_numpy() * np.abs(v0), feat["Delta_Floor"].to_numpy())
        res["delta_allow"] = d_allow
        s_spec = d_allow / H
        # (b) lot robust slope (DPAT on bound slopes)
        g_up, g_lo = res.groupby(GROUP)["slope_upper"], res.groupby(GROUP)["slope_lower"]
        med_up, sig_up = g_up.transform("median"), g_up.transform(lambda s: robust_sigma(s.values))
        med_lo, sig_lo = g_lo.transform("median"), g_lo.transform(lambda s: robust_sigma(s.values))
        s_lot_up, s_lot_lo = med_up + cfg.slope_k * sig_up, med_lo - cfg.slope_k * sig_lo
        # (c) mission life: V0 + A*(T_mission/AF)^n <= limit, A = delta168 / 168^n
        factor = (H * cfg.af / cfg.mission_hours) ** res["n_exp"].to_numpy()
        head_up = (feat["Spec_Max"].to_numpy() - v0) * (1 - cfg.mission_margin)
        head_lo = (v0 - feat["Spec_Min"].to_numpy()) * (1 - cfg.mission_margin)
        s_mis_up = np.where(np.isfinite(head_up), np.maximum(head_up, 0) * factor / H, np.inf)
        s_mis_lo = np.where(np.isfinite(head_lo), -np.maximum(head_lo, 0) * factor / H, -np.inf)

        up = np.column_stack([s_spec, s_lot_up, s_mis_up])
        lo_ = np.column_stack([-s_spec, s_lot_lo, s_mis_lo])
        names = np.array(list(LIMIT_NAMES))
        res["safe_slope_spec"], res["safe_slope_lot"], res["safe_slope_mission"] = s_spec, s_lot_up, s_mis_up
        res["safety_slope"] = up.min(axis=1)
        res["binding_up"] = names[up.argmin(axis=1)]
        res["safety_slope_lower"] = lo_.max(axis=1)
        res["binding_lo"] = names[lo_.argmax(axis=1)]
        res["safe_lower_spec"], res["safe_lower_lot"], res["safe_lower_mission"] = -s_spec, s_lot_lo, s_mis_lo

        # Scores: 1.0 exactly at the binding threshold, scaled by the lot's own spread.
        d_up = np.maximum.reduce([res["safety_slope"] - med_up, sig_up, np.full(len(res), 1e-12)])
        d_lo = np.maximum.reduce([med_lo - res["safety_slope_lower"], sig_lo, np.full(len(res), 1e-12)])
        res["B_up"] = 1 + (res["slope_upper"] - res["safety_slope"]) / d_up
        two = feat["Two_Sided"].to_numpy(bool)
        res["B_down"] = np.where(two, 1 + (res["safety_slope_lower"] - res["slope_lower"]) / d_lo, -np.inf)
        # Datasheet breach by 168h: 0 at the lot's median bound, 1 exactly at the limit.
        spec_max, spec_min = feat["Spec_Max"].to_numpy(), feat["Spec_Min"].to_numpy()
        m_hi = res.groupby(GROUP)["pred_168_upper"].transform("median").to_numpy()
        m_lo = res.groupby(GROUP)["pred_168_lower"].transform("median").to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            b_hi = (res["pred_168_upper"].to_numpy() - m_hi) / np.maximum(spec_max - m_hi, 1e-12)
            b_lo = (m_lo - res["pred_168_lower"].to_numpy()) / np.maximum(m_lo - spec_min, 1e-12)
        res["B_spec"] = np.maximum(np.where(np.isfinite(spec_max), b_hi, -np.inf),
                                   np.where(np.isfinite(spec_min) & (spec_min > 0), b_lo, -np.inf))
        res["B_score"] = res[["B_up", "B_down", "B_spec"]].max(axis=1)
        res["B_direction"] = res[["B_up", "B_down", "B_spec"]].idxmax(axis=1).str.replace("B_", "")
        res["B_flag"] = res["B_score"] >= 1.0
        return res

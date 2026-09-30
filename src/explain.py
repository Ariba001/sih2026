"""Explainability: SHAP values, reason codes and plain-language QA justifications."""
import numpy as np
import pandas as pd

from .config import VALUE_COLS
from .features import A_FEATURES, B_FEATURES, FEATURE_LABELS
from .module_b import H_FEATURES, LIMIT_NAMES

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

FEATURE_LABELS = {**FEATURE_LABELS, "phys": "physics power-law baseline (Δ24·7ⁿ)"}


def tree_shap(model, X: np.ndarray) -> np.ndarray:
    """Per-feature SHAP contributions for a tree regressor (log-ratio space)."""
    if len(X) == 0:
        return np.zeros((0, X.shape[1]))
    if HAS_SHAP:
        try:
            return np.asarray(shap.TreeExplainer(model).shap_values(X))
        except Exception:
            pass
    if hasattr(model, "get_booster"):  # XGBoost's native exact TreeSHAP
        import xgboost as xgb
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    return np.zeros_like(X)


def iforest_shap(model, X: np.ndarray) -> np.ndarray:
    """Contribution of each feature towards *anomaly* (positive = makes the part more anomalous)."""
    if len(X) == 0:
        return np.zeros((0, X.shape[1]))
    if HAS_SHAP:
        try:
            # TreeExplainer explains the path length (higher = more normal) -> negate
            return -np.asarray(shap.TreeExplainer(model).shap_values(X))
        except Exception:
            pass
    base = -model.score_samples(X)  # fallback: occlusion to the lot median (z = 0)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        Xo = X.copy()
        Xo[:, j] = 0.0
        out[:, j] = base + model.score_samples(Xo)
    return out


def b_features(module_b) -> list:
    return B_FEATURES if module_b.default in ("xgboost", "ridge") else H_FEATURES


def b_shap(module_b, rows: pd.DataFrame) -> tuple:
    """SHAP values (log-ratio space) of the *default* forecast model, and its feature names.

    hybrid: TreeSHAP of the residual model + the physics baseline as its own feature;
    xgboost: TreeSHAP; ridge: exact linear SHAP coef * standardized feature;
    physics: all attribution on the physics baseline.
    """
    X = rows[B_FEATURES].to_numpy(float)
    phys = rows["r24"].to_numpy() * 7.0 ** rows["n_exp"].to_numpy()
    d = module_b.default
    s = module_b._scale(rows["Param_Name"])[:, None]
    if d == "xgboost":
        return s * tree_shap(module_b.m_xgb, X), B_FEATURES
    if d == "ridge":
        sc, rg = module_b.m_ridge.named_steps["standardscaler"], module_b.m_ridge.named_steps["ridge"]
        return sc.transform(X) * rg.coef_, B_FEATURES
    if d == "physics":
        return np.column_stack([np.zeros_like(X), phys]), H_FEATURES
    sv = s * tree_shap(module_b.m_hyb, np.column_stack([X, phys]))
    sv[:, -1] += phys
    return sv, H_FEATURES


def reason_codes(row, k_dpat: float) -> list:
    codes = []
    if row["spec_fail"]:
        codes.append("SPEC_FAIL")
    if row["delta_fail"]:
        codes.append(f"DELTA_FAIL_{row['delta_worst_t']}")
    for c in VALUE_COLS + ["Drift_0_168"]:
        if abs(row[f"dpat_z_{c}"]) >= k_dpat:
            codes.append("DPAT_" + c.replace("Value_", "").replace("Drift_0_168", "DRIFT"))
    if row["maha_score"] >= 1:
        codes.append("MAHALANOBIS")
    if row.get("vae_score", 0) >= 1:
        codes.append("VAE_RECONSTRUCTION")
    if row.get("spatial_score", 0) >= 1:
        codes.append("SPATIAL_ANOMALY")
    if row["if_score"] >= 1:
        codes.append("ISOLATION_FOREST")
    if row["B_up"] >= 1:
        codes.append(f"EARLY_DRIFT_UP[{row['binding_up']}]")
    if row["B_down"] >= 1:
        codes.append(f"EARLY_DRIFT_DOWN[{row['binding_lo']}]")
    if row["B_spec"] >= 1:
        codes.append("PRED_168H_OUT_OF_SPEC")
    return codes


def _fmt(v):
    if not np.isfinite(v):
        return "∞" if v > 0 else "-∞"
    return f"{v:.3g}" if abs(v) < 100 else f"{v:.1f}"


def _dpat_sentence(row, lot_stats, u):
    zc = {c: row[f"dpat_z_{c}"] for c in VALUE_COLS + ["Drift_0_168"]}
    worst = max(zc, key=lambda c: abs(zc[c]))
    st = lot_stats[(row["Lot_ID"], row["Param_Name"], worst)]
    if worst == "Drift_0_168":
        if st["log_scale"]:
            val = f"×{row['Value_168h'] / row['Value_0h']:.3g}"
            med, win = f"×{st['median']:.3g}", f"×{st['lower']:.3g}–×{st['upper']:.3g}"
        else:
            val = f"{row['Value_168h'] - row['Value_0h']:+.3g} {u}"
            med, win = f"{st['median']:+.3g} {u}", f"{_fmt(st['lower'])}–{_fmt(st['upper'])} {u}"
        return (f"0h→168h drift {val} is {zc[worst]:+.1f} robust-σ from lot median {med} "
                f"(DPAT window {win}).")
    scale = " (log scale)" if st["log_scale"] else ""
    return (f"{worst} = {_fmt(row[worst])} {u} is {zc[worst]:+.1f} robust-σ{scale} from lot median "
            f"{_fmt(st['median'])} {u} (DPAT window {_fmt(st['lower'])}–{_fmt(st['upper'])} {u}).")


def _drift_sentence(row, model_name, u):
    down = row["B_down"] > row["B_up"] and np.isfinite(row["B_down"])
    if down:
        s, safe, bind = row["slope_lower"], row["safety_slope_lower"], row["binding_lo"]
        cands = (row["safe_lower_spec"], row["safe_lower_lot"], row["safe_lower_mission"])
        agg, bound, cmp = "max", "lower", row["slope_lower"] < safe
    else:
        s, safe, bind = row["slope_upper"], row["safety_slope"], row["binding_up"]
        cands = (row["safe_slope_spec"], row["safe_slope_lot"], row["safe_slope_mission"])
        agg, bound, cmp = "min", "upper", row["slope_upper"] > safe
    verdict = "EXCEEDS" if cmp else "within"
    ratio = s / safe if safe not in (0, np.inf, -np.inf) else np.nan
    return (f"Drift (Module B, {model_name} model, 0h+24h only, n={row['n_exp']:.2f}): predicted 168h = "
            f"{_fmt(row['pred_168'])} {u} ({row['pi_pct']}% conformal band {_fmt(row['pred_168_lower'])}–"
            f"{_fmt(row['pred_168_upper'])} {u}; actual {_fmt(row['Value_168h'])} {u}); {bound}-bound slope "
            f"{s:+.4f} {u}/h vs safety slope {safe:+.4f} {u}/h = {agg}(spec-delta {_fmt(cands[0])}, "
            f"lot {_fmt(cands[1])}, mission-life {_fmt(cands[2])}) → binding limit: {LIMIT_NAMES[bind]} → "
            f"{verdict}" + (f" ({ratio:.1f}×)." if np.isfinite(ratio) else "."))


def explain_row(row, lot_stats: dict, model_name: str = "hybrid") -> str:
    """Plain-language justification for one (component, parameter) row."""
    u, p = row["Unit"], row["Param_Name"]
    parts = [f"{p}: " + _dpat_sentence(row, lot_stats, u)]
    peak = max(VALUE_COLS, key=lambda c: row[c])
    if row["spec_fail"]:
        parts.append(f"Datasheet limit EXCEEDED (peak {_fmt(row[peak])} {u} vs max {_fmt(row['Spec_Max'])} {u}).")
    else:
        parts.append(f"All readings within datasheet max {_fmt(row['Spec_Max'])} {u} (peak {_fmt(row[peak])} {u} "
                     f"at {peak.replace('Value_', '')}).")
    if np.isfinite(row["delta_allow"]):
        state = "DELTA FAIL" if row["delta_fail"] else "within delta limit"
        parts.append(f"Largest Δ vs 0h: {row['delta_worst']:+.3g} {u} at {row['delta_worst_t']} vs allowed "
                     f"±{_fmt(row['delta_allow'])} {u} (max({100 * row['Delta_Pct']:.0f}% of initial, "
                     f"{_fmt(row['Delta_Floor'])} {u})) → {state}.")
    if not row["spec_fail"] and not row["delta_fail"]:
        parts.append("→ latent / lot-relative anomaly that static and delta limits would miss.")
    mc = np.array([row[f"mc_{f}"] for f in A_FEATURES])
    if row["maha_d2"] > 0:
        j = int(np.argmax(mc))
        parts.append(f"Top Mahalanobis contributor: {FEATURE_LABELS[A_FEATURES[j]]} "
                     f"({100 * mc[j] / row['maha_d2']:.0f}% of D², score {row['maha_score']:.1f}).")
    if isinstance(row.get("if_shap_top"), str):
        parts.append(f"Top Isolation-Forest SHAP driver: {row['if_shap_top']}.")
    txt = _drift_sentence(row, model_name, u)
    if isinstance(row.get("b_shap_top"), str):
        txt += f" Main SHAP driver of the forecast: {row['b_shap_top']}."
    parts.append(txt)
    return " ".join(parts)


def attach_shap(rows: pd.DataFrame, mask: np.ndarray, module_a, module_b) -> pd.DataFrame:
    """SHAP vectors + top drivers, computed only for rows that need an explanation."""
    rows = rows.copy()
    feats_b = b_features(module_b)
    new = {"if_shap_top": None, "b_shap_top": None}
    new.update({f"shapA_{f}": np.nan for f in A_FEATURES})
    new.update({f"shapB_{f}": np.nan for f in feats_b})
    rows = rows.assign(**new)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return rows
    sub = rows.iloc[idx]
    for p, pidx in sub.groupby("Param_Name").indices.items():
        if p in module_a.iforests:
            sv = iforest_shap(module_a.iforests[p], sub.iloc[pidx][A_FEATURES].to_numpy(float))
            ix = sub.index[pidx]
            rows.loc[ix, [f"shapA_{f}" for f in A_FEATURES]] = sv
            rows.loc[ix, "if_shap_top"] = [FEATURE_LABELS[A_FEATURES[int(np.argmax(v))]] for v in sv]
    svb, _ = b_shap(module_b, sub)
    rows.loc[sub.index, [f"shapB_{f}" for f in feats_b]] = svb
    rows.loc[sub.index, "b_shap_top"] = [FEATURE_LABELS[feats_b[int(np.argmax(np.abs(v)))]] for v in svb]
    return rows

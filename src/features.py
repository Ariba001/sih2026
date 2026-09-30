"""Lot-localized feature engineering.

All anomaly features are robust z-scores computed *within* each (Lot_ID, Param_Name) group,
so a part is always judged against its own lot population (contextual outlier detection).
"""
import numpy as np
import pandas as pd

from .config import TIMEPOINTS, VALUE_COLS

GROUP = ["Lot_ID", "Param_Name"]
MAD_TO_SIGMA = 1.4826

# Module A (anomaly) features, all lot-normalized robust z-scores.
A_FEATURES = ["z_0h", "z_24h", "z_96h", "z_168h",
              "zd_0_24", "zd_24_96", "zd_96_168", "zd_0_168", "z_accel"]

FEATURE_LABELS = {
    "z_0h": "Value_0h lot robust z-score",
    "z_24h": "Value_24h lot robust z-score",
    "z_96h": "Value_96h lot robust z-score",
    "z_168h": "Value_168h lot robust z-score",
    "zd_0_24": "0h→24h drift (lot z-score)",
    "zd_24_96": "24h→96h drift (lot z-score)",
    "zd_96_168": "96h→168h drift (lot z-score)",
    "zd_0_168": "total 0h→168h drift (lot z-score)",
    "z_accel": "drift acceleration (late vs early rate, lot z-score)",
    # Module B
    "log_v0": "log Value_0h", "log_v24": "log Value_24h", "r24": "log ratio 24h/0h",
    "lot_med_log_v0": "lot median log Value_0h", "lot_med_log_v24": "lot median log Value_24h",
    "lot_med_r24": "lot median 24h/0h ratio", "rel_v0": "Value_0h vs lot median",
    "rel_r24": "0h→24h drift vs lot median drift",
    "is_Iddq": "param=Iddq", "is_Leakage": "param=Leakage", "is_Prop_Delay": "param=Prop_Delay",
}

B_PARAMS = ["Iddq", "Leakage", "Prop_Delay"]
B_FEATURES = ["log_v0", "log_v24", "r24", "lot_med_log_v0", "lot_med_log_v24", "lot_med_r24",
              "rel_v0", "rel_r24"] + [f"is_{p}" for p in B_PARAMS]


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    s = MAD_TO_SIGMA * np.median(np.abs(x - med))
    if s <= 0:
        q75, q25 = np.percentile(x, [75, 25])
        s = (q75 - q25) / 1.349
    return float(max(s, 1e-9 + 1e-6 * abs(med)))


def _robust_z(s: pd.Series) -> pd.Series:
    return (s - s.median()) / robust_sigma(s.values)


def lot_z(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(GROUP)[col].transform(_robust_z)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with Module A and Module B features appended (row = component x param)."""
    out = df.copy()
    logs = {t: np.log(out[f"Value_{t}h"]) for t in TIMEPOINTS}
    for t in TIMEPOINTS:
        out[f"_log_{t}"] = logs[t]
    out["_d_0_24"] = logs[24] - logs[0]
    out["_d_24_96"] = logs[96] - logs[24]
    out["_d_96_168"] = logs[168] - logs[96]
    out["_d_0_168"] = logs[168] - logs[0]
    out["_accel"] = out["_d_96_168"] / 72 - out["_d_0_24"] / 24
    # NEW FEATURE: Divergence of drift (late vs early) specifically to target late-bloomers
    out["_divergence"] = out["_d_96_168"] - out["_d_24_96"]

    for t in TIMEPOINTS:
        out[f"z_{t}h"] = lot_z(out, f"_log_{t}")
    for a, b in [("0", "24"), ("24", "96"), ("96", "168"), ("0", "168")]:
        out[f"zd_{a}_{b}"] = lot_z(out, f"_d_{a}_{b}")
    out["z_accel"] = lot_z(out, "_accel")
    out["z_divergence"] = lot_z(out, "_divergence")

    # Module B features: only 0h and 24h information + lot context at 0h/24h.
    g = out.groupby(GROUP)
    out["log_v0"], out["log_v24"], out["r24"] = logs[0], logs[24], out["_d_0_24"]
    out["lot_med_log_v0"] = g["_log_0"].transform("median")
    out["lot_med_log_v24"] = g["_log_24"].transform("median")
    out["lot_med_r24"] = g["_d_0_24"].transform("median")
    out["rel_v0"] = out["log_v0"] - out["lot_med_log_v0"]
    out["rel_r24"] = out["r24"] - out["lot_med_r24"]
    for p in B_PARAMS:
        out[f"is_{p}"] = (out["Param_Name"] == p).astype(float)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def _log_scale(df: pd.DataFrame) -> np.ndarray:
    return df["Log_Scale"].to_numpy(bool) if "Log_Scale" in df else np.zeros(len(df), bool)


def dpat_measures(df: pd.DataFrame) -> dict:
    """DPAT measures per row: values and 0->168h drift, log-transformed for log-scale parameters."""
    ls = _log_scale(df)
    out = {c: pd.Series(np.where(ls, np.log(df[c]), df[c]), index=df.index) for c in VALUE_COLS}
    out["Drift_0_168"] = pd.Series(np.where(ls, np.log(df["Value_168h"] / df["Value_0h"]),
                                            df["Value_168h"] - df["Value_0h"]), index=df.index)
    return out


def dpat_limits(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Dynamic PAT limits per (lot, param, measure): median +/- k robust sigma, reported in native units
    (log-scale parameters: exp of the log-space window; drift as a ratio V168/V0)."""
    meas = dpat_measures(df)
    ls = _log_scale(df)
    recs = []
    for (lot, p), idx in df.groupby(GROUP).indices.items():
        is_log = bool(ls[idx[0]])
        for c, s in meas.items():
            x = s.values[idx]
            med, sig = float(np.median(x)), robust_sigma(x)
            lo, hi = med - k * sig, med + k * sig
            if is_log:
                med, lo, hi = np.exp(med), np.exp(lo), np.exp(hi)
            recs.append({"Lot_ID": lot, "Param_Name": p, "Measure": c, "log_scale": is_log,
                         "median": med, "sigma": sig, "lower": lo, "upper": hi})
    return pd.DataFrame(recs)

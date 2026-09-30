"""Asymmetric (escape-averse) scoring, cost-optimal thresholding and regression metrics."""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {"tp": int((y_true & y_pred).sum()), "fp": int((~y_true & y_pred).sum()),
            "fn": int((y_true & ~y_pred).sum()), "tn": int((~y_true & ~y_pred).sum())}


def cost(cm: dict, c_fn: float, c_fp: float) -> float:
    return c_fn * cm["fn"] + c_fp * cm["fp"]


def fbeta(precision, recall, beta):
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def clopper_pearson_upper(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact upper bound on a binomial rate (0 of n -> 1 - (1-conf)^(1/n) ~ 3/n)."""
    if n == 0:
        return 1.0
    return 1.0 if x >= n else float(beta_dist.ppf(conf, x + 1, n - x))


def classification_report(y_true, y_pred, c_fn, c_fp, beta=2.0) -> dict:
    cm = confusion(y_true, y_pred)
    n_pos = cm["tp"] + cm["fn"]
    recall = cm["tp"] / max(n_pos, 1)
    precision = cm["tp"] / max(cm["tp"] + cm["fp"], 1)
    return {"recall": recall, "precision": precision, f"f{beta:g}": fbeta(precision, recall, beta),
            "cost": cost(cm, c_fn, c_fp), "escapes": cm["fn"], "false_rejects": cm["fp"],
            "overkill_rate": cm["fp"] / max(cm["fp"] + cm["tn"], 1),
            "escape_rate_ub95": clopper_pearson_upper(cm["fn"], n_pos),
            "n": len(y_true), "n_defective": int(np.sum(y_true)), "confusion_matrix": cm}


def recall_overkill(y_true, y_pred) -> tuple:
    cm = confusion(y_true, y_pred)
    return cm["tp"] / max(cm["tp"] + cm["fn"], 1), cm["fp"] / max(cm["fp"] + cm["tn"], 1), cm["fn"]


def dpat_k_curve(rows: pd.DataFrame, comp_labels: pd.DataFrame, ks) -> list:
    """Recall vs overkill over DPAT k with no tuning: (a) rules + DPAT only, (b) full Module A at k."""
    y = comp_labels.set_index("Component_ID")["Is_Defective"]
    out = []
    for k in ks:
        a_row = np.maximum.reduce([rows["dpat_maxz"] / k, rows["maha_score"], rows["if_score"]])
        tmp = rows[["Component_ID"]].assign(
            dpat=rows["rule_fail"] | (rows["dpat_maxz"] >= k), mod_a=rows["rule_fail"] | (a_row >= 1))
        g = tmp.groupby("Component_ID")[["dpat", "mod_a"]].any().reindex(y.index)
        r1, o1, e1 = recall_overkill(y, g["dpat"])
        r2, o2, e2 = recall_overkill(y, g["mod_a"])
        out.append({"k": float(k), "dpat_recall": r1, "dpat_overkill": o1, "dpat_escapes": e1,
                    "moduleA_recall": r2, "moduleA_overkill": o2, "moduleA_escapes": e2})
    return out


def choose_threshold(scores, y_true, c_fn, c_fp) -> dict:
    """Threshold minimising C_FN*FN + C_FP*FP for the rule `score >= threshold`.

    Candidate thresholds are midpoints between consecutive distinct scores (margin on both
    sides). Ties in cost are broken towards the lower threshold (recall-first).
    """
    s, y = np.asarray(scores, float), np.asarray(y_true).astype(bool)
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
    order = np.argsort(s)
    s_sorted, y_sorted = s[order], y[order]
    # for threshold t: FN = defects with score < t, FP = goods with score >= t
    pos_cum = np.concatenate([[0], np.cumsum(y_sorted)])
    neg_cum = np.concatenate([[0], np.cumsum(~y_sorted)])
    k = np.searchsorted(s_sorted, cands, side="left")
    fn = pos_cum[k]
    fp = neg_cum[-1] - neg_cum[k]
    costs = c_fn * fn + c_fp * fp
    best = int(np.flatnonzero(costs == costs.min())[0])
    return {"threshold": float(cands[best]), "val_cost": float(costs[best]),
            "val_fn": int(fn[best]), "val_fp": int(fp[best])}


def choose_threshold_pair(a_scores, b_scores, y_true, c_fn, c_fp, n_grid=200) -> dict:
    """Jointly choose (tau_a, tau_b) for the OR-rule `A >= tau_a or B >= tau_b` minimising cost."""
    a, b = np.asarray(a_scores, float), np.asarray(b_scores, float)
    y = np.asarray(y_true).astype(bool)
    grid = np.unique(np.quantile(b, np.linspace(0.5, 1.0, n_grid)))
    grid = np.r_[grid[grid > 0.05] + 1e-9, np.inf]
    best = None
    for tb in grid:
        caught = b >= tb
        fp_b = int((caught & ~y).sum())
        rest = ~caught
        if rest.sum() == 0 or y[rest].sum() == 0 and (~y[rest]).sum() == 0:
            continue
        ta = choose_threshold(a[rest], y[rest], c_fn, c_fp)
        total = ta["val_cost"] + c_fp * fp_b
        if best is None or total < best["val_cost"]:
            best = {"tau_a": max(ta["threshold"], 1e-3), "tau_b": float(tb), "val_cost": float(total),
                    "val_fn": ta["val_fn"], "val_fp": ta["val_fp"] + fp_b}
    return best


def mae_report(df: pd.DataFrame, pred_cols: dict) -> dict:
    """MAE of predicted vs actual Value_168h, overall and per Param_Name, for each model.

    Overall MAE mixes units (µA, ns); normalized MAE (MAE / mean actual) is also reported.
    """
    out = {}
    y = df["Value_168h"].to_numpy()
    drift = np.abs(np.log(y / df["Value_0h"].to_numpy()))
    tail = np.zeros(len(df), bool)  # top 5 % drifters per parameter (the safety-relevant tail)
    for _, idx in df.groupby("Param_Name").indices.items():
        tail[idx] = drift[idx] >= np.quantile(drift[idx], 0.95)
    for name, col in pred_cols.items():
        err = np.abs(df[col].to_numpy() - y)
        rep = {"overall_mae": float(err.mean()), "overall_nmae_pct": float(100 * (err / y).mean()),
               "tail_nmae_pct": float(100 * (err[tail] / y[tail]).mean()), "per_param": {}}
        for p, idx in df.groupby("Param_Name").indices.items():
            rep["per_param"][p] = {"mae": float(err[idx].mean()),
                                   "nmae_pct": float(100 * (err[idx] / y[idx]).mean()),
                                   "tail_mae": float(err[idx][tail[idx]].mean())}
        out[name] = rep
    return out

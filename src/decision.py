"""Combined decision engine: ACCEPT / REVIEW / REJECT per component with justification.

REJECT  if a rule screen fails (datasheet limit or delta limit - never waivable),
        or Module A score >= tau_A, or Module B score >= tau_B (cost-optimal, tuned on validation lots)
REVIEW  if the combined score (in multiples of the thresholds) >= review_frac
ACCEPT  otherwise
"""
import numpy as np
import pandas as pd

from .explain import explain_row, reason_codes

HARD_FAIL = 1e6


def row_scores(rows: pd.DataFrame, tau_a: float = 1.0, tau_b: float = 1.0) -> pd.DataFrame:
    rows = rows.copy()
    rows["rule_fail"] = rows["spec_fail"] | rows["delta_fail"]
    rows["A_score_eff"] = np.where(rows["rule_fail"], HARD_FAIL, rows["A_score"])
    rows["C_score"] = np.maximum(rows["A_score_eff"] / tau_a, rows["B_score"] / tau_b)
    return rows


def component_scores(rows: pd.DataFrame) -> pd.DataFrame:
    g = rows.groupby(["Lot_ID", "Component_ID"])
    return pd.DataFrame({
        "A_score": g["A_score_eff"].max(), "A_raw": g["A_score"].max(), "B_score": g["B_score"].max(),
        "C_score": g["C_score"].max(), "spec_fail": g["spec_fail"].any(), "delta_fail": g["delta_fail"].any(),
        "rule_fail": g["rule_fail"].any(), "B_flag": g["B_flag"].any(),
    }).reset_index()


def decide(comp: pd.DataFrame, review_frac: float) -> pd.Series:
    return pd.Series(np.select(
        [comp["rule_fail"] | (comp["C_score"] >= 1.0), comp["C_score"] >= review_frac],
        ["REJECT", "REVIEW"], "ACCEPT"), index=comp.index)


def build_explanations(rows, comp, lot_stats, k_dpat, tau_a, tau_b, model_name, max_params=2):
    """Attach reason codes, primary parameter and QA text to every non-ACCEPT component."""
    comp = comp.copy()
    comp["Primary_Param"], comp["Reason_Codes"], comp["Explanation"] = "", "", ""
    flagged = comp.index[comp["Decision"] != "ACCEPT"]
    by_comp = {k: g for k, g in rows.groupby("Component_ID")}
    for i in flagged:
        c = comp.loc[i]
        g = by_comp[c["Component_ID"]].sort_values("C_score", ascending=False)
        codes, texts = [], []
        for _, r in g.head(max_params).iterrows():
            rc = reason_codes(r, k_dpat)
            if not rc and texts:
                continue
            codes += [f"{r['Param_Name']}:{x}" for x in rc]
            texts.append(explain_row(r, lot_stats, model_name))
        rules = [n for n, f in [("datasheet failure", c["spec_fail"]), ("delta-limit failure", c["delta_fail"])] if f]
        score_txt = "; ".join(rules) if rules else f"combined score {c['C_score']:.2f}× threshold"
        head = (f"Component {c['Component_ID']} {c['Decision']} ({score_txt}; Module A lot-outlier score "
                f"{c['A_raw']:.2f} vs τA={tau_a:.2f}, Module B drift score {c['B_score']:.2f} vs τB={tau_b:.2f}).")
        comp.at[i, "Primary_Param"] = g.iloc[0]["Param_Name"]
        comp.at[i, "Reason_Codes"] = ";".join(codes) or "SCORE_NEAR_THRESHOLD"
        comp.at[i, "Explanation"] = head + " " + " | ".join(texts)
    return comp

"""Layer 0 rule-based screens in MIL-STD-883 / ESCC language (docs/RESEARCH.md §1.5).

* SPEC_FAIL  : any reading outside the datasheet limits
* DELTA_FAIL : |Value_t - Value_0h| > max(Delta_Pct * |Value_0h|, Delta_Floor) at 24/96/168h
* PDA        : lot-level percent defective allowable = (spec + delta failures) / parts; lot fails > 5 %
The AI layers only ever *add* rejects; these rules can never be waived.
"""
import numpy as np
import pandas as pd

from .config import VALUE_COLS

PDA_LIMIT = 0.05


def rule_screens(feat: pd.DataFrame) -> pd.DataFrame:
    v = feat[VALUE_COLS].to_numpy(float)
    res = pd.DataFrame(index=feat.index)
    res["spec_fail"] = ((v > feat["Spec_Max"].to_numpy()[:, None]) |
                        (v < feat["Spec_Min"].to_numpy()[:, None])).any(axis=1)
    v0 = v[:, 0]
    allow = np.maximum(feat["Delta_Pct"].to_numpy() * np.abs(v0), feat["Delta_Floor"].to_numpy())
    deltas = v[:, 1:] - v0[:, None]
    ratio = np.abs(deltas) / allow[:, None]
    res["delta_allow"] = allow
    res["delta_ratio"] = ratio.max(axis=1)
    res["delta_worst_t"] = np.array(["24h", "96h", "168h"])[ratio.argmax(axis=1)]
    res["delta_worst"] = deltas[np.arange(len(v)), ratio.argmax(axis=1)]
    res["delta_fail"] = res["delta_ratio"] > 1.0
    return res


def lot_pda(comp: pd.DataFrame) -> pd.DataFrame:
    """Percent defective allowable per lot from rule failures (spec + delta), as in TM 5004 / ESCC 9000."""
    g = comp.groupby("Lot_ID")
    out = pd.DataFrame({"parts": g.size(), "rule_failures": g["rule_fail"].sum(),
                        "ai_rejects": g["Decision"].apply(lambda s: int((s == "REJECT").sum()))})
    out["PDA_pct"] = 100 * out["rule_failures"] / out["parts"]
    out["Lot_Status"] = np.where(out["PDA_pct"] > 100 * PDA_LIMIT, "LOT REJECT (PDA > 5%)", "PDA OK")
    return out.reset_index()

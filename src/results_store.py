"""Helpers to build / persist BurnTestr dashboard result payloads."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = ROOT / "reports" / "current_results.json"
DECISIONS_CSV = ROOT / "reports" / "test_decisions.csv"
SYNTHETIC_CSV = ROOT / "data" / "burnin_synthetic.csv"

# Operator-facing decision definitions (also mirrored in the dashboard UI).
DECISION_GUIDE = {
    "ACCEPT": {
        "label": "Accept",
        "short": "Cleared for use",
        "meaning": (
            "The component stays inside lot-relative anomaly and early-drift thresholds. "
            "Static datasheet and delta limits also pass."
        ),
        "action": (
            "Release with the lot. No hold is required; keep the scores in the quality "
            "record for traceability."
        ),
    },
    "REVIEW": {
        "label": "Review",
        "short": "Engineering hold",
        "meaning": (
            "Scores sit in the gray band below hard reject but above the accept floor — "
            "a borderline lot-outlier or drift signal."
        ),
        "action": (
            "Hold the part for engineering review. Compare Module A/B drivers, re-measure "
            "if needed, then accept or reject with a signed disposition."
        ),
    },
    "REJECT": {
        "label": "Reject",
        "short": "Do not ship",
        "meaning": (
            "A hard rule failed (datasheet/delta) or the combined anomaly/drift score "
            "crossed the reject threshold. These calls are not waivable by default."
        ),
        "action": (
            "Quarantine from the flight lot. Route to MRB only under formal process; "
            "do not mix with accepted inventory."
        ),
    },
}


def _pct(n: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{round(100.0 * n / total)}%"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def _clean_text(val) -> str:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        s = str(val).strip()
    except Exception:
        return ""
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _confidence(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return float(max(0.0, min(1.0, 1.0 - abs(a - b) / denom)))


def describe_decision(
    decision: str,
    a: float,
    b: float,
    reason_codes: str = "",
    explanation: str = "",
) -> str:
    """Human-readable screening description for a component row.

    Always produces clear operator copy from the decision class and scores.
    Full pipeline ``Explanation`` text (when present) is stored separately on
    the payload as ``explanation`` — it is not required for the primary blurb.
    """
    del explanation  # kept in signature for call-site compatibility

    d = str(decision or "ACCEPT").strip().upper()
    if d == "ACCEPTED":
        d = "ACCEPT"
    if d == "REJECTED":
        d = "REJECT"

    guide = DECISION_GUIDE.get(d, {})
    codes = _clean_text(reason_codes)
    if codes:
        codes = codes.replace(";", ", ")

    if d == "ACCEPT":
        text = (
            f"Within lot-relative and drift thresholds (Module A {a:.2f}, Module B {b:.2f}). "
            "Ship-ready for the screened application; no operator hold required."
        )
    elif d == "REVIEW":
        text = (
            f"Near the decision boundary (Module A {a:.2f}, Module B {b:.2f}). "
            + guide.get(
                "action",
                "Hold for engineering review before disposition.",
            )
        )
    elif d == "REJECT":
        if a >= 1e5:
            text = (
                "Hard rule failure (datasheet limit or delta limit). "
                "Quarantine — do not waive without formal MRB."
            )
        else:
            text = (
                f"Above reject threshold (Module A {a:.2f}, Module B {b:.2f}). "
                + guide.get(
                    "action",
                    "Quarantine from the flight lot.",
                )
            )
    else:
        text = guide.get("meaning", f"Screening decision: {d}.")

    if codes:
        text = f"{text} Triggers: {codes}."
    return text


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add decision_guide + per-row description fields without removing existing keys."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    out.setdefault("decision_guide", DECISION_GUIDE)
    comps = []
    for item in out.get("components") or []:
        row = dict(item)
        a = _safe_float(row.get("score_a", 0.0))
        b = _safe_float(row.get("score_b", 0.0))
        explanation = _clean_text(row.get("explanation"))
        reason = _clean_text(row.get("reason_codes"))
        # Always refresh operator-facing description so copy stays consistent
        row["description"] = describe_decision(
            row.get("decision", "ACCEPT"),
            a,
            b,
            reason_codes=reason,
            explanation=explanation,
        )
        if explanation:
            row["explanation"] = explanation
        comps.append(row)
    if comps:
        out["components"] = comps
    return out


def build_payload_from_comp(
    comp: pd.DataFrame,
    source_df: Optional[pd.DataFrame] = None,
    *,
    source: str = "scored",
) -> dict[str, Any]:
    """Normalize a component-level decisions frame into the dashboard JSON shape."""
    if "Decision" not in comp.columns:
        raise ValueError("Component frame missing Decision column")

    decisions = comp["Decision"].astype(str).str.upper()
    total = int(len(comp))
    accepted = int((decisions == "ACCEPT").sum())
    reviewed = int((decisions == "REVIEW").sum())
    rejected = int((decisions == "REJECT").sum())

    # Decisions by lot
    decisions_by_lot = []
    if "Lot_ID" in comp.columns:
        for lot_id, g in comp.groupby("Lot_ID", sort=True):
            d = g["Decision"].astype(str).str.upper()
            decisions_by_lot.append({
                "lot_id": str(lot_id),
                "accepted": int((d == "ACCEPT").sum()),
                "review": int((d == "REVIEW").sum()),
                "rejected": int((d == "REJECT").sum()),
            })

    # Parameter stats from source measurements when available
    parameter_stats = []
    if source_df is not None and "Param_Name" in source_df.columns and "Value_168h" in source_df.columns:
        for param, g in source_df.groupby("Param_Name", sort=True):
            parameter_stats.append({
                "parameter": str(param),
                "mean": _safe_float(g["Value_168h"].mean()),
                "min": _safe_float(g["Value_168h"].min()),
                "max": _safe_float(g["Value_168h"].max()),
            })

    components = []
    recent_scores = []
    for _, row in comp.iterrows():
        a = _safe_float(row.get("A_score", 0.0))
        b = _safe_float(row.get("B_score", 0.0))
        decision = str(row.get("Decision", "ACCEPT")).upper()
        conf = _confidence(a, b)
        reason_codes = _clean_text(row.get("Reason_Codes", ""))
        explanation = _clean_text(row.get("Explanation", ""))
        description = describe_decision(
            decision, a, b, reason_codes=reason_codes, explanation=explanation
        )
        item = {
            "id": str(row.get("Component_ID", "")),
            "component_id": str(row.get("Component_ID", "")),
            "lot_id": str(row.get("Lot_ID", "")),
            "decision": decision.title() if decision in {"ACCEPT", "REVIEW", "REJECT"} else decision,
            "score_a": a,
            "score_b": b,
            "confidence": conf,
            "survival_prob": _safe_float(row.get("survival_prob", row.get("S_proba", 0.99)), 0.99),
            "mission_risk": str(row.get("mission_risk", "LOW")),
            "reason_codes": reason_codes,
            "explanation": explanation,
            "description": description,
        }
        components.append(item)
        recent_scores.append({
            "component_id": item["component_id"],
            "lot_id": item["lot_id"],
            "decision": item["decision"],
            "score_a": a,
            "score_b": b,
            "confidence": conf,
            "description": description,
        })

    # Prefer flagged components first in "recent"
    order = {"Reject": 0, "Review": 1, "Accept": 2}
    recent_scores.sort(key=lambda r: (order.get(r["decision"], 9), -r["score_a"]))

    return {
        "meta": {
            "product": "BurnTestr",
            "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "component_count": total,
        },
        "summary": {
            "total_components": total,
            "accepted_count": accepted,
            "accepted_pct": _pct(accepted, total),
            "review_count": reviewed,
            "review_pct": _pct(reviewed, total),
            "rejected_count": rejected,
            "rejected_pct": _pct(rejected, total),
            # Upload response aliases
            "total": total,
            "accept": accepted,
            "review": reviewed,
            "reject": rejected,
        },
        "decision_guide": DECISION_GUIDE,
        "decisions_by_lot": decisions_by_lot,
        "parameter_stats": parameter_stats,
        "parameter_insights": parameter_stats,
        "recent_scores": recent_scores[:50],
        "components": components,
    }


def load_current_results() -> Optional[dict[str, Any]]:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return enrich_payload(json.load(f))
    return None


def save_current_results(payload: dict[str, Any]) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def persist_scored_results(
    comp: pd.DataFrame,
    source_df: Optional[pd.DataFrame] = None,
    *,
    source: str = "streamlit",
) -> dict[str, Any]:
    """Persist scored component decisions for FastAPI / Vite dashboard consumers."""
    payload = build_payload_from_comp(comp, source_df, source=source)
    save_current_results(payload)
    return payload


def load_from_decisions_csv(path: Path = DECISIONS_CSV) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return build_payload_from_comp(df, source="test_decisions.csv")


def ensure_seed_results(system=None, max_lots: int = 4) -> dict[str, Any]:
    """Return persisted results, or seed from CSV / a small live score run."""
    existing = load_current_results()
    if existing:
        return existing

    seeded = load_from_decisions_csv()
    if seeded:
        save_current_results(seeded)
        return seeded

    if system is not None and SYNTHETIC_CSV.exists():
        from src.data import load_csv

        df = load_csv(SYNTHETIC_CSV)
        lots = sorted(df["Lot_ID"].unique())[:max_lots]
        sample = df[df["Lot_ID"].isin(lots)].copy()
        _, comp = system.score(sample, explain=False)
        payload = build_payload_from_comp(comp, sample, source="synthetic_sample")
        save_current_results(payload)
        return payload

    # Empty shell so UI still renders
    empty = build_payload_from_comp(
        pd.DataFrame(columns=["Lot_ID", "Component_ID", "Decision", "A_score", "B_score"]),
        source="empty",
    )
    return empty

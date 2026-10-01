"""BurnTestr FastAPI — scoring, CSV upload, and current-results dashboard APIs."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data import validate
from src.results_store import (
    build_payload_from_comp,
    enrich_payload,
    ensure_seed_results,
    load_current_results,
    save_current_results,
)

MODEL_PATH = Path(__file__).parent / "models" / "burnin_system.joblib"
system = None
llm_gen = None

app = FastAPI(
    title="BurnTestr API",
    description="Aerospace burn-in anomaly detection for component screening",
    version="2.1",
)


def _cors_origins() -> list[str]:
    defaults = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    extra = os.getenv("CORS_ORIGINS", "").strip()
    if not extra:
        return defaults
    return defaults + [o.strip() for o in extra.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================ Models

class ParameterInput(BaseModel):
    param_name: str = Field(..., description="Iddq, Leakage, or Prop_Delay")
    values: List[float] = Field(..., description="Value_0h, Value_24h, Value_96h, Value_168h")


class ComponentInput(BaseModel):
    component_id: str
    parameters: List[ParameterInput]
    die_x: Optional[float] = None
    die_y: Optional[float] = None
    wafer_id: Optional[str] = None


class ScoringRequest(BaseModel):
    lot_id: str
    components: List[ComponentInput]
    user_id: Optional[str] = None
    notes: Optional[str] = None


class ComponentScore(BaseModel):
    component_id: str
    decision: str
    a_score: float
    b_score: float
    survival_prob: float
    mission_risk: str
    reasoning: List[str]
    llm_report: Optional[str] = None


class LotSummary(BaseModel):
    total_components: int
    rejected: int
    reviewed: int
    accepted: int
    overkill_rate: float
    mean_survival_prob: float
    lot_summary_report: Optional[str] = None


class ScoringResponse(BaseModel):
    lot_id: str
    timestamp: str
    results: List[ComponentScore]
    lot_summary: LotSummary


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    product: str = "BurnTestr"
    timestamp: str


# ================================================================ Startup

@app.on_event("startup")
async def startup():
    global system, llm_gen
    try:
        import joblib

        if MODEL_PATH.exists():
            system = joblib.load(MODEL_PATH)
            print(f"[OK] BurnTestr model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] Model not found at {MODEL_PATH}")
        try:
            from src.llm_reports import LLMReportGenerator

            llm_gen = LLMReportGenerator()
        except Exception as llm_err:
            llm_gen = None
            print(f"[WARN] LLM reports disabled: {llm_err}")
        # Seed dashboard current results from reports/test_decisions.csv when available
        ensure_seed_results(system)
        print("[OK] Current results ready")
    except Exception as e:
        print(f"[ERROR] Startup error: {e}")


def _attach_survival(comp: pd.DataFrame) -> pd.DataFrame:
    """Ensure survival columns exist even if pipeline assignment misaligned."""
    out = comp.copy()
    if "survival_prob" not in out.columns or out["survival_prob"].isna().all():
        if "S_proba" in out.columns:
            out["survival_prob"] = out["S_proba"].fillna(0.99)
        else:
            out["survival_prob"] = 0.99
    out["survival_prob"] = pd.to_numeric(out["survival_prob"], errors="coerce").fillna(0.99)
    if "mission_risk" not in out.columns:
        out["mission_risk"] = out["survival_prob"].apply(
            lambda p: "HIGH" if p < 0.99 else ("MEDIUM" if p < 0.999 else "LOW")
        )
    return out


def _score_dataframe(df: pd.DataFrame, explain: bool = False):
    if not system:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = validate(df)
    try:
        rows_scored, comp_scored = system.score(df, explain=explain)
    except ValueError as e:
        # Common cause: survival_prob length mismatch — retry without weibull attach by patching
        raise HTTPException(status_code=400, detail=f"Scoring error: {e}") from e
    except Exception as e:
        # Fallback: score_rows + component decide without Weibull if full score fails
        try:
            from src.decision import decide
            from src.pipeline import component_scores

            rows_scored = system.score_rows(df)
            comp_scored = component_scores(rows_scored)
            comp_scored["Decision"] = decide(comp_scored, system.cfg.review_frac)
            comp_scored["ModuleA_Reject"] = comp_scored["rule_fail"] | (
                comp_scored["A_score"] >= (system.tau_a or 1.0)
            )
            print(f"[WARN] Full score failed ({e}); used score_rows fallback")
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Scoring error: {e}; fallback: {e2}") from e2
    return rows_scored, _attach_survival(comp_scored)


# ================================================================ Core endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if system else "degraded",
        model_version="2.1",
        model_loaded=system is not None,
        product="BurnTestr",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/score", response_model=ScoringResponse)
async def score_components(request: ScoringRequest):
    if not system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = []
    for comp in request.components:
        for param in comp.parameters:
            if len(param.values) != 4:
                raise HTTPException(status_code=400, detail=f"Expected 4 values for {param.param_name}")
            rows.append({
                "Lot_ID": request.lot_id,
                "Component_ID": comp.component_id,
                "Param_Name": param.param_name,
                "Value_0h": param.values[0],
                "Value_24h": param.values[1],
                "Value_96h": param.values[2],
                "Value_168h": param.values[3],
                "die_x": comp.die_x,
                "die_y": comp.die_y,
                "wafer_id": comp.wafer_id,
            })

    df = pd.DataFrame(rows)
    _, comp_scored = _score_dataframe(df, explain=False)

    results = []
    for _, row in comp_scored.iterrows():
        reasons = row.get("Reason_Codes", "")
        if isinstance(reasons, str) and reasons.strip():
            reasoning = [r.strip() for r in reasons.split(",") if r.strip()]
        else:
            reasoning = ["see explanation"]
        results.append(ComponentScore(
            component_id=str(row["Component_ID"]),
            decision=str(row["Decision"]),
            a_score=float(row["A_score"]),
            b_score=float(row["B_score"]),
            survival_prob=float(row.get("survival_prob", 0.99)),
            mission_risk=str(row.get("mission_risk", "LOW")),
            reasoning=reasoning,
            llm_report=None,
        ))

    accepted = int((comp_scored["Decision"] == "ACCEPT").sum())
    reviewed = int((comp_scored["Decision"] == "REVIEW").sum())
    rejected = int((comp_scored["Decision"] == "REJECT").sum())
    total = len(comp_scored)

    lot_summary = LotSummary(
        total_components=total,
        rejected=rejected,
        reviewed=reviewed,
        accepted=accepted,
        overkill_rate=float((total - accepted) / total) if total else 0.0,
        mean_survival_prob=float(comp_scored["survival_prob"].mean()) if total else 0.99,
        lot_summary_report=None,
    )

    payload = build_payload_from_comp(comp_scored, df, source=f"api_score:{request.lot_id}")
    save_current_results(payload)

    return ScoringResponse(
        lot_id=request.lot_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=results,
        lot_summary=lot_summary,
    )


@app.get("/lot/{lot_id}")
async def get_lot_results(lot_id: str):
    data = load_current_results() or ensure_seed_results(system)
    comps = [c for c in data.get("components", []) if str(c.get("lot_id")) == lot_id]
    if not comps:
        raise HTTPException(status_code=404, detail=f"No results for lot {lot_id}")
    return {"lot_id": lot_id, "components": comps, "count": len(comps)}


@app.get("/version")
async def get_version():
    return {
        "product": "BurnTestr",
        "version": "2.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ================================================================ Dashboard APIs

@app.get("/api/health")
async def api_health():
    return await health_check()


@app.get("/api/dashboard")
@app.get("/api/results")
async def get_dashboard():
    """Current / existing BurnTestr results for charts and tables."""
    try:
        data = load_current_results() or ensure_seed_results(system)
        return enrich_payload(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {e}") from e


@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload CSV → run scoring pipeline → persist as current results."""
    if not system:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        text = contents.decode("utf-8-sig")
        df = pd.read_csv(StringIO(text))
        df = validate(df)

        _, comp_scored = _score_dataframe(df, explain=False)
        payload = build_payload_from_comp(comp_scored, df, source=f"upload:{file.filename}")
        save_current_results(payload)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {e}") from e


@app.get("/api/version")
async def get_api_version():
    return {
        "product": "BurnTestr",
        "version": "2.1",
        "model_loaded": system is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
FastAPI routes for InvestIQ.

POST /analyze         — start a new analysis pipeline run (async)
GET  /status/{run_id} — get current pipeline state / step progress
GET  /report/{run_id} — get the final report JSON
GET  /report/{run_id}/pdf — download the PDF report
GET  /jobs            — list all run_ids
DELETE /jobs/{run_id} — remove a job
GET  /health          — health check
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from services.sanitizer import sanitize_analysis_request

from api.job_store import delete_job, get_job, list_jobs, store_job
from pipeline.orchestrator import PipelineOrchestrator

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Primary ticker symbol, e.g. AAPL", max_length=15)
    competitors: list[str] = Field(
        default_factory=list,
        description="Up to 3 competitor tickers",
        max_length=3,
    )

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        from services.sanitizer import sanitize_ticker
        result = sanitize_ticker(v)
        if not result.ok:
            raise ValueError(result.error)
        return result.value

    @field_validator("competitors", mode="before")
    @classmethod
    def _validate_competitors(cls, v: list) -> list:
        from services.sanitizer import sanitize_competitors
        clean, _ = sanitize_competitors([str(t) for t in v])
        return clean


class AnalyzeResponse(BaseModel):
    run_id: str
    ticker: str
    competitors: list[str]
    status: str = "started"
    message: str = ""


class StatusResponse(BaseModel):
    run_id: str
    ticker: str
    steps: list[dict[str, Any]]
    is_complete: bool
    has_error: bool
    errors: list[dict]


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline(ticker: str, competitors: list[str], run_id: str) -> None:
    """Executed in a background thread; updates job store after each step."""
    orchestrator = PipelineOrchestrator(
        step_callback=lambda state: store_job(run_id, state)
    )
    state = orchestrator.run(ticker, competitors)
    store_job(run_id, state)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start a new pipeline analysis run asynchronously."""
    from models.pipeline_state import PipelineState

    # Double-check sanitization (validators already ran, this is defensive)
    sanitized = sanitize_analysis_request(req.ticker, req.competitors)
    if not sanitized.ok:
        raise HTTPException(status_code=422, detail="; ".join(sanitized.errors))

    # Pre-create state so /status is immediately available
    state = PipelineState(
        ticker=sanitized.ticker,
        competitors=sanitized.competitors,
    )
    store_job(state.run_id, state)

    background_tasks.add_task(
        _run_pipeline, sanitized.ticker, sanitized.competitors, state.run_id
    )

    logger.info("Analysis started: run_id=%s ticker=%s", state.run_id, sanitized.ticker)
    return AnalyzeResponse(
        run_id=state.run_id,
        ticker=sanitized.ticker,
        competitors=sanitized.competitors,
        status="started",
        message=f"Pipeline running in background. Poll /status/{state.run_id}",
    )


@router.get("/status/{run_id}", response_model=StatusResponse)
async def get_status(run_id: str):
    """Get real-time pipeline step status."""
    state = get_job(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    return StatusResponse(
        run_id=state.run_id,
        ticker=state.ticker,
        steps=[s.model_dump() for s in state.steps],
        is_complete=state.is_complete,
        has_error=state.has_critical_error,
        errors=[e.model_dump() for e in state.errors],
    )


@router.get("/report/{run_id}")
async def get_report(run_id: str):
    """Return the final report as JSON."""
    state = get_job(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    if state.report is None:
        raise HTTPException(status_code=202, detail="Report not yet ready")
    return state.report.model_dump()


@router.get("/report/{run_id}/pdf")
async def download_pdf(run_id: str):
    """Stream the PDF report file."""
    state = get_job(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    if state.report is None or not state.report.pdf_path:
        raise HTTPException(status_code=404, detail="PDF not available")
    pdf_path = Path(state.report.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on server")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"InvestIQ_{state.ticker}_{run_id[:8]}.pdf",
    )


@router.get("/jobs")
async def list_all_jobs():
    """List all job run IDs."""
    return {"run_ids": list_jobs()}


@router.delete("/jobs/{run_id}")
async def remove_job(run_id: str):
    """Remove a job from the store."""
    if not delete_job(run_id):
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    return {"deleted": run_id}


@router.get("/health")
async def health_check():
    from config.settings import settings
    return {
        "status": "ok",
        "demo_mode": settings.demo_mode,
        "granite_model": settings.granite_model_id,
        "watsonx_configured": bool(settings.watsonx_api_key),
        "cos_configured": bool(settings.cos_api_key),
    }

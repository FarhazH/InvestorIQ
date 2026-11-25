"""
Master PipelineState — the single typed object passed through all agents.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from models.market_data import RawMarketData
from models.validation import ValidationResult
from models.kpi import KPIResult
from models.ratios import RatioResult
from models.benchmarking import BenchmarkResult
from models.sentiment import SentimentResult
from models.risk import RiskResult
from models.recommendation import RecommendationResult
from models.report import FinalReport
from models.advanced_analysis import AdvancedAnalysisResult
from models.rag import RAGResult
from models.analyst_review import AnalystReview
from models.investor_profile import InvestorProfile


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class AgentError(BaseModel):
    agent: str
    error: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StepInfo(BaseModel):
    name: str
    status: StepStatus = StepStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    message: str = ""


AGENT_STEPS = [
    "data_gathering",
    "validation",
    "kpi_extraction",
    "ratio_analysis",
    "advanced_analysis",
    "benchmarking",
    "sentiment",
    "risk_trend",
    "recommendation",
    "report",
    "rag_ingestion",
]


class PipelineState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    competitors: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    granite_mode: str = "demo"     # "live" | "demo" | "fallback" — set when first Granite call succeeds

    # Step tracking
    steps: list[StepInfo] = Field(
        default_factory=lambda: [StepInfo(name=s) for s in AGENT_STEPS]
    )
    current_step: int = 0  # index into steps

    # Agent outputs (populated sequentially)
    raw_data: RawMarketData | None = None
    validation: ValidationResult | None = None
    kpis: KPIResult | None = None
    ratios: RatioResult | None = None
    advanced: AdvancedAnalysisResult | None = None
    benchmark: BenchmarkResult | None = None
    sentiment: SentimentResult | None = None
    risk: RiskResult | None = None
    recommendation: RecommendationResult | None = None
    report: FinalReport | None = None
    rag: RAGResult | None = None

    errors: list[AgentError] = Field(default_factory=list)
    analyst_reviews: list[AnalystReview] = Field(default_factory=list)

    # Client-facing / Phase 5
    investor_profile: InvestorProfile | None = None

    @property
    def is_complete(self) -> bool:
        return self.report is not None

    @property
    def has_critical_error(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_step(self, name: str) -> StepInfo | None:
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def step_summary(self) -> dict[str, str]:
        return {s.name: s.status.value for s in self.steps}

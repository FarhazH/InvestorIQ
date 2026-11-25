"""
Sequential pipeline orchestrator — executes all 9 agents in strict order.
Each agent receives the full PipelineState, mutates it, and returns it.
If a critical failure occurs, the pipeline stops early.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

from models.pipeline_state import PipelineState, StepStatus
from agents import (
    DataGatheringAgent,
    ValidationAgent,
    KPIAgent,
    RatioAgent,
    BenchmarkingAgent,
    SentimentAgent,
    RiskAgent,
    RecommendationAgent,
    ReportAgent,
)
from agents.advanced_analysis_agent import AdvancedAnalysisAgent
from agents.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

# Ordered pipeline nodes
PIPELINE_NODES = [
    DataGatheringAgent,
    ValidationAgent,
    KPIAgent,
    RatioAgent,
    AdvancedAnalysisAgent,   # Phase 2: DCF, Altman Z, scoring, backtest
    BenchmarkingAgent,
    SentimentAgent,
    RiskAgent,
    RecommendationAgent,
    ReportAgent,
    RAGAgent,                # Phase 3: SEC filing RAG + risk summary
]


class PipelineOrchestrator:
    """
    Runs all agents sequentially. Provides optional step_callback for
    real-time progress updates (used by the FastAPI SSE endpoint).
    """

    def __init__(
        self,
        step_callback: Callable[[PipelineState], None] | None = None,
    ):
        self.step_callback = step_callback
        self.agents = [AgentClass() for AgentClass in PIPELINE_NODES]

    def run(self, ticker: str, competitors: list[str] | None = None) -> PipelineState:
        """
        Execute the full pipeline and return the final PipelineState.

        Args:
            ticker: Primary ticker symbol (e.g. "AAPL")
            competitors: Up to 3 competitor tickers

        Returns:
            Completed PipelineState with all agent outputs populated.
        """
        state = PipelineState(
            ticker=ticker.upper(),
            competitors=[c.upper() for c in (competitors or [])[:3]],
        )
        logger.info(
            "Pipeline started: run_id=%s ticker=%s competitors=%s",
            state.run_id, state.ticker, state.competitors
        )

        for agent in self.agents:
            # Skip if a previous step critically failed (not just warned)
            if state.has_critical_error:
                step = state.get_step(agent.name)
                if step:
                    step.status = StepStatus.SKIPPED
                    step.message = "Skipped due to upstream critical failure"
                logger.warning("Skipping agent '%s' due to upstream failure", agent.name)
                continue

            t_start = time.monotonic()
            try:
                state = agent.run(state)
            except Exception as exc:
                # Uncaught exception — mark step failed and continue
                step = state.get_step(agent.name)
                if step:
                    step.status = StepStatus.FAILED
                    step.message = str(exc)
                from models.pipeline_state import AgentError
                state.errors.append(AgentError(agent=agent.name, error=str(exc)))
                logger.exception("Unhandled error in agent '%s'", agent.name)

            elapsed = time.monotonic() - t_start
            step = state.get_step(agent.name)
            if step and step.duration_s is None:
                step.duration_s = round(elapsed, 2)

            # Notify subscriber (e.g. SSE stream)
            if self.step_callback:
                try:
                    self.step_callback(state)
                except Exception:
                    pass

        logger.info(
            "Pipeline complete: run_id=%s steps=%s",
            state.run_id,
            state.step_summary()
        )
        return state

    def run_until(self, ticker: str, competitors: list[str] | None, stop_after: str) -> PipelineState:
        """Run pipeline but stop after a named step (useful for partial runs / testing)."""
        state = PipelineState(
            ticker=ticker.upper(),
            competitors=[c.upper() for c in (competitors or [])[:3]],
        )
        for agent in self.agents:
            if state.has_critical_error:
                break
            state = agent.run(state)
            if agent.name == stop_after:
                break
        return state

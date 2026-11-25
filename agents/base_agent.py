"""
Abstract BaseAgent — shared retry logic, structured logging, and step-tracking.

Changes in v2:
- Exponential backoff: delay * 2^(attempt-1)
- _log_input / _log_output helpers for structured per-step telemetry
- _warn_step sets duration like _finish_step
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from models.pipeline_state import PipelineState, StepStatus, AgentError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self):
        self.logger = logging.getLogger(f"investiq.agents.{self.name}")
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay

    # ── Step lifecycle helpers ────────────────────────────────────────────────

    def _start_step(self, state: PipelineState) -> None:
        step = state.get_step(self.name)
        if step:
            step.status = StepStatus.RUNNING
            step.started_at = _now_iso()
        self.logger.info(
            "START | agent=%s run_id=%s ticker=%s",
            self.name, state.run_id, state.ticker,
        )

    def _finish_step(self, state: PipelineState, message: str = "") -> None:
        step = state.get_step(self.name)
        duration = 0.0
        if step:
            step.status = StepStatus.DONE
            step.finished_at = _now_iso()
            if step.started_at:
                started = datetime.fromisoformat(step.started_at)
                now = datetime.now(timezone.utc)
                # Handle both tz-aware and tz-naive started_at strings
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                duration = (now - started).total_seconds()
                step.duration_s = round(duration, 3)
            if message:
                step.message = message
        self.logger.info(
            "DONE  | agent=%s duration=%.2fs output=%s",
            self.name, duration, (message or "ok")[:120],
        )

    def _warn_step(self, state: PipelineState, message: str) -> None:
        step = state.get_step(self.name)
        duration = 0.0
        if step:
            step.status = StepStatus.WARNING
            step.finished_at = _now_iso()
            if step.started_at:
                started = datetime.fromisoformat(step.started_at)
                now = datetime.now(timezone.utc)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                duration = (now - started).total_seconds()
                step.duration_s = round(duration, 3)
            step.message = message
        self.logger.warning(
            "WARN  | agent=%s duration=%.2fs msg=%s",
            self.name, duration, message[:120],
        )

    def _fail_step(self, state: PipelineState, error: str) -> None:
        step = state.get_step(self.name)
        duration = 0.0
        if step:
            step.status = StepStatus.FAILED
            step.finished_at = _now_iso()
            if step.started_at:
                started = datetime.fromisoformat(step.started_at)
                now = datetime.now(timezone.utc)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                duration = (now - started).total_seconds()
                step.duration_s = round(duration, 3)
            step.message = error[:300]
        state.errors.append(AgentError(agent=self.name, error=error[:500]))
        self.logger.error(
            "FAIL  | agent=%s duration=%.2fs error=%s",
            self.name, duration, error[:200],
        )

    # ── Structured telemetry helpers ──────────────────────────────────────────

    def _log_input(self, **kwargs: Any) -> None:
        """Log a summary of the agent's input at DEBUG level."""
        kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug("INPUT | agent=%s %s", self.name, kv)

    def _log_output(self, **kwargs: Any) -> None:
        """Log a summary of the agent's output at DEBUG level."""
        kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug("OUTPUT| agent=%s %s", self.name, kv)

    # ── Retry with exponential backoff ────────────────────────────────────────

    def _with_retry(self, fn, *args, **kwargs):
        """
        Call fn(*args, **kwargs) with exponential backoff.
        Backoff: retry_delay * 2^(attempt-1)  →  1s, 2s, 4s, ...
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                self.logger.warning(
                    "RETRY | agent=%s attempt=%d/%d error=%s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(backoff)
        raise RuntimeError(f"All {self.max_retries} retries failed for {self.name}") from last_exc

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute this agent. Mutates state in-place and returns it."""
        ...

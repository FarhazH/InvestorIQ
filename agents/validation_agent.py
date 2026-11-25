"""
Agent 2 — Validation Agent.
Checks for missing values, stale data, outliers, and currency consistency.
Outputs a DataQualityReport per ticker and an overall pass/fail decision.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.market_data import TickerData
from models.validation import ValidationResult, DataQualityReport, QualityFlag

# Minimum acceptable quality score to continue the pipeline
QUALITY_THRESHOLD = 0.5

# Fields we consider essential
REQUIRED_INFO_FIELDS = [
    "totalRevenue", "netIncomeToCommon", "grossMargins",
    "trailingPE", "debtToEquity", "currentRatio",
]
REQUIRED_STMT_KEYS = ["Total Revenue", "Net Income"]


def _check_ticker(td: TickerData) -> DataQualityReport:
    flags: list[QualityFlag] = []
    missing: list[str] = []
    stale: list[str] = []
    outliers: list[str] = []
    score = 1.0

    info = td.info or {}

    # --- Missing fields -------------------------------------------------
    for field in REQUIRED_INFO_FIELDS:
        if info.get(field) is None:
            missing.append(field)
            flags.append(QualityFlag(field=field, issue="missing", severity="warning"))
            score -= 0.08

    # --- Price history --------------------------------------------------
    ph = td.price_history or {}
    close_data = ph.get("Close", {})
    if not close_data:
        flags.append(QualityFlag(field="price_history", issue="empty", severity="error"))
        missing.append("price_history")
        score -= 0.20
    else:
        # Stale check: last date should be within 5 business days
        try:
            last_date_str = max(str(k) for k in close_data.keys())
            last_date = datetime.fromisoformat(last_date_str[:10])
            if (datetime.now(timezone.utc) - last_date) > timedelta(days=10):
                stale.append("price_history")
                flags.append(QualityFlag(
                    field="price_history",
                    issue=f"stale (last date {last_date_str[:10]})",
                    severity="warning",
                ))
                score -= 0.10
        except Exception:
            pass

        # Outlier check: any price < 0 is a data error
        prices = list(close_data.values())
        if prices:
            try:
                if min(float(v) for v in prices if v is not None) <= 0:
                    outliers.append("price_history.Close")
                    flags.append(QualityFlag(
                        field="price_history.Close",
                        issue="negative or zero price",
                        severity="error",
                    ))
                    score -= 0.15
            except Exception:
                pass

    # --- Income statement -----------------------------------------------
    income = td.income_stmt or {}
    for key in REQUIRED_STMT_KEYS:
        if key not in income:
            missing.append(f"income_stmt.{key}")
            flags.append(QualityFlag(field=f"income_stmt.{key}", issue="missing", severity="warning"))
            score -= 0.05

    # --- Currency consistency -------------------------------------------
    currency = info.get("currency", "USD")
    if currency not in ("USD", "GBp", "EUR", "GBP", "JPY", "CAD", "AUD"):
        flags.append(QualityFlag(
            field="currency",
            issue=f"unexpected currency {currency}",
            severity="warning",
        ))
        score -= 0.03

    score = max(0.0, min(1.0, score))
    passed = score >= QUALITY_THRESHOLD

    return DataQualityReport(
        ticker=td.ticker,
        quality_score=round(score, 3),
        flags=flags,
        missing_fields=missing,
        stale_fields=stale,
        outlier_fields=outliers,
        currency=currency,
        passed=passed,
        message="" if passed else f"Quality too low ({score:.2f} < {QUALITY_THRESHOLD})",
    )


class ValidationAgent(BaseAgent):
    name = "validation"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        if state.raw_data is None:
            self._fail_step(state, "No raw data available for validation")
            return state

        try:
            primary_qr = _check_ticker(state.raw_data.primary)
            competitor_qrs = [_check_ticker(c) for c in state.raw_data.competitors]

            all_scores = [primary_qr.quality_score] + [c.quality_score for c in competitor_qrs]
            overall_score = round(sum(all_scores) / len(all_scores), 3)
            overall_passed = primary_qr.passed  # primary must pass

            state.validation = ValidationResult(
                primary=primary_qr,
                competitors=competitor_qrs,
                overall_passed=overall_passed,
                overall_quality_score=overall_score,
            )

            if not overall_passed:
                self._warn_step(
                    state,
                    f"Primary data quality score {primary_qr.quality_score:.2f} below threshold {QUALITY_THRESHOLD}. "
                    "Proceeding with warnings — results may be unreliable.",
                )
            else:
                self._finish_step(state, f"Quality score: {overall_score:.2f}")
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

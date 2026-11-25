"""
Pydantic models for Phase 4 — Analyst Review and Human-in-the-Loop audit trail.

An AnalystReview records every human override or annotation applied to a
pipeline run so decisions are fully traceable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ── Individual review entry ───────────────────────────────────────────────────

class AnalystReview(BaseModel):
    """A single human-analyst annotation attached to a pipeline run."""

    run_id: str
    ticker: str

    # Override — if the analyst disagrees with the AI recommendation
    override_recommendation: Literal["Overweight", "Hold", "Underweight", ""] = ""
    override_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Analyst notes (free-text, max 2000 chars)
    notes: str = Field(default="", max_length=2000)

    # Analyst identity (optional — for multi-user scenarios)
    analyst_name: str = "Anonymous"

    # Flags raised by the analyst
    flags: list[str] = Field(default_factory=list)

    # Timestamp (always UTC, timezone-aware)
    reviewed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_override(self) -> bool:
        return bool(self.override_recommendation)

    @property
    def effective_recommendation(self) -> str:
        """Return the override if set, otherwise empty string (use AI)."""
        return self.override_recommendation or ""


# ── Explanation entry — per-reason grounding trace ───────────────────────────

class ExplanationEntry(BaseModel):
    """
    One row in the explanation trace table.
    Links an AI-cited reason to the actual computed metric value so an analyst
    can verify the AI's claim at a glance.
    """
    reason: str                             # the AI's reasoning sentence
    metric: str = ""                        # metric name (may be empty)
    cited_value: str = ""                   # value as cited by AI
    computed_value: str | None = None       # value as computed by pipeline
    grounded: bool = False                  # True if cited ≈ computed (±10%)
    source: str = "Granite"                 # "Granite" | "Rule-based"


# ── Factor contribution entry — waterfall chart data ─────────────────────────

class FactorContribution(BaseModel):
    """
    One bar in the factor contribution waterfall chart.
    Shows how much each scoring dimension contributed (positively or negatively)
    to the final composite score relative to a neutral baseline of 50.
    """
    factor: str                             # e.g. "Fundamentals"
    raw_score: float                        # 0–100 component score
    weight: float                           # fractional weight (0–1)
    contribution: float                     # (raw_score - 50) * weight
    direction: str                          # "positive" | "negative" | "neutral"


# ── Confidence breakdown ──────────────────────────────────────────────────────

class ConfidenceBreakdown(BaseModel):
    """
    Per-dimension confidence contribution for display as a horizontal bar chart.
    """
    ticker: str
    composite_score: float                  # 0–100
    signal: str                             # Overweight | Hold | Underweight
    signal_confidence: float                # 0–1
    factors: list[FactorContribution] = Field(default_factory=list)
    baseline: float = 50.0                  # neutral score
    explanation: str = ""                   # one-line natural-language summary


def build_factor_contributions(ws: "WeightedScore") -> list[FactorContribution]:  # noqa: F821
    """
    Build a FactorContribution list from a WeightedScore object.
    contribution_i = (raw_score_i − 50) × weight_i
    Positive contribution means this dimension is pulling the signal UP.
    """
    mapping = [
        ("Fundamentals", ws.fundamentals_score, ws.weights.get("fundamentals", 0.30)),
        ("Valuation",    ws.valuation_score,    ws.weights.get("valuation",    0.20)),
        ("Momentum",     ws.momentum_score,     ws.weights.get("momentum",     0.20)),
        ("Sentiment",    ws.sentiment_score,     ws.weights.get("sentiment",    0.15)),
        ("Risk (inv.)",  ws.risk_score,          ws.weights.get("risk",         0.15)),
    ]
    result = []
    for factor, raw, w in mapping:
        contrib = (raw - 50.0) * w
        direction = "positive" if contrib > 0.5 else ("negative" if contrib < -0.5 else "neutral")
        result.append(FactorContribution(
            factor=factor,
            raw_score=raw,
            weight=w,
            contribution=round(contrib, 2),
            direction=direction,
        ))
    return result


def build_confidence_breakdown(ws: "WeightedScore") -> ConfidenceBreakdown:  # noqa: F821
    """Build a ConfidenceBreakdown from a WeightedScore."""
    factors = build_factor_contributions(ws)
    top_pos = max(factors, key=lambda f: f.contribution, default=None)
    top_neg = min(factors, key=lambda f: f.contribution, default=None)

    parts = []
    if top_pos and top_pos.contribution > 0:
        parts.append(f"{top_pos.factor} is the strongest positive driver")
    if top_neg and top_neg.contribution < 0:
        parts.append(f"{top_neg.factor} is the main drag")

    explanation = (
        f"{ws.signal} signal (score {ws.composite_score:.1f}/100). "
        + ("; ".join(parts) + "." if parts else "No dominant driver.")
    )
    return ConfidenceBreakdown(
        ticker=ws.ticker,
        composite_score=ws.composite_score,
        signal=ws.signal,
        signal_confidence=ws.signal_confidence,
        factors=factors,
        explanation=explanation,
    )


def build_explanation_trace(
    rec: "RecommendationResult",  # noqa: F821
    computed_lookup: dict[str, float | None],
) -> list[ExplanationEntry]:
    """
    Build an explanation trace from a RecommendationResult's reasons + grounding flags.
    Each row links the AI's textual reason to the pipeline-computed value.
    """
    # Build a quick metric→grounding_flag index
    flag_index: dict[str, "GroundingFlag"] = {}  # noqa: F821
    for f in rec.grounding_flags:
        flag_index[f.metric] = f

    entries: list[ExplanationEntry] = []
    for reason in rec.reasons:
        metric = reason.metric or ""
        flag = flag_index.get(metric)
        computed_val = None
        grounded = False
        if flag:
            computed_val = flag.computed_value
            grounded = flag.match
        elif metric and metric in computed_lookup:
            cv = computed_lookup.get(metric)
            computed_val = str(round(cv, 4)) if cv is not None else None

        entries.append(ExplanationEntry(
            reason=reason.point,
            metric=metric,
            cited_value=reason.value or "",
            computed_value=computed_val,
            grounded=grounded,
            source="Granite" if rec.granite_mode in ("live", "demo") else "Rule-based",
        ))
    return entries

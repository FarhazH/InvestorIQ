"""
Agent 8 — Recommendation Agent.
Uses IBM Granite to generate a structured investment recommendation
(Overweight/Hold/Underweight) grounded in all prior agent outputs.

v2 improvements:
- Passes validate_recommendation_json validator to call_granite_json
- Grounding check: every cited metric value is cross-checked against
  computed KPIs/ratios/sentiment/risk (±10% tolerance)
- Rule-based deterministic fallback if all Granite attempts fail
- Structured _log_input / _log_output telemetry
- Sets state.granite_mode and recommendation.granite_mode
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.recommendation import RecommendationResult, RecommendationReason, GroundingFlag
from services.watsonx_client import call_granite_json, validate_recommendation_json

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "recommendation_prompt.txt"


# ── Prompt building ───────────────────────────────────────────────────────────

def _load_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "Generate investment recommendation for {{TICKER}} based on: "
            "KPIs={{KPIS}} Ratios={{RATIOS}} Benchmark={{BENCHMARK}} "
            "Sentiment={{SENTIMENT}} Risk={{RISK}}"
        )


def _fmt(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, indent=2)
    except Exception:
        return str(obj)


def _build_prompt(state: PipelineState) -> str:
    template = _load_prompt()
    kpis_dict = state.kpis.primary.model_dump() if state.kpis else {}
    ratios_dict = state.ratios.primary.model_dump() if state.ratios else {}
    benchmark_dict = {
        "overall_rank": state.benchmark.overall_rank if state.benchmark else None,
        "over_performing": state.benchmark.over_performing_metrics if state.benchmark else [],
        "under_performing": state.benchmark.under_performing_metrics if state.benchmark else [],
        "peer_comparison": (state.benchmark.peer_comparison_table[:4] if state.benchmark else []),
    }
    sentiment_dict = {
        "label": state.sentiment.sentiment_label if state.sentiment else "N/A",
        "score": state.sentiment.aggregate_score if state.sentiment else 0.0,
        "top_themes": state.sentiment.top_themes if state.sentiment else [],
    }
    risk_dict = {
        "overall_risk_score": state.risk.overall_risk_score if state.risk else 50,
        "overall_risk_level": state.risk.overall_risk_level if state.risk else "Medium",
        "indicators": [
            {"category": r.category, "level": r.level, "score": r.score}
            for r in (state.risk.indicators if state.risk else [])
        ],
    }
    # Investor profile context (Phase 5)
    profile_ctx = ""
    if state.investor_profile is not None:
        profile_ctx = "\n\n" + state.investor_profile.to_prompt_context()

    return (
        template
        .replace("{{TICKER}}", state.ticker)
        .replace("{{DATE}}", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        .replace("{{KPIS}}", _fmt(kpis_dict))
        .replace("{{RATIOS}}", _fmt(ratios_dict))
        .replace("{{BENCHMARK}}", _fmt(benchmark_dict))
        .replace("{{SENTIMENT}}", _fmt(sentiment_dict))
        .replace("{{RISK}}", _fmt(risk_dict))
        + profile_ctx
    )


# ── Grounding check ───────────────────────────────────────────────────────────

def _extract_numeric(value_str: Any) -> float | None:
    """Extract the first number from a string like '8.1%', '28.5', '$2.3B'."""
    if value_str is None:
        return None
    s = str(value_str).replace("%", "").replace("$", "").replace("B", "e9").replace("M", "e6")
    match = re.search(r"-?\d+\.?\d*(?:e[+-]?\d+)?", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _build_computed_lookup(state: PipelineState) -> dict[str, float | None]:
    """Build a flat {metric_name: computed_value} dict from pipeline state."""
    lookup: dict[str, float | None] = {}
    if state.ratios:
        r = state.ratios.primary
        lookup.update({
            "pe_ratio": r.pe_ratio,
            "forward_pe": r.forward_pe,
            "pb_ratio": r.pb_ratio,
            "roe": (r.roe * 100) if r.roe is not None else None,
            "roa": (r.roa * 100) if r.roa is not None else None,
            "net_margin": (r.net_margin * 100) if r.net_margin is not None else None,
            "operating_margin": (r.operating_margin * 100) if r.operating_margin is not None else None,
            "gross_margin": (r.gross_margin * 100) if r.gross_margin is not None else None,
            "debt_to_equity": r.debt_to_equity,
            "beta": r.beta,
            "volatility_annualised": r.volatility_annualised,
            "rsi_14": r.rsi_14,
        })
    if state.kpis:
        k = state.kpis.primary
        lookup["revenue_growth_yoy"] = k.revenue_growth_yoy
        lookup["earnings_growth_yoy"] = k.earnings_growth_yoy
        lookup["eps_ttm"] = k.eps_ttm
    if state.sentiment:
        lookup["sentiment_score"] = state.sentiment.aggregate_score
    if state.risk:
        lookup["overall_risk_score"] = state.risk.overall_risk_score
    return lookup


def _grounding_check(
    metrics_cited: dict,
    reasons: list[RecommendationReason],
    computed: dict[str, float | None],
    tolerance_pct: float = 10.0,
) -> list[GroundingFlag]:
    """
    Cross-check every cited metric value against computed pipeline values.
    Returns a list of GroundingFlag records.
    """
    flags: list[GroundingFlag] = []

    # Check metrics_cited dict
    for metric, cited_val in metrics_cited.items():
        computed_val = computed.get(metric)
        cited_num = _extract_numeric(cited_val)
        flags.append(_make_flag(metric, cited_val, computed_val, cited_num, tolerance_pct))

    # Also check reason values
    for reason in reasons:
        if reason.metric and reason.value:
            computed_val = computed.get(reason.metric)
            cited_num = _extract_numeric(reason.value)
            flags.append(_make_flag(reason.metric, reason.value, computed_val, cited_num, tolerance_pct))

    return flags


def _make_flag(
    metric: str,
    cited_raw: Any,
    computed_val: float | None,
    cited_num: float | None,
    tolerance_pct: float,
) -> GroundingFlag:
    if computed_val is None or cited_num is None:
        # Can't verify — mark as unchecked (match=False, no computed_value)
        return GroundingFlag(
            metric=metric,
            cited_value=str(cited_raw),
            computed_value=None,
            match=False,
            tolerance_pct=tolerance_pct,
        )
    # Allow ±tolerance_pct relative difference
    if abs(computed_val) > 1e-9:
        rel_diff = abs(cited_num - computed_val) / abs(computed_val) * 100
        match = rel_diff <= tolerance_pct
    else:
        match = abs(cited_num) <= 0.01  # both near zero → match
    return GroundingFlag(
        metric=metric,
        cited_value=str(cited_raw),
        computed_value=str(round(computed_val, 4)),
        match=match,
        tolerance_pct=tolerance_pct,
    )


# ── Rule-based fallback recommendation ───────────────────────────────────────

def _rule_based_recommendation(state: PipelineState) -> RecommendationResult:
    """
    Deterministic scoring when Granite is unavailable.
    Scoring rubric (each criterion is 0 or 1 point):
      Fundamental: net_margin>15%, rev_growth>5%, positive FCF        → 0-3
      Valuation:   PE < 30, pb < 5                                     → 0-2
      Risk:        risk_score < 50                                     → 0-1
      Sentiment:   sentiment > 0.1                                     → 0-1
    Score ≥5 → Overweight, 3-4 → Hold, <3 → Underweight
    """
    score = 0
    reasons: list[RecommendationReason] = []

    kpis = state.kpis.primary if state.kpis else None
    ratios = state.ratios.primary if state.ratios else None
    risk = state.risk
    sent = state.sentiment

    # Fundamentals
    if kpis:
        if kpis.net_margin and kpis.net_margin > 0.15:
            score += 1
            reasons.append(RecommendationReason(
                point="Net margin above 15%", metric="net_margin",
                value=f"{kpis.net_margin*100:.1f}%",
            ))
        if kpis.revenue_growth_yoy and kpis.revenue_growth_yoy > 5:
            score += 1
            reasons.append(RecommendationReason(
                point="Revenue growth above 5% YoY", metric="revenue_growth_yoy",
                value=f"{kpis.revenue_growth_yoy:.1f}%",
            ))
        if kpis.free_cash_flow and kpis.free_cash_flow > 0:
            score += 1
            reasons.append(RecommendationReason(
                point="Positive free cash flow", metric="free_cash_flow",
                value=f"{kpis.free_cash_flow/1e9:.2f}B",
            ))

    # Valuation
    if ratios:
        if ratios.pe_ratio and ratios.pe_ratio < 30:
            score += 1
            reasons.append(RecommendationReason(
                point="P/E ratio below 30x", metric="pe_ratio",
                value=f"{ratios.pe_ratio:.1f}",
            ))
        if ratios.pb_ratio and ratios.pb_ratio < 5:
            score += 1
            reasons.append(RecommendationReason(
                point="P/B ratio below 5x", metric="pb_ratio",
                value=f"{ratios.pb_ratio:.1f}",
            ))

    # Risk
    if risk and risk.overall_risk_score < 50:
        score += 1
        reasons.append(RecommendationReason(
            point="Overall risk score below 50", metric="overall_risk_score",
            value=f"{risk.overall_risk_score:.0f}",
        ))

    # Sentiment
    if sent and sent.aggregate_score > 0.1:
        score += 1
        reasons.append(RecommendationReason(
            point="Positive news sentiment", metric="sentiment_score",
            value=f"{sent.aggregate_score:.2f}",
        ))

    # Decision
    if score >= 5:
        rec = "Overweight"
        confidence = 0.60 + (score - 5) * 0.05
    elif score >= 3:
        rec = "Hold"
        confidence = 0.50 + (score - 3) * 0.05
    else:
        rec = "Underweight"
        confidence = 0.40 + (3 - score - 1) * 0.05

    confidence = round(min(max(confidence, 0.0), 1.0), 2)
    key_risks = []
    if risk:
        key_risks = [i.category for i in risk.indicators if i.level == "High"][:3]

    return RecommendationResult(
        ticker=state.ticker,
        recommendation=rec,
        confidence=confidence,
        target_horizon="12 months",
        reasons=reasons[:5],
        key_risks=key_risks,
        forward_outlook=(
            f"Rule-based analysis: {state.ticker} scored {score}/7 on fundamentals, "
            "valuation, risk, and sentiment criteria."
        ),
        metrics_cited={},
        grounding_flags=[],
        granite_mode="fallback",
        disclaimer=(
            "This recommendation was generated by a deterministic rule-based fallback "
            "(IBM Granite unavailable). It is NOT financial advice."
        ),
    )


# ── Recommendation parsing ────────────────────────────────────────────────────

def _parse_recommendation(
    result: dict,
    ticker: str,
    granite_mode: str,
    state: PipelineState,
) -> RecommendationResult:
    reasons = [
        RecommendationReason(
            point=r.get("point", ""),
            metric=r.get("metric", ""),
            value=str(r.get("value", "")),
        )
        for r in result.get("reasons", [])
    ]
    metrics_cited = result.get("metrics_cited", {})

    # Grounding check
    computed = _build_computed_lookup(state)
    grounding_flags = _grounding_check(metrics_cited, reasons, computed)

    return RecommendationResult(
        ticker=ticker,
        recommendation=result.get("recommendation", "Hold"),
        confidence=float(result.get("confidence", 0.5)),
        target_horizon=result.get("target_horizon", "12 months"),
        reasons=reasons,
        key_risks=result.get("key_risks", []),
        forward_outlook=result.get("forward_outlook", ""),
        metrics_cited=metrics_cited,
        grounding_flags=grounding_flags,
        granite_mode=granite_mode,
        disclaimer=result.get("disclaimer", (
            "This output is generated by an AI model and is NOT financial advice. "
            "Always consult a licensed financial advisor before making investment decisions."
        )),
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class RecommendationAgent(BaseAgent):
    name = "recommendation"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        self._log_input(
            ticker=state.ticker,
            has_kpis=state.kpis is not None,
            has_ratios=state.ratios is not None,
            has_risk=state.risk is not None,
            has_sentiment=state.sentiment is not None,
        )

        granite_mode = "demo"
        try:
            prompt = _build_prompt(state)
            result, granite_mode = call_granite_json(
                prompt,
                max_tokens=2000,
                validator=validate_recommendation_json,
            )
            state.granite_mode = granite_mode
            state.recommendation = _parse_recommendation(result, state.ticker, granite_mode, state)

        except Exception as exc:
            self.logger.warning(
                "Granite recommendation failed (%s) — using rule-based fallback", exc
            )
            state.recommendation = _rule_based_recommendation(state)
            state.granite_mode = "fallback"

            self._log_output(
                ticker=state.ticker,
                rec=state.recommendation.recommendation,
                confidence=state.recommendation.confidence,
                mode="fallback",
                grounding_flags=0,
            )
            self._warn_step(
                state,
                f"Granite unavailable; rule-based fallback used. "
                f"Recommendation: {state.recommendation.recommendation} "
                f"({state.recommendation.confidence:.0%} confidence)"
            )
            return state

        # Count grounding mismatches
        unverified = sum(1 for f in state.recommendation.grounding_flags if not f.match)
        self._log_output(
            ticker=state.ticker,
            rec=state.recommendation.recommendation,
            confidence=state.recommendation.confidence,
            mode=granite_mode,
            grounding_flags=len(state.recommendation.grounding_flags),
            unverified=unverified,
        )
        self._finish_step(
            state,
            f"Recommendation: {state.recommendation.recommendation} "
            f"(confidence={state.recommendation.confidence:.0%}, "
            f"mode={granite_mode}, grounding={len(state.recommendation.grounding_flags)-unverified}"
            f"/{len(state.recommendation.grounding_flags)} verified)"
        )
        return state

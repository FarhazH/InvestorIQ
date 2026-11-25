"""
Phase 4 — Explainability & Human-in-the-Loop tests.

Covers:
- FactorContribution model
- ConfidenceBreakdown model
- build_factor_contributions()
- build_confidence_breakdown()
- build_explanation_trace()
- AnalystReview model
- ExplanationEntry model
- PipelineState.analyst_reviews field
- edge cases: empty reasons, all positive factors, all negative, neutral
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from models.analyst_review import (
    AnalystReview,
    ExplanationEntry,
    FactorContribution,
    ConfidenceBreakdown,
    build_factor_contributions,
    build_confidence_breakdown,
    build_explanation_trace,
)
from models.advanced_analysis import WeightedScore
from models.recommendation import RecommendationResult, RecommendationReason, GroundingFlag
from models.pipeline_state import PipelineState


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_weighted_score(
    fundamentals=70.0,
    valuation=55.0,
    momentum=65.0,
    sentiment=60.0,
    risk_inv=75.0,
    composite=66.0,
    signal="Overweight",
    confidence=0.72,
) -> WeightedScore:
    return WeightedScore(
        ticker="AAPL",
        fundamentals_score=fundamentals,
        valuation_score=valuation,
        momentum_score=momentum,
        sentiment_score=sentiment,
        risk_score=risk_inv,
        weights={
            "fundamentals": 0.30,
            "valuation": 0.20,
            "momentum": 0.20,
            "sentiment": 0.15,
            "risk": 0.15,
        },
        composite_score=composite,
        signal=signal,
        signal_confidence=confidence,
    )


def _make_recommendation(
    ticker="AAPL",
    rec="Overweight",
    confidence=0.70,
    granite_mode="demo",
) -> RecommendationResult:
    return RecommendationResult(
        ticker=ticker,
        recommendation=rec,
        confidence=confidence,
        reasons=[
            RecommendationReason(
                point="Net margin above 15%",
                metric="net_margin",
                value="28.3%",
            ),
            RecommendationReason(
                point="Revenue growth above 5%",
                metric="revenue_growth_yoy",
                value="8.1%",
            ),
            RecommendationReason(
                point="Positive free cash flow",
                metric="free_cash_flow",
                value="90.5B",
            ),
        ],
        grounding_flags=[
            GroundingFlag(
                metric="net_margin",
                cited_value="28.3%",
                computed_value="27.5",
                match=True,
                tolerance_pct=10.0,
            ),
            GroundingFlag(
                metric="revenue_growth_yoy",
                cited_value="8.1%",
                computed_value="7.9",
                match=True,
                tolerance_pct=10.0,
            ),
        ],
        granite_mode=granite_mode,
    )


# ── FactorContribution ────────────────────────────────────────────────────────

class TestFactorContribution:
    def test_model_fields_exist(self):
        fc = FactorContribution(
            factor="Fundamentals",
            raw_score=70.0,
            weight=0.30,
            contribution=6.0,
            direction="positive",
        )
        assert fc.factor == "Fundamentals"
        assert fc.raw_score == 70.0
        assert fc.weight == 0.30
        assert fc.contribution == 6.0
        assert fc.direction == "positive"

    def test_negative_direction(self):
        fc = FactorContribution(
            factor="Valuation",
            raw_score=30.0,
            weight=0.20,
            contribution=-4.0,
            direction="negative",
        )
        assert fc.direction == "negative"

    def test_neutral_direction(self):
        fc = FactorContribution(
            factor="Momentum",
            raw_score=50.0,
            weight=0.20,
            contribution=0.0,
            direction="neutral",
        )
        assert fc.direction == "neutral"


# ── build_factor_contributions ────────────────────────────────────────────────

class TestBuildFactorContributions:
    def test_returns_five_factors(self):
        ws = _make_weighted_score()
        factors = build_factor_contributions(ws)
        assert len(factors) == 5

    def test_factor_names(self):
        ws = _make_weighted_score()
        factors = build_factor_contributions(ws)
        names = [f.factor for f in factors]
        assert "Fundamentals" in names
        assert "Valuation" in names
        assert "Momentum" in names
        assert "Sentiment" in names
        assert "Risk (inv.)" in names

    def test_contribution_math(self):
        ws = _make_weighted_score(fundamentals=70.0)
        factors = build_factor_contributions(ws)
        fund = next(f for f in factors if f.factor == "Fundamentals")
        # (70 - 50) * 0.30 = 6.0
        assert abs(fund.contribution - 6.0) < 0.01

    def test_zero_contribution_at_fifty(self):
        ws = _make_weighted_score(fundamentals=50.0)
        factors = build_factor_contributions(ws)
        fund = next(f for f in factors if f.factor == "Fundamentals")
        assert abs(fund.contribution) < 0.01
        assert fund.direction == "neutral"

    def test_negative_contribution(self):
        ws = _make_weighted_score(valuation=30.0)
        factors = build_factor_contributions(ws)
        val = next(f for f in factors if f.factor == "Valuation")
        assert val.contribution < 0
        assert val.direction == "negative"

    def test_direction_positive_above_threshold(self):
        ws = _make_weighted_score(sentiment=80.0)
        factors = build_factor_contributions(ws)
        sent = next(f for f in factors if f.factor == "Sentiment")
        assert sent.direction == "positive"

    def test_contributions_sum_approx_composite_minus_baseline(self):
        """Sum of contributions should equal (composite - 50) × 1 approximately."""
        ws = _make_weighted_score(
            fundamentals=70, valuation=55, momentum=65, sentiment=60, risk_inv=75,
            composite=66.0,
        )
        factors = build_factor_contributions(ws)
        total = sum(f.contribution for f in factors)
        # expected: (70-50)*.3 + (55-50)*.2 + (65-50)*.2 + (60-50)*.15 + (75-50)*.15
        expected = 20*.3 + 5*.2 + 15*.2 + 10*.15 + 25*.15
        assert abs(total - expected) < 0.1


# ── build_confidence_breakdown ────────────────────────────────────────────────

class TestBuildConfidenceBreakdown:
    def test_returns_confidence_breakdown(self):
        ws = _make_weighted_score()
        cbd = build_confidence_breakdown(ws)
        assert isinstance(cbd, ConfidenceBreakdown)
        assert cbd.ticker == "AAPL"
        assert cbd.signal == "Overweight"
        assert len(cbd.factors) == 5

    def test_explanation_non_empty(self):
        ws = _make_weighted_score()
        cbd = build_confidence_breakdown(ws)
        assert len(cbd.explanation) > 10

    def test_explanation_contains_signal(self):
        ws = _make_weighted_score()
        cbd = build_confidence_breakdown(ws)
        assert "Overweight" in cbd.explanation

    def test_hold_signal(self):
        ws = _make_weighted_score(composite=50.0, signal="Hold", confidence=0.50)
        cbd = build_confidence_breakdown(ws)
        assert "Hold" in cbd.explanation

    def test_underweight_signal(self):
        ws = _make_weighted_score(
            fundamentals=30, valuation=35, momentum=30, sentiment=25, risk_inv=20,
            composite=29.0, signal="Underweight", confidence=0.72,
        )
        cbd = build_confidence_breakdown(ws)
        assert "Underweight" in cbd.explanation

    def test_baseline_is_fifty(self):
        ws = _make_weighted_score()
        cbd = build_confidence_breakdown(ws)
        assert cbd.baseline == 50.0

    def test_composite_score_matches(self):
        ws = _make_weighted_score(composite=72.5)
        cbd = build_confidence_breakdown(ws)
        assert cbd.composite_score == 72.5


# ── build_explanation_trace ───────────────────────────────────────────────────

class TestBuildExplanationTrace:
    def test_returns_one_entry_per_reason(self):
        rec = _make_recommendation()
        trace = build_explanation_trace(rec, {})
        assert len(trace) == 3

    def test_verified_reason_grounded_true(self):
        rec = _make_recommendation()
        trace = build_explanation_trace(rec, {"net_margin": 27.5})
        nm_entry = next(e for e in trace if e.metric == "net_margin")
        assert nm_entry.grounded is True

    def test_unverifiable_reason_no_computed(self):
        rec = _make_recommendation()
        trace = build_explanation_trace(rec, {})
        fcf_entry = next(e for e in trace if e.metric == "free_cash_flow")
        # free_cash_flow not in grounding_flags and not in computed_lookup
        assert fcf_entry.computed_value is None

    def test_source_is_granite_for_demo_mode(self):
        rec = _make_recommendation(granite_mode="demo")
        trace = build_explanation_trace(rec, {})
        assert all(e.source == "Granite" for e in trace)

    def test_source_is_rule_based_for_fallback(self):
        rec = _make_recommendation(granite_mode="fallback")
        trace = build_explanation_trace(rec, {})
        assert all(e.source == "Rule-based" for e in trace)

    def test_empty_reasons_returns_empty_trace(self):
        rec = RecommendationResult(
            ticker="AAPL",
            recommendation="Hold",
            confidence=0.5,
            reasons=[],
        )
        trace = build_explanation_trace(rec, {})
        assert trace == []

    def test_reason_text_preserved(self):
        rec = _make_recommendation()
        trace = build_explanation_trace(rec, {})
        texts = [e.reason for e in trace]
        assert "Net margin above 15%" in texts


# ── AnalystReview ─────────────────────────────────────────────────────────────

class TestAnalystReview:
    def test_default_fields(self):
        ar = AnalystReview(run_id="abc", ticker="AAPL")
        assert ar.analyst_name == "Anonymous"
        assert ar.override_recommendation == ""
        assert ar.override_confidence == 0.0
        assert ar.flags == []
        assert ar.has_override is False

    def test_override_set(self):
        ar = AnalystReview(
            run_id="abc", ticker="AAPL",
            override_recommendation="Underweight",
            override_confidence=0.75,
        )
        assert ar.has_override is True
        assert ar.effective_recommendation == "Underweight"

    def test_no_override_effective_is_empty(self):
        ar = AnalystReview(run_id="abc", ticker="AAPL")
        assert ar.effective_recommendation == ""

    def test_reviewed_at_is_timezone_aware(self):
        ar = AnalystReview(run_id="abc", ticker="AAPL")
        # reviewed_at is an ISO string; check it contains timezone info
        assert "T" in ar.reviewed_at  # ISO 8601 format
        # Should end with +00:00 or Z
        assert ar.reviewed_at.endswith("+00:00") or ar.reviewed_at.endswith("Z") or "+" in ar.reviewed_at

    def test_flags_stored(self):
        ar = AnalystReview(
            run_id="abc", ticker="AAPL",
            flags=["Data quality concern", "Stale data"],
        )
        assert len(ar.flags) == 2
        assert "Stale data" in ar.flags

    def test_notes_max_length(self):
        long_notes = "x" * 2001
        with pytest.raises(Exception):
            AnalystReview(run_id="abc", ticker="AAPL", notes=long_notes)

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            AnalystReview(run_id="abc", ticker="AAPL", override_confidence=1.5)
        with pytest.raises(Exception):
            AnalystReview(run_id="abc", ticker="AAPL", override_confidence=-0.1)

    def test_serialization_round_trip(self):
        ar = AnalystReview(
            run_id="run-123", ticker="MSFT",
            override_recommendation="Hold",
            override_confidence=0.60,
            notes="Looks good but valuation is stretched.",
            analyst_name="Jane Doe",
            flags=["DCF assumptions too aggressive"],
        )
        d = ar.model_dump()
        ar2 = AnalystReview(**d)
        assert ar2.ticker == "MSFT"
        assert ar2.override_recommendation == "Hold"
        assert ar2.analyst_name == "Jane Doe"


# ── PipelineState.analyst_reviews ─────────────────────────────────────────────

class TestPipelineStateAnalystReviews:
    def test_analyst_reviews_field_exists(self):
        ps = PipelineState(ticker="AAPL")
        assert hasattr(ps, "analyst_reviews")
        assert ps.analyst_reviews == []

    def test_append_review_to_state(self):
        ps = PipelineState(ticker="AAPL")
        ar = AnalystReview(run_id=ps.run_id, ticker="AAPL", notes="Test note.")
        ps.analyst_reviews.append(ar)
        assert len(ps.analyst_reviews) == 1
        assert ps.analyst_reviews[0].notes == "Test note."

    def test_multiple_reviews_ordered(self):
        ps = PipelineState(ticker="AAPL")
        for i in range(3):
            ps.analyst_reviews.append(
                AnalystReview(run_id=ps.run_id, ticker="AAPL", notes=f"Note {i}")
            )
        assert len(ps.analyst_reviews) == 3
        assert ps.analyst_reviews[2].notes == "Note 2"

    def test_serialization_includes_reviews(self):
        ps = PipelineState(ticker="AAPL")
        ps.analyst_reviews.append(
            AnalystReview(run_id=ps.run_id, ticker="AAPL", override_recommendation="Hold")
        )
        d = ps.model_dump()
        assert len(d["analyst_reviews"]) == 1
        assert d["analyst_reviews"][0]["override_recommendation"] == "Hold"

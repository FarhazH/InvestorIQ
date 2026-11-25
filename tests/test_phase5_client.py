"""
Phase 5 — Client-Facing Features tests.

Covers:
- InvestorProfile model: fields, validators, to_prompt_context()
- WatchlistEntry model
- PortfolioPosition model
- PortfolioConfig model: normalised_weights(), tickers property
- PortfolioAnalysisResult model
- Investor profile injected into PipelineState
- Prompt personalisation: profile context in recommendation prompt
- Edge cases: empty notes, ESG, sector, all risk tolerances
"""
from __future__ import annotations

import pytest

from models.investor_profile import (
    InvestorProfile,
    WatchlistEntry,
    PortfolioPosition,
    PortfolioConfig,
    PortfolioAnalysisResult,
)
from models.pipeline_state import PipelineState


# ── InvestorProfile ───────────────────────────────────────────────────────────

class TestInvestorProfile:
    def test_default_fields(self):
        p = InvestorProfile()
        assert p.risk_tolerance == "Moderate"
        assert p.horizon == "Medium-term (1–3 years)"
        assert p.intent == "Research only"
        assert p.portfolio_size == "$10K – $100K"
        assert p.esg_preference == "No preference"
        assert p.sector_notes == ""

    def test_risk_level_num_conservative(self):
        p = InvestorProfile(risk_tolerance="Conservative")
        assert p.risk_level_num == 1

    def test_risk_level_num_moderate(self):
        p = InvestorProfile(risk_tolerance="Moderate")
        assert p.risk_level_num == 2

    def test_risk_level_num_aggressive(self):
        p = InvestorProfile(risk_tolerance="Aggressive")
        assert p.risk_level_num == 3

    def test_horizon_years_short(self):
        p = InvestorProfile(horizon="Short-term (< 1 year)")
        assert p.horizon_years == 0.5

    def test_horizon_years_medium(self):
        p = InvestorProfile(horizon="Medium-term (1–3 years)")
        assert p.horizon_years == 2.0

    def test_horizon_years_long(self):
        p = InvestorProfile(horizon="Long-term (3+ years)")
        assert p.horizon_years == 5.0

    def test_to_prompt_context_contains_risk(self):
        p = InvestorProfile(risk_tolerance="Aggressive")
        ctx = p.to_prompt_context()
        assert "Aggressive" in ctx

    def test_to_prompt_context_contains_horizon(self):
        p = InvestorProfile(horizon="Long-term (3+ years)")
        ctx = p.to_prompt_context()
        assert "Long-term" in ctx

    def test_to_prompt_context_contains_disclaimer(self):
        p = InvestorProfile()
        ctx = p.to_prompt_context()
        assert "tailor" in ctx.lower() or "conservative" in ctx.lower()

    def test_to_prompt_context_includes_sector_notes(self):
        p = InvestorProfile(sector_notes="Avoid tobacco stocks")
        ctx = p.to_prompt_context()
        assert "tobacco" in ctx

    def test_to_prompt_context_no_sector_notes_when_empty(self):
        p = InvestorProfile(sector_notes="")
        ctx = p.to_prompt_context()
        assert "Sector notes" not in ctx

    def test_esg_preference_stored(self):
        p = InvestorProfile(esg_preference="ESG-only")
        assert p.esg_preference == "ESG-only"

    def test_sector_notes_max_length(self):
        with pytest.raises(Exception):
            InvestorProfile(sector_notes="x" * 201)

    def test_invalid_risk_tolerance(self):
        with pytest.raises(Exception):
            InvestorProfile(risk_tolerance="Ultra-Aggressive")

    def test_invalid_horizon(self):
        with pytest.raises(Exception):
            InvestorProfile(horizon="10 years")

    def test_serialization_round_trip(self):
        p = InvestorProfile(
            risk_tolerance="Conservative",
            horizon="Long-term (3+ years)",
            intent="New position",
            esg_preference="ESG-aware",
            sector_notes="Focus on healthcare",
        )
        d = p.model_dump()
        p2 = InvestorProfile(**d)
        assert p2.risk_tolerance == "Conservative"
        assert p2.sector_notes == "Focus on healthcare"


# ── WatchlistEntry ────────────────────────────────────────────────────────────

class TestWatchlistEntry:
    def test_default_fields(self):
        we = WatchlistEntry(ticker="AAPL", run_id="abc", recommendation="Overweight")
        assert we.ticker == "AAPL"
        assert we.recommendation == "Overweight"
        assert we.confidence == 0.0
        assert we.competitors == []

    def test_full_fields(self):
        we = WatchlistEntry(
            ticker="MSFT",
            run_id="xyz",
            recommendation="Hold",
            confidence=0.65,
            composite_score=58.3,
            risk_level="Low",
            generated_at="2024-11-01",
            competitors=["GOOGL", "AMZN"],
            granite_mode="live",
            notes="Strong fundamentals",
        )
        assert we.composite_score == 58.3
        assert "GOOGL" in we.competitors
        assert we.notes == "Strong fundamentals"

    def test_serialization(self):
        we = WatchlistEntry(ticker="AAPL", run_id="r1", recommendation="Underweight")
        d = we.model_dump()
        assert d["ticker"] == "AAPL"
        assert d["recommendation"] == "Underweight"


# ── PortfolioPosition ─────────────────────────────────────────────────────────

class TestPortfolioPosition:
    def test_valid_position(self):
        p = PortfolioPosition(ticker="AAPL", weight_pct=40.0)
        assert p.ticker == "AAPL"
        assert p.weight_pct == 40.0

    def test_weight_bounds(self):
        with pytest.raises(Exception):
            PortfolioPosition(ticker="AAPL", weight_pct=101.0)
        with pytest.raises(Exception):
            PortfolioPosition(ticker="AAPL", weight_pct=-1.0)

    def test_zero_weight_allowed(self):
        p = PortfolioPosition(ticker="AAPL", weight_pct=0.0)
        assert p.weight_pct == 0.0


# ── PortfolioConfig ───────────────────────────────────────────────────────────

class TestPortfolioConfig:
    def _make_config(self) -> PortfolioConfig:
        return PortfolioConfig(
            positions=[
                PortfolioPosition(ticker="AAPL", weight_pct=40.0),
                PortfolioPosition(ticker="MSFT", weight_pct=30.0),
                PortfolioPosition(ticker="GOOGL", weight_pct=30.0),
            ],
            name="Test Portfolio",
        )

    def test_tickers_property(self):
        pc = self._make_config()
        assert pc.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_total_weight(self):
        pc = self._make_config()
        assert abs(pc.total_weight - 100.0) < 0.01

    def test_normalised_weights_sum_to_one(self):
        pc = self._make_config()
        weights = pc.normalised_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_normalised_weights_proportional(self):
        pc = self._make_config()
        weights = pc.normalised_weights()
        assert abs(weights["AAPL"] - 0.40) < 0.001
        assert abs(weights["MSFT"] - 0.30) < 0.001

    def test_normalised_weights_zero_total(self):
        pc = PortfolioConfig(positions=[
            PortfolioPosition(ticker="AAPL", weight_pct=0.0),
        ])
        weights = pc.normalised_weights()
        assert weights["AAPL"] == 0.0

    def test_empty_config(self):
        pc = PortfolioConfig()
        assert pc.tickers == []
        assert pc.total_weight == 0.0

    def test_name_default(self):
        pc = PortfolioConfig()
        assert pc.name == "My Portfolio"


# ── PortfolioAnalysisResult ───────────────────────────────────────────────────

class TestPortfolioAnalysisResult:
    def test_default_fields(self):
        pr = PortfolioAnalysisResult()
        assert pr.blended_recommendation == "Hold"
        assert pr.blended_confidence == 0.5
        assert pr.blended_risk_level == "Medium"
        assert pr.ticker_results == {}

    def test_full_fields(self):
        pr = PortfolioAnalysisResult(
            portfolio_name="Growth",
            positions=[
                PortfolioPosition(ticker="AAPL", weight_pct=60.0),
                PortfolioPosition(ticker="MSFT", weight_pct=40.0),
            ],
            weighted_composite_score=68.5,
            blended_recommendation="Overweight",
            blended_confidence=0.72,
            weighted_risk_score=42.0,
            blended_risk_level="Medium",
            ticker_results={
                "AAPL": {"recommendation": "Overweight", "confidence": 0.75, "composite_score": 70.0, "risk_score": 40.0},
                "MSFT": {"recommendation": "Overweight", "confidence": 0.70, "composite_score": 66.0, "risk_score": 45.0},
            },
            diversification_note="2-ticker portfolio",
        )
        assert pr.weighted_composite_score == 68.5
        assert pr.blended_recommendation == "Overweight"
        assert len(pr.ticker_results) == 2

    def test_serialization(self):
        pr = PortfolioAnalysisResult(
            blended_recommendation="Hold",
            weighted_composite_score=52.0,
        )
        d = pr.model_dump()
        assert d["blended_recommendation"] == "Hold"
        assert d["weighted_composite_score"] == 52.0


# ── InvestorProfile on PipelineState ─────────────────────────────────────────

class TestInvestorProfileOnState:
    def test_pipeline_state_has_investor_profile_field(self):
        ps = PipelineState(ticker="AAPL")
        assert hasattr(ps, "investor_profile")
        assert ps.investor_profile is None

    def test_attach_profile_to_state(self):
        ps = PipelineState(ticker="AAPL")
        ps.investor_profile = InvestorProfile(risk_tolerance="Conservative")
        assert ps.investor_profile is not None
        assert ps.investor_profile.risk_tolerance == "Conservative"

    def test_serialization_includes_profile(self):
        ps = PipelineState(ticker="AAPL")
        ps.investor_profile = InvestorProfile(risk_tolerance="Aggressive", sector_notes="Tech only")
        d = ps.model_dump()
        assert d["investor_profile"]["risk_tolerance"] == "Aggressive"
        assert d["investor_profile"]["sector_notes"] == "Tech only"

    def test_profile_none_serializes_cleanly(self):
        ps = PipelineState(ticker="AAPL")
        d = ps.model_dump()
        assert d["investor_profile"] is None


# ── Prompt personalisation ────────────────────────────────────────────────────

class TestPromptPersonalisation:
    def test_profile_context_injected_into_prompt(self):
        """Verify that _build_prompt appends the investor profile context."""
        from agents.recommendation_agent import _build_prompt
        from models.pipeline_state import PipelineState
        from models.investor_profile import InvestorProfile

        ps = PipelineState(ticker="AAPL")
        ps.investor_profile = InvestorProfile(risk_tolerance="Conservative")
        prompt = _build_prompt(ps)
        assert "Conservative" in prompt
        assert "INVESTOR PROFILE" in prompt

    def test_no_profile_no_context_injected(self):
        """With no profile, the prompt should not contain investor profile header."""
        from agents.recommendation_agent import _build_prompt
        from models.pipeline_state import PipelineState

        ps = PipelineState(ticker="AAPL")
        ps.investor_profile = None
        prompt = _build_prompt(ps)
        assert "INVESTOR PROFILE" not in prompt

    def test_aggressive_profile_in_prompt(self):
        from agents.recommendation_agent import _build_prompt
        from models.pipeline_state import PipelineState
        from models.investor_profile import InvestorProfile

        ps = PipelineState(ticker="MSFT")
        ps.investor_profile = InvestorProfile(risk_tolerance="Aggressive", sector_notes="Tech growth")
        prompt = _build_prompt(ps)
        assert "Aggressive" in prompt
        assert "Tech growth" in prompt

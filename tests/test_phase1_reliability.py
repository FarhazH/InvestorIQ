"""
Phase 1 unit tests: watsonx client, grounding check, fallbacks, validator.
"""
import sys
import os
import json
import pytest

os.environ["DEMO_MODE"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── watsonx client tests ──────────────────────────────────────────────────────

class TestCallGraniteJson:
    """call_granite_json now returns (dict, mode) tuple."""

    def test_returns_tuple_in_demo_mode(self):
        from services.watsonx_client import call_granite_json
        result = call_granite_json("Generate investment recommendation for AAPL", 512)
        assert isinstance(result, tuple), "Should return (dict, mode) tuple"
        d, mode = result
        assert isinstance(d, dict)
        assert mode == "demo"

    def test_recommendation_stub_valid(self):
        from services.watsonx_client import call_granite_json, validate_recommendation_json
        d, mode = call_granite_json(
            "You are a senior equity research analyst. Recommend AAPL.",
            512,
            validator=validate_recommendation_json,
        )
        assert d["recommendation"] in ("Overweight", "Hold", "Underweight")
        assert isinstance(d["reasons"], list)
        assert len(d["reasons"]) >= 1

    def test_sentiment_stub_valid(self):
        from services.watsonx_client import call_granite_json, validate_sentiment_json
        d, mode = call_granite_json(
            "You are a financial sentiment analyst. Analyze news about AAPL.",
            512,
            validator=validate_sentiment_json,
        )
        assert "items" in d
        assert "aggregate_score" in d
        assert isinstance(d["items"], list)

    def test_json_parse_fence_stripping(self):
        from services.watsonx_client import _parse_json_response
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_parse_raises_on_garbage(self):
        from services.watsonx_client import _parse_json_response
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_json_response("This is not JSON at all.")

    def test_json_parse_nested_braces(self):
        from services.watsonx_client import _parse_json_response
        raw = 'Some text before {"a": {"b": 1}} and after'
        result = _parse_json_response(raw)
        assert result["a"]["b"] == 1


class TestValidators:
    def test_validate_recommendation_passes(self):
        from services.watsonx_client import validate_recommendation_json
        good = {
            "recommendation": "Overweight",
            "confidence": 0.8,
            "reasons": [{"point": "Strong growth"}],
        }
        validate_recommendation_json(good)  # should not raise

    def test_validate_recommendation_bad_value(self):
        from services.watsonx_client import validate_recommendation_json
        with pytest.raises(ValueError, match="Invalid recommendation"):
            validate_recommendation_json({
                "recommendation": "BUY",
                "confidence": 0.8,
                "reasons": [{"point": "x"}],
            })

    def test_validate_recommendation_missing_field(self):
        from services.watsonx_client import validate_recommendation_json
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_recommendation_json({"recommendation": "Hold", "confidence": 0.5})

    def test_validate_recommendation_empty_reasons(self):
        from services.watsonx_client import validate_recommendation_json
        with pytest.raises(ValueError, match="non-empty list"):
            validate_recommendation_json({
                "recommendation": "Hold",
                "confidence": 0.5,
                "reasons": [],
            })

    def test_validate_sentiment_passes(self):
        from services.watsonx_client import validate_sentiment_json
        good = {"items": [{"headline": "x"}], "aggregate_score": 0.5}
        validate_sentiment_json(good)  # should not raise

    def test_validate_sentiment_missing_items(self):
        from services.watsonx_client import validate_sentiment_json
        with pytest.raises(ValueError, match="Missing 'items'"):
            validate_sentiment_json({"aggregate_score": 0.5})

    def test_validate_sentiment_items_not_list(self):
        from services.watsonx_client import validate_sentiment_json
        with pytest.raises(ValueError, match="must be a list"):
            validate_sentiment_json({"items": "not a list", "aggregate_score": 0.5})


# ── Grounding check tests ─────────────────────────────────────────────────────

class TestGroundingCheck:
    def _make_state(self):
        """Minimal pipeline state with known computed values."""
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        return orch.run_until("AAPL", [], stop_after="risk_trend")

    def test_extract_numeric_percent(self):
        from agents.recommendation_agent import _extract_numeric
        assert _extract_numeric("8.1%") == pytest.approx(8.1)

    def test_extract_numeric_dollar(self):
        from agents.recommendation_agent import _extract_numeric
        assert _extract_numeric("$28.5") == pytest.approx(28.5)

    def test_extract_numeric_none(self):
        from agents.recommendation_agent import _extract_numeric
        assert _extract_numeric(None) is None

    def test_extract_numeric_text_only(self):
        from agents.recommendation_agent import _extract_numeric
        assert _extract_numeric("N/A") is None

    def test_grounding_check_match(self):
        from agents.recommendation_agent import _grounding_check
        from models.recommendation import RecommendationReason
        # Cited 25.3%, computed 25.5% → within 10% tolerance → match
        metrics_cited = {"net_margin": "25.3%"}
        reasons = []
        computed = {"net_margin": 25.5}
        flags = _grounding_check(metrics_cited, reasons, computed, tolerance_pct=10.0)
        assert len(flags) == 1
        assert flags[0].match is True

    def test_grounding_check_mismatch(self):
        from agents.recommendation_agent import _grounding_check
        # Cited 50%, computed 10 → >10% difference → no match
        metrics_cited = {"pe_ratio": "50"}
        reasons = []
        computed = {"pe_ratio": 10.0}
        flags = _grounding_check(metrics_cited, reasons, computed, tolerance_pct=10.0)
        assert len(flags) == 1
        assert flags[0].match is False

    def test_grounding_check_no_computed(self):
        from agents.recommendation_agent import _grounding_check
        # Metric not in computed → match=False, computed_value=None
        metrics_cited = {"unknown_metric": "42"}
        flags = _grounding_check(metrics_cited, [], {})
        assert flags[0].match is False
        assert flags[0].computed_value is None

    def test_grounding_flags_on_full_pipeline(self):
        """Full demo pipeline should have grounding_flags populated."""
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        state = orch.run("AAPL", [])
        assert state.recommendation is not None
        # Demo mode uses pre-baked cited values; grounding flags should be present
        assert isinstance(state.recommendation.grounding_flags, list)
        assert len(state.recommendation.grounding_flags) > 0

    def test_recommendation_has_granite_mode(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        state = orch.run("AAPL", [])
        assert state.recommendation.granite_mode in ("live", "demo", "fallback")

    def test_pipeline_granite_mode_set(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        state = orch.run("AAPL", [])
        assert state.granite_mode in ("live", "demo", "fallback")


# ── Rule-based fallback tests ─────────────────────────────────────────────────

class TestRuleBasedFallback:
    def test_fallback_recommendation_valid(self):
        from pipeline.orchestrator import PipelineOrchestrator
        from agents.recommendation_agent import _rule_based_recommendation
        orch = PipelineOrchestrator()
        state = orch.run_until("AAPL", [], stop_after="risk_trend")
        rec = _rule_based_recommendation(state)
        assert rec.recommendation in ("Overweight", "Hold", "Underweight")
        assert 0.0 <= rec.confidence <= 1.0
        assert rec.granite_mode == "fallback"
        assert len(rec.reasons) >= 1

    def test_fallback_uses_computed_values(self):
        from pipeline.orchestrator import PipelineOrchestrator
        from agents.recommendation_agent import _rule_based_recommendation
        orch = PipelineOrchestrator()
        state = orch.run_until("AAPL", [], stop_after="risk_trend")
        rec = _rule_based_recommendation(state)
        # All reason values should come from computed data, not Granite
        for r in rec.reasons:
            assert r.metric != "", "Each reason should reference a metric"
            assert r.value != "", "Each reason should have a computed value"

    def test_keyword_sentiment_positive(self):
        from agents.sentiment_agent import _keyword_sentiment
        label, conf = _keyword_sentiment("Apple reports record revenue growth profit beat")
        assert label == "positive"
        assert conf > 0.5

    def test_keyword_sentiment_negative(self):
        from agents.sentiment_agent import _keyword_sentiment
        label, conf = _keyword_sentiment("Sales decline loss layoff risk warning concern")
        assert label == "negative"
        assert conf > 0.5

    def test_keyword_sentiment_neutral(self):
        from agents.sentiment_agent import _keyword_sentiment
        label, conf = _keyword_sentiment("Apple announced a new product yesterday")
        assert label == "neutral"


# ── Exponential backoff test ──────────────────────────────────────────────────

class TestExponentialBackoff:
    def test_backoff_sequence(self):
        """BaseAgent._with_retry should use exponential backoff delays."""
        import time
        from agents.base_agent import BaseAgent
        from models.pipeline_state import PipelineState

        class TestAgent(BaseAgent):
            name = "test"
            def run(self, state):
                return state

        agent = TestAgent()
        agent.max_retries = 3
        agent.retry_delay = 0.001  # tiny for testing

        call_count = 0
        delays = []
        original_sleep = time.sleep

        def mock_sleep(s):
            delays.append(s)

        time.sleep = mock_sleep
        try:
            with pytest.raises(RuntimeError):
                def always_fails():
                    nonlocal call_count
                    call_count += 1
                    raise ValueError("fail")
                agent._with_retry(always_fails)
        finally:
            time.sleep = original_sleep

        assert call_count == 3
        # Delays should be exponential: 0.001, 0.002
        assert len(delays) == 2
        assert delays[1] == pytest.approx(delays[0] * 2, rel=0.01)

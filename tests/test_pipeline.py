"""
Integration test — runs the full pipeline in demo mode (no API keys required).
Verifies all 9 agents complete and the final report is populated.
"""
import sys
import os
import pytest

# Force demo mode before any imports
os.environ["DEMO_MODE"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def pipeline_state():
    """Run a full AAPL + MSFT, GOOGL pipeline in demo mode."""
    from pipeline.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator()
    state = orch.run("AAPL", ["MSFT", "GOOGL"])
    return state


class TestFullPipeline:
    def test_pipeline_completes_without_critical_error(self, pipeline_state):
        assert not pipeline_state.has_critical_error, \
            f"Pipeline had critical errors: {pipeline_state.errors}"

    def test_all_steps_not_failed(self, pipeline_state):
        from models.pipeline_state import StepStatus
        failed = [s.name for s in pipeline_state.steps if s.status == StepStatus.FAILED]
        assert not failed, f"Failed steps: {failed}"

    def test_raw_data_populated(self, pipeline_state):
        assert pipeline_state.raw_data is not None
        assert pipeline_state.raw_data.primary.ticker == "AAPL"
        assert len(pipeline_state.raw_data.competitors) == 2

    def test_validation_passed(self, pipeline_state):
        assert pipeline_state.validation is not None
        assert pipeline_state.validation.overall_passed is True
        assert pipeline_state.validation.primary.quality_score > 0.5

    def test_kpis_populated(self, pipeline_state):
        assert pipeline_state.kpis is not None
        kpis = pipeline_state.kpis.primary
        assert kpis.revenue.current is not None
        assert kpis.net_margin is not None
        assert kpis.eps_ttm is not None

    def test_ratios_populated(self, pipeline_state):
        assert pipeline_state.ratios is not None
        r = pipeline_state.ratios.primary
        assert r.pe_ratio is not None
        assert r.beta is not None
        # RSI may be None if not enough price data, but most should be set
        assert r.volatility_annualised is not None

    def test_benchmark_populated(self, pipeline_state):
        assert pipeline_state.benchmark is not None
        assert pipeline_state.benchmark.overall_rank is not None
        assert len(pipeline_state.benchmark.peer_comparison_table) == 3  # AAPL + 2 peers
        assert len(pipeline_state.benchmark.rankings) > 0

    def test_sentiment_populated(self, pipeline_state):
        assert pipeline_state.sentiment is not None
        s = pipeline_state.sentiment
        assert s.total_articles > 0
        assert s.sentiment_label in ("Bullish", "Neutral", "Bearish")
        assert -1.0 <= s.aggregate_score <= 1.0

    def test_risk_populated(self, pipeline_state):
        assert pipeline_state.risk is not None
        risk = pipeline_state.risk
        assert 0 <= risk.overall_risk_score <= 100
        assert risk.overall_risk_level in ("Low", "Medium", "High")
        assert len(risk.indicators) == 5

    def test_recommendation_populated(self, pipeline_state):
        assert pipeline_state.recommendation is not None
        rec = pipeline_state.recommendation
        assert rec.recommendation in ("Overweight", "Hold", "Underweight")
        assert 0.0 <= rec.confidence <= 1.0
        assert len(rec.reasons) >= 1
        assert "NOT financial advice" in rec.disclaimer

    def test_report_populated(self, pipeline_state):
        assert pipeline_state.report is not None
        report = pipeline_state.report
        assert report.ticker == "AAPL"
        assert report.run_id == pipeline_state.run_id
        assert report.json_path != ""
        assert len(report.recommendation) > 0

    def test_step_timings_recorded(self, pipeline_state):
        timed = [s for s in pipeline_state.steps if s.duration_s is not None]
        assert len(timed) > 0

    def test_competitors_in_report(self, pipeline_state):
        assert "MSFT" in pipeline_state.report.competitors or "GOOGL" in pipeline_state.report.competitors

    def test_no_negative_prices_in_forecast(self, pipeline_state):
        if pipeline_state.risk and pipeline_state.risk.price_forecast:
            for pt in pipeline_state.risk.price_forecast:
                assert pt.value > 0, "Forecast prices should be positive"


class TestPartialPipeline:
    def test_run_until_validation(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        state = orch.run_until("AAPL", ["MSFT"], stop_after="validation")
        assert state.validation is not None
        assert state.kpis is None  # should stop before KPI

    def test_run_until_kpis(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        state = orch.run_until("MSFT", [], stop_after="kpi_extraction")
        assert state.kpis is not None
        assert state.ratios is None

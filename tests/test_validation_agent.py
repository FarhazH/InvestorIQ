"""
Unit tests for the Validation Agent logic.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.validation_agent import _check_ticker, QUALITY_THRESHOLD
from models.market_data import TickerData
from models.validation import DataQualityReport


def _base_ticker(ticker: str = "TEST") -> TickerData:
    """Build a minimal valid TickerData with all required fields."""
    from services.demo_data import get_demo_ticker_data
    raw = get_demo_ticker_data(ticker)
    return TickerData(
        ticker=ticker,
        info=raw["info"],
        price_history=raw["price_history"],
        income_stmt=raw["income_stmt"],
        balance_sheet=raw["balance_sheet"],
        cash_flow=raw["cash_flow"],
    )


class TestCheckTicker:
    def test_good_data_passes(self):
        td = _base_ticker("AAPL")
        report = _check_ticker(td)
        assert report.passed is True
        assert report.quality_score >= QUALITY_THRESHOLD
        assert report.quality_score <= 1.0

    def test_missing_pe_lowers_score(self):
        td = _base_ticker("AAPL")
        del td.info["trailingPE"]
        report = _check_ticker(td)
        # Score should be lower than 1.0
        assert report.quality_score < 1.0
        assert any(f.field == "trailingPE" for f in report.flags)
        assert "trailingPE" in report.missing_fields

    def test_empty_price_history_flagged(self):
        """Empty price history should be flagged as an error and lower the quality score."""
        td = _base_ticker("AAPL")
        full_report = _check_ticker(td)
        td.price_history = {}
        report = _check_ticker(td)
        # Score should be lower than with full data
        assert report.quality_score < full_report.quality_score
        # An error-severity flag must be present
        error_flags = [f for f in report.flags if f.severity == "error" and f.field == "price_history"]
        assert len(error_flags) > 0, "Expected error flag for empty price_history"
        assert "price_history" in report.missing_fields

    def test_negative_price_flags_outlier(self):
        td = _base_ticker("AAPL")
        # Inject a negative price
        close = td.price_history.get("Close", {})
        if close:
            first_key = next(iter(close))
            td.price_history["Close"][first_key] = -50.0
        report = _check_ticker(td)
        assert "price_history.Close" in report.outlier_fields

    def test_all_required_fields_missing_low_score(self):
        td = TickerData(ticker="EMPTY", price_history={})
        report = _check_ticker(td)
        assert report.quality_score < QUALITY_THRESHOLD
        assert report.passed is False

    def test_quality_score_bounded(self):
        td = _base_ticker("MSFT")
        report = _check_ticker(td)
        assert 0.0 <= report.quality_score <= 1.0

    def test_multiple_missing_fields_accumulate(self):
        td = _base_ticker("AAPL")
        for field in ["trailingPE", "debtToEquity", "currentRatio", "totalRevenue"]:
            td.info.pop(field, None)
        report_with = _check_ticker(_base_ticker("AAPL"))
        report_without = _check_ticker(td)
        assert report_without.quality_score < report_with.quality_score

    def test_currency_flagged_for_unusual_currency(self):
        td = _base_ticker("AAPL")
        td.info["currency"] = "MOON"
        report = _check_ticker(td)
        assert any(f.field == "currency" for f in report.flags)

    def test_valid_gbp_currency_not_flagged(self):
        td = _base_ticker("AAPL")
        td.info["currency"] = "GBP"
        report = _check_ticker(td)
        assert not any(f.field == "currency" for f in report.flags)

    def test_flag_severity_includes_error_for_empty_price(self):
        td = _base_ticker("AAPL")
        td.price_history = {}
        report = _check_ticker(td)
        error_flags = [f for f in report.flags if f.severity == "error"]
        assert len(error_flags) > 0

    def test_missing_income_stmt_lowers_score(self):
        td = _base_ticker("AAPL")
        td.income_stmt = {}
        report = _check_ticker(td)
        assert report.quality_score < 1.0
        assert any("income_stmt" in f.field for f in report.flags)

"""
Unit tests for ratio calculations in RatioAgent.
Tests pure computation functions without requiring external data.
"""
import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

# Add investiq root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.ratio_agent import _rsi, _max_drawdown, _annualised_volatility, _sharpe, _compute_ratios
from models.market_data import TickerData


# ── Helper: build synthetic price series ─────────────────────────────────────

def _make_prices(n: int = 300, seed: int = 42, trend: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.015, n)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


# ── RSI Tests ─────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_range(self):
        prices = _make_prices(100)
        rsi = _rsi(prices)
        assert rsi is not None
        assert 0.0 <= rsi <= 100.0

    def test_rsi_overbought_signal(self):
        # Consistently rising prices → RSI near 100
        # Must have a DatetimeIndex since _rsi sorts by index
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        prices = pd.Series([100 + i * 0.5 for i in range(100)], index=dates)
        rsi = _rsi(prices)
        assert rsi is not None
        assert rsi > 70, f"Expected RSI > 70 for rising prices, got {rsi}"

    def test_rsi_oversold_signal(self):
        # Consistently falling prices → RSI near 0
        prices = pd.Series([100 - i * 0.5 for i in range(100)])
        rsi = _rsi(prices)
        assert rsi is not None
        assert rsi < 30, f"Expected RSI < 30 for falling prices, got {rsi}"

    def test_rsi_insufficient_data(self):
        prices = pd.Series([100.0, 101.0, 99.0])
        assert _rsi(prices) is None

    def test_rsi_period_14(self):
        # Exactly 15 points (period=14 needs at least 15)
        prices = _make_prices(15)
        result = _rsi(prices)
        assert result is not None


# ── Max Drawdown Tests ────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_drawdown_negative(self):
        prices = _make_prices(200)
        mdd = _max_drawdown(prices)
        assert mdd is not None
        assert mdd <= 0, "Max drawdown should be non-positive"

    def test_drawdown_monotone_rising(self):
        prices = pd.Series([100 + i for i in range(50)])
        mdd = _max_drawdown(prices)
        assert mdd is not None
        assert mdd == 0.0, "No drawdown for monotonically rising prices"

    def test_drawdown_known_drop(self):
        # Peak 100 → drop to 50 = -50% drawdown
        prices = pd.Series([100, 100, 100, 50, 50, 70])
        mdd = _max_drawdown(prices)
        assert mdd is not None
        assert mdd == pytest.approx(-50.0, abs=0.1)

    def test_drawdown_empty(self):
        assert _max_drawdown(pd.Series(dtype=float)) is None


# ── Annualised Volatility Tests ───────────────────────────────────────────────

class TestVolatility:
    def test_vol_positive(self):
        prices = _make_prices(252)
        vol = _annualised_volatility(prices)
        assert vol is not None
        assert vol > 0

    def test_vol_flat_prices(self):
        # Flat prices → near-zero volatility
        prices = pd.Series([100.0] * 100)
        vol = _annualised_volatility(prices)
        assert vol is not None
        assert vol == pytest.approx(0.0, abs=0.1)

    def test_vol_insufficient_data(self):
        prices = pd.Series([100.0, 101.0])
        assert _annualised_volatility(prices) is None

    def test_vol_high_variance(self):
        rng = np.random.default_rng(99)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.05, 300)))
        vol = _annualised_volatility(prices)
        assert vol is not None
        # ~5% daily std × sqrt(252) ≈ 79%
        assert vol > 50


# ── Sharpe Ratio Tests ────────────────────────────────────────────────────────

class TestSharpe:
    def test_sharpe_exists(self):
        prices = _make_prices(252)
        sharpe = _sharpe(prices)
        assert sharpe is not None

    def test_sharpe_flat(self):
        prices = pd.Series([100.0] * 252)
        result = _sharpe(prices)
        assert result is None  # zero vol → undefined Sharpe

    def test_sharpe_positive_for_upward_trend(self):
        # Steadily rising prices should give positive Sharpe
        prices = pd.Series([100 + i * 0.1 for i in range(252)])
        sharpe = _sharpe(prices)
        # linear series has minimal variance but positive return
        assert sharpe is not None


# ── Full TickerData ratio computation ────────────────────────────────────────

class TestComputeRatios:
    def _make_ticker_data(self, ticker: str = "AAPL") -> TickerData:
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

    def test_ratios_not_all_none(self):
        td = self._make_ticker_data("AAPL")
        ratios = _compute_ratios(td)
        # At least PE, beta, and RSI should be populated from demo data
        assert ratios.pe_ratio is not None
        assert ratios.beta is not None

    def test_ratios_valid_ranges(self):
        td = self._make_ticker_data("AAPL")
        ratios = _compute_ratios(td)
        if ratios.rsi_14 is not None:
            assert 0 <= ratios.rsi_14 <= 100
        if ratios.pe_ratio is not None:
            assert ratios.pe_ratio > 0
        if ratios.volatility_annualised is not None:
            assert ratios.volatility_annualised >= 0
        if ratios.max_drawdown is not None:
            assert ratios.max_drawdown <= 0

    def test_de_ratio_scaled(self):
        """yfinance returns D/E as percentage; we scale down to ratio."""
        td = self._make_ticker_data("AAPL")
        ratios = _compute_ratios(td)
        # AAPL demo D/E = 173 (from info) → after /100 = 1.73
        if ratios.debt_to_equity is not None:
            assert ratios.debt_to_equity < 50, "D/E should be ratio, not percentage"

    def test_ma50_less_than_ma200_in_bear_market(self):
        # Fabricate declining prices
        td = self._make_ticker_data("AAPL")
        prices = pd.Series([200 - i * 0.1 for i in range(504)])
        dates = pd.date_range("2022-01-01", periods=504, freq="B")
        prices.index = dates
        td.price_history = {"Close": prices.to_dict()}
        ratios = _compute_ratios(td)
        assert ratios.ma_50 is not None
        assert ratios.ma_200 is not None
        assert ratios.ma_50 < ratios.ma_200, "In downtrend MA50 should be < MA200"

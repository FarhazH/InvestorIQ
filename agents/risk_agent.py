"""
Agent 7 — Risk & Trend Agent.
Detects declining margins, rising debt, negative cash flow, high volatility,
sentiment drops. Produces risk scores (Low/Med/High) per category.
Includes ARIMA-based revenue and price forecasting.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.ratios import TickerRatios
from models.kpi import TickerKPIs
from models.sentiment import SentimentResult
from models.risk import RiskResult, RiskIndicator, ForecastPoint

logger = logging.getLogger(__name__)


# ── Risk scoring helpers ──────────────────────────────────────────────────────

def _level(score: float) -> str:
    if score >= 65:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _assess_margin_risk(kpis: TickerKPIs) -> RiskIndicator:
    score = 20.0
    desc = "Margins appear stable."
    gross = kpis.gross_margin or 0.0
    op = kpis.operating_margin or 0.0
    net = kpis.net_margin or 0.0

    if net < 0:
        score += 50
        desc = "Negative net margin — company is losing money."
    elif net < 0.05:
        score += 30
        desc = "Very thin net margin (<5%)."
    elif net < 0.10:
        score += 15
        desc = f"Below-average net margin ({net:.1%})."

    if op < 0:
        score += 20

    rev_growth = kpis.revenue_growth_yoy or 0
    if rev_growth < -5:
        score += 20
        desc += " Revenue declining YoY."

    return RiskIndicator(category="Margins", level=_level(score), score=min(score, 100), description=desc)


def _assess_debt_risk(ratios: TickerRatios, kpis: TickerKPIs) -> RiskIndicator:
    score = 20.0
    desc = "Debt levels appear manageable."
    de = ratios.debt_to_equity
    ic = ratios.interest_coverage

    if de is not None:
        if de > 3.0:
            score += 40; desc = f"Very high D/E ratio ({de:.2f})."
        elif de > 1.5:
            score += 25; desc = f"Elevated D/E ratio ({de:.2f})."
        elif de > 0.8:
            score += 10

    if ic is not None:
        if ic < 1.5:
            score += 35; desc += " Interest barely covered."
        elif ic < 3.0:
            score += 15; desc += " Moderate interest coverage."

    net_debt = kpis.net_debt
    if net_debt is not None and net_debt > 0:
        ebitda = kpis.ebitda
        if ebitda and ebitda > 0 and (net_debt / ebitda) > 4:
            score += 20; desc += " Net Debt/EBITDA > 4x."

    return RiskIndicator(category="Debt", level=_level(score), score=min(score, 100), description=desc)


def _assess_cashflow_risk(kpis: TickerKPIs) -> RiskIndicator:
    score = 20.0
    desc = "Free cash flow is positive."
    fcf = kpis.free_cash_flow

    if fcf is None:
        score += 20; desc = "Free cash flow data unavailable."
    elif fcf < 0:
        score += 55; desc = f"Negative free cash flow ({fcf:,.0f})."
    elif kpis.revenue.current and kpis.revenue.current > 0:
        fcf_margin = fcf / kpis.revenue.current
        if fcf_margin < 0.03:
            score += 25; desc = f"Very low FCF margin ({fcf_margin:.1%})."
        elif fcf_margin < 0.08:
            score += 10; desc = f"Modest FCF margin ({fcf_margin:.1%})."

    return RiskIndicator(category="Cash Flow", level=_level(score), score=min(score, 100), description=desc)


def _assess_volatility_risk(ratios: TickerRatios) -> RiskIndicator:
    score = 20.0
    desc = "Volatility within normal range."
    vol = ratios.volatility_annualised
    beta = ratios.beta
    mdd = ratios.max_drawdown

    if vol is not None:
        if vol > 50:
            score += 40; desc = f"Extreme volatility ({vol:.1f}% annualised)."
        elif vol > 30:
            score += 25; desc = f"High volatility ({vol:.1f}% annualised)."
        elif vol > 20:
            score += 10

    if beta is not None:
        if beta > 1.5:
            score += 15
        elif beta < 0:
            score += 10

    if mdd is not None:
        if mdd < -50:
            score += 20; desc += f" Max drawdown {mdd:.1f}%."
        elif mdd < -30:
            score += 10; desc += f" Significant drawdown {mdd:.1f}%."

    return RiskIndicator(category="Volatility", level=_level(score), score=min(score, 100), description=desc)


def _assess_sentiment_risk(sentiment: SentimentResult | None) -> RiskIndicator:
    if sentiment is None:
        return RiskIndicator(category="Sentiment", level="Medium", score=50, description="Sentiment data not available.")
    score = sentiment.aggregate_score  # -1.0 to 1.0
    # Map to risk: negative sentiment = higher risk
    risk_score = round((1 - score) / 2 * 60 + 20, 1)  # 20-80 range
    desc = f"Sentiment: {sentiment.sentiment_label} (score={score:.2f}). "
    if sentiment.top_themes:
        desc += f"Key themes: {', '.join(sentiment.top_themes[:3])}."
    return RiskIndicator(
        category="Sentiment",
        level=_level(risk_score),
        score=min(risk_score, 100),
        description=desc,
    )


# ── Forecasting ───────────────────────────────────────────────────────────────

def _arima_forecast(series: list[float], periods: int = 4, labels: list[str] | None = None) -> list[ForecastPoint]:
    if len(series) < 8:
        return []
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=(1, 1, 1))
            fit = model.fit()
            forecast = fit.get_forecast(steps=periods)
            mean = forecast.predicted_mean
            conf = forecast.conf_int(alpha=0.2)
            # Support both pandas Series/DataFrame and plain ndarray returns
            import numpy as np
            mean_arr = np.asarray(mean).flatten()
            conf_arr = np.asarray(conf)
            points = []
            for i in range(periods):
                label = labels[i] if labels and i < len(labels) else f"P+{i+1}"
                points.append(ForecastPoint(
                    period=label,
                    value=round(float(mean_arr[i]), 2),
                    lower=round(float(conf_arr[i, 0]), 2),
                    upper=round(float(conf_arr[i, 1]), 2),
                ))
            return points
    except Exception as exc:
        logger.warning("ARIMA forecast failed: %s", exc)
        return _linear_forecast(series, periods, labels)


def _linear_forecast(series: list[float], periods: int = 4, labels: list[str] | None = None) -> list[ForecastPoint]:
    """Simple linear extrapolation fallback."""
    if len(series) < 2:
        return []
    x = np.arange(len(series))
    y = np.array(series, dtype=float)
    # Remove NaN
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return []
    coeffs = np.polyfit(x[mask], y[mask], 1)
    slope, intercept = coeffs
    points = []
    for i in range(1, periods + 1):
        label = labels[i-1] if labels and (i-1) < len(labels) else f"P+{i}"
        val = slope * (len(series) + i - 1) + intercept
        points.append(ForecastPoint(period=label, value=round(float(val), 2)))
    return points


def _build_revenue_forecast(kpis: TickerKPIs) -> list[ForecastPoint]:
    # Build a mini revenue series from statement data if available
    # Using single-point revenue + growth rate for estimation
    rev = kpis.revenue.current
    if rev is None:
        return []
    growth = (kpis.revenue_growth_yoy or 5.0) / 100
    series = [rev * (1 - growth) ** i for i in range(3, -1, -1)]
    labels = ["FY+1", "FY+2", "FY+3", "FY+4"]
    return _linear_forecast(series, 4, labels)


def _build_price_forecast(td_price_history: dict) -> list[ForecastPoint]:
    close = td_price_history.get("Close", {})
    if not close:
        return []
    vals = list(close.values())
    if not vals:
        return []
    # Sample every ~5 days (weekly)
    series = [float(v) for v in vals if v is not None][::5]
    if len(series) < 8:
        return []
    labels = [f"W+{i}" for i in range(1, 9)]
    return _arima_forecast(series, 8, labels)


class RiskAgent(BaseAgent):
    name = "risk_trend"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        if state.ratios is None or state.kpis is None:
            self._fail_step(state, "Missing ratios or KPIs for risk analysis")
            return state

        try:
            ratios = state.ratios.primary
            kpis = state.kpis.primary
            sentiment = state.sentiment

            margin_risk = _assess_margin_risk(kpis)
            debt_risk = _assess_debt_risk(ratios, kpis)
            cf_risk = _assess_cashflow_risk(kpis)
            vol_risk = _assess_volatility_risk(ratios)
            sent_risk = _assess_sentiment_risk(sentiment)

            indicators = [margin_risk, debt_risk, cf_risk, vol_risk, sent_risk]
            overall_score = round(sum(i.score for i in indicators) / len(indicators), 1)

            # Forecasts
            rev_forecast = _build_revenue_forecast(kpis)
            price_history = state.raw_data.primary.price_history if state.raw_data else {}
            price_forecast = _build_price_forecast(price_history)

            # Determine forecast method used
            method = "ARIMA" if price_forecast else "Linear"

            state.risk = RiskResult(
                ticker=state.ticker,
                indicators=indicators,
                overall_risk_score=overall_score,
                overall_risk_level=_level(overall_score),
                declining_margins=margin_risk.level in ("Medium", "High"),
                rising_debt=debt_risk.level in ("Medium", "High"),
                negative_cash_flow=kpis.free_cash_flow is not None and kpis.free_cash_flow < 0,
                high_volatility=vol_risk.level == "High",
                negative_sentiment=sentiment is not None and sentiment.aggregate_score < -0.2,
                revenue_forecast=rev_forecast,
                price_forecast=price_forecast,
                forecast_method=method,
                summary=(
                    f"Overall risk: {_level(overall_score)} ({overall_score:.0f}/100). "
                    f"Key concerns: {', '.join(i.category for i in indicators if i.level == 'High') or 'None'}."
                ),
            )
            self._finish_step(
                state,
                f"Risk score: {overall_score:.0f}/100 ({_level(overall_score)})"
            )
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

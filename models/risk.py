"""
Pydantic models for risk & trend analysis (Agent 7 output).
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class RiskIndicator(BaseModel):
    category: str
    level: str        # Low | Medium | High
    score: float      # 0-100
    description: str = ""


class ForecastPoint(BaseModel):
    period: str       # e.g. "2025Q1"
    value: float
    lower: float | None = None
    upper: float | None = None


class RiskResult(BaseModel):
    ticker: str
    indicators: list[RiskIndicator] = Field(default_factory=list)
    overall_risk_score: float = Field(ge=0, le=100)
    overall_risk_level: str = "Medium"   # Low | Medium | High

    # Specific risk flags
    declining_margins: bool = False
    rising_debt: bool = False
    negative_cash_flow: bool = False
    high_volatility: bool = False
    negative_sentiment: bool = False

    # Forecasts
    revenue_forecast: list[ForecastPoint] = Field(default_factory=list)
    price_forecast: list[ForecastPoint] = Field(default_factory=list)

    forecast_method: str = "ARIMA"
    summary: str = ""

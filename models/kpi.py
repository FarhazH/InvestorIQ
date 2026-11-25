"""
Pydantic models for KPI extraction (Agent 3 output).
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class YoYMetric(BaseModel):
    current: float | None = None
    prior_year: float | None = None
    yoy_change_pct: float | None = None


class TickerKPIs(BaseModel):
    ticker: str
    currency: str = "USD"
    # Revenue
    revenue: YoYMetric = Field(default_factory=YoYMetric)
    # Net Income
    net_income: YoYMetric = Field(default_factory=YoYMetric)
    # EBITDA
    ebitda: float | None = None
    # EPS (trailing twelve months)
    eps_ttm: float | None = None
    eps_yoy_change_pct: float | None = None
    # Free cash flow
    free_cash_flow: float | None = None
    fcf_yoy_change_pct: float | None = None
    # Margins
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    # Debt
    total_debt: float | None = None
    total_cash: float | None = None
    net_debt: float | None = None
    # Growth labels
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None


class KPIResult(BaseModel):
    primary: TickerKPIs
    competitors: list[TickerKPIs] = Field(default_factory=list)

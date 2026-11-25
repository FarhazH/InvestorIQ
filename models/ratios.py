"""
Pydantic models for ratio analysis (Agent 4 output).
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class TickerRatios(BaseModel):
    ticker: str
    # Valuation
    pe_ratio: float | None = None
    forward_pe: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    # Profitability
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    # Leverage
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    # Liquidity
    current_ratio: float | None = None
    # Technical / market
    beta: float | None = None
    rsi_14: float | None = None
    ma_50: float | None = None
    ma_200: float | None = None
    price_vs_ma50_pct: float | None = None
    price_vs_ma200_pct: float | None = None
    volatility_annualised: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None
    # Dividends
    dividend_yield: float | None = None


class RatioResult(BaseModel):
    primary: TickerRatios
    competitors: list[TickerRatios] = Field(default_factory=list)

"""
Pydantic models for Phase 2 advanced analysis outputs:
- DCFResult: DCF-lite fair value range
- AltmanZResult: Altman Z-score (bankruptcy risk)
- ValuationPercentile: peer-relative percentile per metric
- WeightedScore: composite scoring model
- BacktestResult: historical signal performance
- ForecastQuality: MAPE on hold-out period
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class DCFResult(BaseModel):
    ticker: str
    wacc: float                             # discount rate used
    terminal_growth_rate: float             # perpetuity growth rate
    projected_fcf: list[float]              # 5-year projected FCF
    terminal_value: float
    enterprise_value: float
    equity_value: float
    shares_outstanding: float
    fair_value_low: float                   # bear-case (wacc+1%)
    fair_value_mid: float                   # base-case
    fair_value_high: float                  # bull-case (wacc-1%)
    current_price: float | None = None
    upside_pct: float | None = None         # (fair_value_mid / current_price - 1) * 100
    method_note: str = "DCF-lite (5yr FCF projection + terminal value)"


class AltmanZResult(BaseModel):
    ticker: str
    z_score: float
    zone: str                               # "Safe" | "Grey" | "Distress"
    components: dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""
    # Altman Z-score zones: >2.99 Safe, 1.81-2.99 Grey, <1.81 Distress


class ValuationPercentile(BaseModel):
    metric: str
    ticker_value: float | None
    peer_values: dict[str, float | None]    # ticker → value
    percentile: float | None                # 0–100; 0=cheapest, 100=most expensive
    rank: int | None                        # 1=best-valued
    n_peers: int


class WeightedScore(BaseModel):
    ticker: str
    # Component scores (0–100 each)
    fundamentals_score: float = 0.0
    valuation_score: float = 0.0
    momentum_score: float = 0.0
    sentiment_score: float = 0.0
    risk_score: float = 0.0               # inverted: lower risk → higher score
    # Weights (default equal, sum to 1.0)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "fundamentals": 0.30,
        "valuation":    0.20,
        "momentum":     0.20,
        "sentiment":    0.15,
        "risk":         0.15,
    })
    composite_score: float = 0.0           # weighted average 0–100
    signal: str = "Hold"                   # Overweight | Hold | Underweight
    signal_confidence: float = 0.5


class BacktestTrade(BaseModel):
    date: str
    signal: str                            # "Overweight" | "Hold" | "Underweight"
    entry_price: float
    exit_price: float | None = None
    pnl_pct: float | None = None


class BacktestResult(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    strategy_return_pct: float
    buyhold_return_pct: float
    excess_return_pct: float
    n_trades: int
    win_rate: float                        # 0–1
    max_drawdown_pct: float
    trades: list[BacktestTrade] = Field(default_factory=list)
    # Daily equity curve for chart
    dates: list[str] = Field(default_factory=list)
    strategy_equity: list[float] = Field(default_factory=list)
    buyhold_equity: list[float] = Field(default_factory=list)
    note: str = ""


class ForecastQuality(BaseModel):
    ticker: str
    forecast_type: str                     # "price" | "revenue"
    method: str                            # "ARIMA" | "Linear"
    mape: float | None = None             # Mean Absolute Percentage Error on holdout
    holdout_periods: int = 0
    in_sample_mape: float | None = None


class AdvancedAnalysisResult(BaseModel):
    ticker: str
    dcf: DCFResult | None = None
    altman_z: AltmanZResult | None = None
    valuation_percentiles: list[ValuationPercentile] = Field(default_factory=list)
    weighted_score: WeightedScore | None = None
    backtest: BacktestResult | None = None
    price_forecast_quality: ForecastQuality | None = None
    revenue_forecast_quality: ForecastQuality | None = None

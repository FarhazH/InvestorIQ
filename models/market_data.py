"""
Pydantic models for raw market data (Agent 1 output).
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str = ""
    description: str = ""
    url: str = ""
    published_at: str = ""
    source: str = ""


class AnalystTargets(BaseModel):
    target_mean_price: float | None = None
    target_high_price: float | None = None
    target_low_price: float | None = None
    recommendation_mean: float | None = None
    recommendation_key: str | None = None
    number_of_analyst_opinions: int | None = None


class TickerData(BaseModel):
    ticker: str
    fetched_at: str = ""
    info: dict[str, Any] = Field(default_factory=dict)
    price_history: dict[str, Any] = Field(default_factory=dict)
    income_stmt: dict[str, Any] = Field(default_factory=dict)
    balance_sheet: dict[str, Any] = Field(default_factory=dict)
    cash_flow: dict[str, Any] = Field(default_factory=dict)
    analyst_ratings: dict[str, Any] = Field(default_factory=dict)
    analyst_targets: dict[str, Any] = Field(default_factory=dict)
    news: list[NewsItem] = Field(default_factory=list)
    sec_filings: list[dict[str, Any]] = Field(default_factory=list)


class RawMarketData(BaseModel):
    primary: TickerData
    competitors: list[TickerData] = Field(default_factory=list)
    cached: bool = False

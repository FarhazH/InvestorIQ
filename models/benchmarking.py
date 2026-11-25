"""
Pydantic models for competitor benchmarking (Agent 5 output).
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class PeerRanking(BaseModel):
    metric: str
    ticker: str
    value: float | None = None
    rank: int  # 1 = best
    total_peers: int


class BenchmarkResult(BaseModel):
    primary_ticker: str
    peer_tickers: list[str] = Field(default_factory=list)
    rankings: list[PeerRanking] = Field(default_factory=list)
    # Aggregate rank (lower = better)
    overall_rank: int | None = None
    over_performing_metrics: list[str] = Field(default_factory=list)
    under_performing_metrics: list[str] = Field(default_factory=list)
    peer_comparison_table: list[dict] = Field(default_factory=list)

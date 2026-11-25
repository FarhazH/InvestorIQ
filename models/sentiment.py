"""
Pydantic models for sentiment analysis (Agent 6 output).
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class ScoredNewsItem(BaseModel):
    headline: str = ""
    description: str = ""
    url: str = ""
    published_at: str = ""
    source: str = ""
    sentiment: str = "neutral"       # positive | neutral | negative
    confidence: float = 0.5
    key_theme: str = ""


class SentimentResult(BaseModel):
    ticker: str
    items: list[ScoredNewsItem] = Field(default_factory=list)
    aggregate_score: float = 0.0     # -1.0 (very bearish) to 1.0 (very bullish)
    sentiment_label: str = "Neutral" # Bullish | Neutral | Bearish
    top_themes: list[str] = Field(default_factory=list)
    total_articles: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0

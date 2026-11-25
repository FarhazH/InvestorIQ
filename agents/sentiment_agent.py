"""
Agent 6 — Sentiment Agent.
Uses IBM Granite to score each news headline/summary, then aggregates
into a sentiment index and key themes.

v2 improvements:
- Passes validate_sentiment_json validator to call_granite_json
- Sets state.granite_mode on success
- Rule-based fallback when Granite fails (keyword scoring)
- Structured _log_input / _log_output telemetry
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.market_data import NewsItem
from models.sentiment import SentimentResult, ScoredNewsItem
from services.watsonx_client import call_granite_json, validate_sentiment_json

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sentiment_prompt.txt"

# ── Keyword-based fallback scorer ─────────────────────────────────────────────
_BULLISH = {
    "record", "beat", "surge", "growth", "profit", "strong", "outperform",
    "upgrade", "buy", "launch", "expand", "win", "acquire", "dividend",
    "rally", "gain", "positive", "bullish", "innovation",
}
_BEARISH = {
    "miss", "decline", "loss", "risk", "warn", "downgrade", "sell",
    "slow", "fall", "drop", "concern", "probe", "lawsuit", "recall",
    "cut", "layoff", "debt", "breach", "penalty", "volatility",
}


def _keyword_sentiment(text: str) -> tuple[str, float]:
    """Return (sentiment_label, confidence) based on keyword matching."""
    words = set(re.sub(r"[^a-z ]", "", text.lower()).split())
    bull = len(words & _BULLISH)
    bear = len(words & _BEARISH)
    total = bull + bear
    if total == 0:
        return "neutral", 0.5
    if bull > bear:
        conf = min(0.5 + (bull - bear) / (2 * total), 0.90)
        return "positive", round(conf, 2)
    elif bear > bull:
        conf = min(0.5 + (bear - bull) / (2 * total), 0.90)
        return "negative", round(conf, 2)
    return "neutral", 0.5


def _rule_based_fallback(news: list[NewsItem], ticker: str) -> SentimentResult:
    """Deterministic keyword-based sentiment when Granite is unavailable."""
    scored = []
    for n in news:
        label, conf = _keyword_sentiment(f"{n.title} {n.description}")
        scored.append(ScoredNewsItem(
            headline=n.title,
            description=n.description,
            url=n.url,
            published_at=n.published_at,
            source=n.source,
            sentiment=label,
            confidence=conf,
            key_theme="keyword-based",
        ))
    pos = sum(1 for s in scored if s.sentiment == "positive")
    neu = sum(1 for s in scored if s.sentiment == "neutral")
    neg = sum(1 for s in scored if s.sentiment == "negative")
    total = len(scored)
    agg = round((pos - neg) / total, 3) if total else 0.0
    label_map = {
        "pos": "Bullish", "neg": "Bearish", "neu": "Neutral",
    }
    if agg > 0.1:
        label = "Bullish"
    elif agg < -0.1:
        label = "Bearish"
    else:
        label = "Neutral"
    return SentimentResult(
        ticker=ticker,
        items=scored,
        aggregate_score=agg,
        sentiment_label=label,
        top_themes=["keyword-based fallback"],
        total_articles=total,
        positive_count=pos,
        neutral_count=neu,
        negative_count=neg,
    )


def _load_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return "Analyze sentiment of these news items about {{TICKER}}: {{NEWS_ITEMS}}"


def _build_prompt(ticker: str, news_items: list[NewsItem]) -> str:
    template = _load_prompt()
    news_text = json.dumps([
        {"headline": n.title, "summary": n.description}
        for n in news_items
    ], indent=2)
    return (
        template
        .replace("{{TICKER}}", ticker)
        .replace("{{NEWS_ITEMS}}", news_text)
    )


def _granite_to_scored_items(
    raw_items: list[dict],
    original_news: list[NewsItem],
) -> list[ScoredNewsItem]:
    scored = []
    for i, item in enumerate(raw_items):
        orig = original_news[i] if i < len(original_news) else None
        scored.append(ScoredNewsItem(
            headline=item.get("headline", orig.title if orig else ""),
            description=orig.description if orig else "",
            url=orig.url if orig else "",
            published_at=orig.published_at if orig else "",
            source=orig.source if orig else "",
            sentiment=item.get("sentiment", "neutral"),
            confidence=float(item.get("confidence", 0.5)),
            key_theme=item.get("key_theme", ""),
        ))
    return scored


class SentimentAgent(BaseAgent):
    name = "sentiment"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        if state.raw_data is None:
            self._fail_step(state, "No raw data for sentiment analysis")
            return state

        news = state.raw_data.primary.news
        ticker = state.raw_data.primary.ticker
        self._log_input(ticker=ticker, news_count=len(news))

        if not news:
            state.sentiment = SentimentResult(
                ticker=ticker,
                aggregate_score=0.0,
                sentiment_label="Neutral",
                total_articles=0,
            )
            self._warn_step(state, "No news articles — neutral sentiment assumed")
            return state

        # Limit to 20 items to keep prompt size manageable
        batch = news[:20]
        prompt = _build_prompt(ticker, batch)

        granite_mode = "demo"
        try:
            result, granite_mode = call_granite_json(
                prompt,
                max_tokens=1500,
                validator=validate_sentiment_json,
            )
            # Propagate mode to pipeline state
            state.granite_mode = granite_mode

            items_raw = result.get("items", [])
            scored_items = _granite_to_scored_items(items_raw, batch)

            # Items beyond batch get neutral assignment
            for n in news[20:]:
                scored_items.append(ScoredNewsItem(
                    headline=n.title,
                    description=n.description,
                    url=n.url,
                    published_at=n.published_at,
                    source=n.source,
                    sentiment="neutral",
                    confidence=0.5,
                    key_theme="",
                ))

            pos = sum(1 for s in scored_items if s.sentiment == "positive")
            neu = sum(1 for s in scored_items if s.sentiment == "neutral")
            neg = sum(1 for s in scored_items if s.sentiment == "negative")
            agg_score = result.get("aggregate_score")
            if agg_score is None:
                total = len(scored_items)
                agg_score = round((pos - neg) / total, 3) if total else 0.0
            label = result.get("sentiment_label", "Neutral")
            top_themes = result.get("top_themes", [])

            state.sentiment = SentimentResult(
                ticker=ticker,
                items=scored_items,
                aggregate_score=float(agg_score),
                sentiment_label=label,
                top_themes=top_themes,
                total_articles=len(scored_items),
                positive_count=pos,
                neutral_count=neu,
                negative_count=neg,
            )

        except Exception as exc:
            # Granite unavailable — use keyword-based fallback
            self.logger.warning(
                "Granite sentiment failed (%s) — using keyword fallback", exc
            )
            state.sentiment = _rule_based_fallback(news, ticker)
            granite_mode = "fallback"
            state.granite_mode = "fallback"
            self._warn_step(
                state,
                f"Granite unavailable; keyword fallback used. {len(news)} articles scored."
            )
            self._log_output(
                ticker=ticker,
                articles=len(news),
                label=state.sentiment.sentiment_label,
                score=state.sentiment.aggregate_score,
                mode="fallback",
            )
            return state

        self._log_output(
            ticker=ticker,
            articles=len(state.sentiment.items),
            label=state.sentiment.sentiment_label,
            score=state.sentiment.aggregate_score,
            mode=granite_mode,
        )
        self._finish_step(
            state,
            f"Sentiment: {state.sentiment.sentiment_label} "
            f"(score={state.sentiment.aggregate_score:.2f}, "
            f"articles={len(state.sentiment.items)}, mode={granite_mode})"
        )
        return state

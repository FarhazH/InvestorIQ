"""
News service — fetches recent news via NewsAPI or falls back to Google News RSS.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


def fetch_news(ticker: str, company_name: str = "", limit: int = 30) -> list[dict]:
    """
    Fetch up to `limit` news items for a ticker/company.
    Tries NewsAPI first, then falls back to Google News RSS.
    Returns list of dicts: {title, description, url, published_at, source}.
    """
    if settings.demo_mode:
        from services.demo_data import get_demo_news
        return get_demo_news(ticker)

    query = company_name or ticker
    articles = []

    # --- Try NewsAPI --------------------------------------------------------
    if settings.news_api_key:
        articles = _fetch_newsapi(query, limit)
        if articles:
            return articles

    # --- Fallback: Google News RSS ------------------------------------------
    articles = _fetch_google_rss(query, limit)
    return articles


def _fetch_newsapi(query: str, limit: int) -> list[dict]:
    try:
        import requests  # type: ignore
        from_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": min(limit, 100),
            "language": "en",
            "apiKey": settings.news_api_key,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
                "source": art.get("source", {}).get("name", ""),
            })
        logger.info("NewsAPI returned %d articles for '%s'", len(articles), query)
        return articles
    except Exception as exc:
        logger.warning("NewsAPI failed for '%s': %s", query, exc)
        return []


def _fetch_google_rss(query: str, limit: int) -> list[dict]:
    try:
        import feedparser  # type: ignore
        import urllib.parse
        encoded = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "url": entry.get("link", ""),
                "published_at": entry.get("published", ""),
                "source": "Google News",
            })
        logger.info("Google News RSS returned %d articles for '%s'", len(articles), query)
        return articles
    except Exception as exc:
        logger.warning("Google News RSS failed for '%s': %s", query, exc)
        return []

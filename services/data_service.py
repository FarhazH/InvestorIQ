"""
Data service — fetches price history, financial statements, analyst data
from yfinance and SEC EDGAR.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def _safe_df_to_dict(df: Any) -> dict:
    """Convert DataFrame to JSON-serialisable dict, handling None."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return {}
    try:
        return df.to_dict()
    except Exception:
        return {}


def fetch_ticker_data(ticker: str, period: str = "2y") -> dict:
    """
    Fetch comprehensive data for a single ticker using yfinance.

    Returns a dict with keys:
      - info, price_history, income_stmt, balance_sheet, cash_flow,
        analyst_ratings, analyst_targets
    """
    if settings.demo_mode:
        from services.demo_data import get_demo_ticker_data
        return get_demo_ticker_data(ticker)

    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise RuntimeError("yfinance not installed") from exc

    logger.info("Fetching yfinance data for %s", ticker)
    t = yf.Ticker(ticker)

    # Price history
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=730)
    hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    # Financial statements (annual)
    income = t.income_stmt
    balance = t.balance_sheet
    cashflow = t.cashflow

    # Info dict
    info: dict = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    # Analyst data
    analyst_ratings: dict = {}
    analyst_targets: dict = {}
    try:
        recs = t.recommendations
        if recs is not None and not recs.empty:
            analyst_ratings = _safe_df_to_dict(recs.tail(20))
    except Exception:
        pass
    try:
        analyst_targets = {
            "targetMeanPrice": info.get("targetMeanPrice"),
            "targetHighPrice": info.get("targetHighPrice"),
            "targetLowPrice": info.get("targetLowPrice"),
            "recommendationMean": info.get("recommendationMean"),
            "recommendationKey": info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
        }
    except Exception:
        pass

    return {
        "ticker": ticker,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "info": info,
        "price_history": _safe_df_to_dict(hist),
        "income_stmt": _safe_df_to_dict(income),
        "balance_sheet": _safe_df_to_dict(balance),
        "cash_flow": _safe_df_to_dict(cashflow),
        "analyst_ratings": analyst_ratings,
        "analyst_targets": analyst_targets,
    }


def fetch_price_history_df(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Return raw price history DataFrame for charting/analysis."""
    if settings.demo_mode:
        from services.demo_data import get_demo_price_df
        return get_demo_price_df(ticker)

    import yfinance as yf  # type: ignore
    t = yf.Ticker(ticker)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=730)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def fetch_sec_filings(ticker: str, filing_type: str = "10-K", limit: int = 1) -> list[dict]:
    """
    Fetch recent SEC EDGAR filings for a ticker.
    Returns a list of filing metadata dicts.
    """
    if settings.demo_mode:
        return [{"filing_type": filing_type, "date": "2024-11-01", "url": "https://sec.gov/demo"}]

    import requests  # type: ignore

    cik_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2022-01-01&forms={filing_type}"
    headers = {"User-Agent": "InvestIQ research@investiq.ai"}
    try:
        resp = requests.get(cik_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits[:limit]:
            src = hit.get("_source", {})
            results.append({
                "filing_type": src.get("form_type"),
                "date": src.get("file_date"),
                "entity": src.get("entity_name"),
                "url": f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id')}/",
            })
        return results
    except Exception as exc:
        logger.warning("SEC EDGAR fetch failed for %s: %s", ticker, exc)
        return []

"""
Demo data — static sample data for AAPL, MSFT, GOOGL, AMZN.
Used when DEMO_MODE=true so the app runs without any external API keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


# ── Price history ─────────────────────────────────────────────────────────────

def get_demo_price_df(ticker: str) -> pd.DataFrame:
    """Generate a synthetic 2-year daily OHLCV DataFrame."""
    rng = np.random.default_rng(seed=hash(ticker) % (2**31))
    days = 504
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    base_prices = {
        "AAPL": 155.0,
        "MSFT": 290.0,
        "GOOGL": 125.0,
        "AMZN": 130.0,
    }
    start = base_prices.get(ticker.upper(), 100.0)
    returns = rng.normal(0.0003, 0.015, size=days)
    close = start * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.02, days))
    low = close * (1 - rng.uniform(0.001, 0.02, days))
    open_ = close * (1 + rng.normal(0, 0.005, days))
    volume = rng.integers(50_000_000, 150_000_000, days).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


# ── Ticker data ───────────────────────────────────────────────────────────────

_DEMO_INFOS = {
    "AAPL": {
        "shortName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 2_800_000_000_000,
        "trailingPE": 28.5,
        "forwardPE": 25.1,
        "priceToBook": 44.3,
        "returnOnEquity": 1.47,
        "returnOnAssets": 0.22,
        "debtToEquity": 173.0,
        "currentRatio": 0.99,
        "beta": 1.24,
        "dividendYield": 0.0055,
        "targetMeanPrice": 198.0,
        "targetHighPrice": 220.0,
        "targetLowPrice": 175.0,
        "recommendationMean": 1.8,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 38,
        "totalRevenue": 383_285_000_000,
        "netIncomeToCommon": 97_000_000_000,
        "ebitda": 123_000_000_000,
        "grossMargins": 0.443,
        "operatingMargins": 0.302,
        "profitMargins": 0.253,
        "freeCashflow": 99_000_000_000,
        "totalDebt": 110_000_000_000,
        "totalCash": 61_000_000_000,
        "earningsQuarterlyGrowth": 0.11,
        "revenueGrowth": 0.081,
        "trailingEps": 6.13,
        "currency": "USD",
        "sharesOutstanding": 15_550_000_000,
        "totalAssets": 352_583_000_000,
        "bookValue": 4.84,
        "currentPrice": 180.0,
    },
    "MSFT": {
        "shortName": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
        "marketCap": 3_100_000_000_000,
        "trailingPE": 36.2,
        "forwardPE": 30.5,
        "priceToBook": 13.8,
        "returnOnEquity": 0.38,
        "returnOnAssets": 0.17,
        "debtToEquity": 42.5,
        "currentRatio": 1.77,
        "beta": 0.90,
        "dividendYield": 0.007,
        "targetMeanPrice": 410.0,
        "targetHighPrice": 450.0,
        "targetLowPrice": 380.0,
        "recommendationMean": 1.6,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 44,
        "totalRevenue": 211_915_000_000,
        "netIncomeToCommon": 72_000_000_000,
        "ebitda": 103_000_000_000,
        "grossMargins": 0.694,
        "operatingMargins": 0.448,
        "profitMargins": 0.340,
        "freeCashflow": 59_000_000_000,
        "totalDebt": 79_000_000_000,
        "totalCash": 80_000_000_000,
        "earningsQuarterlyGrowth": 0.17,
        "revenueGrowth": 0.155,
        "trailingEps": 9.65,
        "currency": "USD",
        "sharesOutstanding": 7_430_000_000,
        "totalAssets": 512_163_000_000,
        "bookValue": 23.44,
        "currentPrice": 415.0,
    },
    "GOOGL": {
        "shortName": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "marketCap": 2_000_000_000_000,
        "trailingPE": 23.1,
        "forwardPE": 19.8,
        "priceToBook": 6.4,
        "returnOnEquity": 0.28,
        "returnOnAssets": 0.15,
        "debtToEquity": 10.5,
        "currentRatio": 2.10,
        "beta": 1.05,
        "dividendYield": 0.0,
        "targetMeanPrice": 180.0,
        "targetHighPrice": 210.0,
        "targetLowPrice": 160.0,
        "recommendationMean": 1.7,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 50,
        "totalRevenue": 307_394_000_000,
        "netIncomeToCommon": 73_000_000_000,
        "ebitda": 97_000_000_000,
        "grossMargins": 0.558,
        "operatingMargins": 0.282,
        "profitMargins": 0.237,
        "freeCashflow": 69_000_000_000,
        "totalDebt": 28_000_000_000,
        "totalCash": 110_000_000_000,
        "earningsQuarterlyGrowth": 0.34,
        "revenueGrowth": 0.141,
        "trailingEps": 5.80,
        "currency": "USD",
        "sharesOutstanding": 12_380_000_000,
        "totalAssets": 462_679_000_000,
        "bookValue": 20.11,
        "currentPrice": 162.0,
    },
    "AMZN": {
        "shortName": "Amazon.com Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "marketCap": 1_900_000_000_000,
        "trailingPE": 55.0,
        "forwardPE": 38.0,
        "priceToBook": 8.9,
        "returnOnEquity": 0.21,
        "returnOnAssets": 0.065,
        "debtToEquity": 61.0,
        "currentRatio": 1.05,
        "beta": 1.15,
        "dividendYield": 0.0,
        "targetMeanPrice": 210.0,
        "targetHighPrice": 240.0,
        "targetLowPrice": 185.0,
        "recommendationMean": 1.5,
        "recommendationKey": "strong_buy",
        "numberOfAnalystOpinions": 55,
        "totalRevenue": 574_785_000_000,
        "netIncomeToCommon": 30_000_000_000,
        "ebitda": 85_000_000_000,
        "grossMargins": 0.469,
        "operatingMargins": 0.067,
        "profitMargins": 0.052,
        "freeCashflow": 35_000_000_000,
        "totalDebt": 140_000_000_000,
        "totalCash": 73_000_000_000,
        "earningsQuarterlyGrowth": 2.21,
        "revenueGrowth": 0.123,
        "trailingEps": 2.90,
        "currency": "USD",
        "sharesOutstanding": 10_330_000_000,
        "totalAssets": 527_854_000_000,
        "bookValue": 19.10,
        "currentPrice": 185.0,
    },
}


def get_demo_ticker_data(ticker: str) -> dict:
    t = ticker.upper()
    info = _DEMO_INFOS.get(t, _DEMO_INFOS["AAPL"])
    # Build minimal income / balance / cashflow stubs
    years = ["2023", "2022", "2021", "2020"]
    revenue_base = info["totalRevenue"]
    income_stmt = {
        "Total Revenue": {y: int(revenue_base * (1 - 0.08 * i)) for i, y in enumerate(years)},
        "Net Income": {y: int(info["netIncomeToCommon"] * (1 - 0.1 * i)) for i, y in enumerate(years)},
        "EBITDA": {y: int(info["ebitda"] * (1 - 0.08 * i)) for i, y in enumerate(years)},
        "Gross Profit": {y: int(revenue_base * info["grossMargins"] * (1 - 0.05 * i)) for i, y in enumerate(years)},
        "Operating Income": {y: int(revenue_base * info["operatingMargins"] * (1 - 0.06 * i)) for i, y in enumerate(years)},
        "Basic EPS": {y: round(info["trailingEps"] * (1 - 0.09 * i), 2) for i, y in enumerate(years)},
        "Interest Expense": {y: int(2_000_000_000 * (1 + 0.05 * i)) for i, y in enumerate(years)},
    }
    balance_sheet = {
        "Total Assets": {y: int(info["totalDebt"] / info["debtToEquity"] * 100 * (1 + 0.03 * i)) for i, y in enumerate(years)},
        "Total Debt": {y: int(info["totalDebt"] * (1 - 0.05 * i)) for i, y in enumerate(years)},
        "Total Stockholder Equity": {y: int(info["totalDebt"] / (info["debtToEquity"] / 100) * (1 + 0.04 * i)) for i, y in enumerate(years)},
        "Cash And Cash Equivalents": {y: int(info["totalCash"] * (1 - 0.03 * i)) for i, y in enumerate(years)},
        "Current Assets": {y: int(info["totalCash"] * 1.5 * (1 - 0.02 * i)) for i, y in enumerate(years)},
        "Current Liabilities": {y: int(info["totalCash"] * 1.5 / info["currentRatio"] * (1 + 0.02 * i)) for i, y in enumerate(years)},
    }
    cashflow = {
        "Free Cash Flow": {y: int(info["freeCashflow"] * (1 - 0.07 * i)) for i, y in enumerate(years)},
        "Operating Cash Flow": {y: int(info["freeCashflow"] * 1.3 * (1 - 0.06 * i)) for i, y in enumerate(years)},
        "Capital Expenditure": {y: int(-info["freeCashflow"] * 0.3 * (1 + 0.05 * i)) for i, y in enumerate(years)},
    }
    price_df = get_demo_price_df(t)
    return {
        "ticker": t,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "info": info,
        "price_history": price_df.to_dict(),
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cashflow,
        "analyst_ratings": {
            "strongBuy": 18, "buy": 12, "hold": 7, "sell": 1, "strongSell": 0,
        },
        "analyst_targets": {
            "targetMeanPrice": info.get("targetMeanPrice"),
            "targetHighPrice": info.get("targetHighPrice"),
            "targetLowPrice": info.get("targetLowPrice"),
            "recommendationMean": info.get("recommendationMean"),
            "recommendationKey": info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
        },
    }


def get_demo_news(ticker: str) -> list[dict]:
    articles = {
        "AAPL": [
            {"title": "Apple reports record Q4 revenue driven by services", "description": "Apple Inc. posted record quarterly revenue, with services division growing 20% YoY.", "url": "https://example.com/apple-q4", "published_at": "2024-11-01", "source": "Reuters"},
            {"title": "iPhone 16 sales exceed analyst expectations globally", "description": "iPhone 16 demand surpassed Wall Street estimates in the first week of availability.", "url": "https://example.com/iphone16", "published_at": "2024-10-28", "source": "Bloomberg"},
            {"title": "Apple expands AI features with on-device processing", "description": "Apple Intelligence rollout strengthens competitive moat in AI-integrated hardware.", "url": "https://example.com/apple-ai", "published_at": "2024-10-20", "source": "CNBC"},
            {"title": "China slowdown raises concerns for Apple market share", "description": "Huawei's resurgence may pressure Apple iPhone sales in mainland China.", "url": "https://example.com/apple-china", "published_at": "2024-10-15", "source": "WSJ"},
            {"title": "Apple authorized $110B share buyback program", "description": "Board approves largest-ever buyback, signaling confidence in cash generation.", "url": "https://example.com/apple-buyback", "published_at": "2024-10-10", "source": "FT"},
        ],
        "MSFT": [
            {"title": "Microsoft Azure cloud revenue jumps 29% YoY", "description": "Azure growth accelerates driven by AI workloads and Copilot enterprise adoption.", "url": "https://example.com/msft-azure", "published_at": "2024-11-01", "source": "Reuters"},
            {"title": "Microsoft Copilot enterprise adoption hits 1M seats", "description": "Generative AI product adoption is driving incremental revenue across Office 365.", "url": "https://example.com/msft-copilot", "published_at": "2024-10-25", "source": "Bloomberg"},
            {"title": "DOJ antitrust probe into Microsoft cloud practices", "description": "Regulators examine bundling practices in Microsoft's cloud and software offerings.", "url": "https://example.com/msft-doj", "published_at": "2024-10-18", "source": "WSJ"},
        ],
        "GOOGL": [
            {"title": "Alphabet Search revenue grows 12% despite AI competition", "description": "Google Search maintains dominance as AI Overviews drive engagement metrics.", "url": "https://example.com/googl-search", "published_at": "2024-11-01", "source": "Reuters"},
            {"title": "Google Cloud reaches $11B quarterly revenue milestone", "description": "GCP growth rate exceeds Azure and AWS on a percentage basis in latest quarter.", "url": "https://example.com/googl-cloud", "published_at": "2024-10-29", "source": "Bloomberg"},
        ],
        "AMZN": [
            {"title": "AWS profit surges as AI infrastructure demand soars", "description": "Amazon Web Services segment profits doubled YoY as AI training workloads scale.", "url": "https://example.com/amzn-aws", "published_at": "2024-11-01", "source": "Reuters"},
            {"title": "Amazon Prime membership reaches 220 million globally", "description": "Prime expansion into emerging markets drives advertising and logistics synergies.", "url": "https://example.com/amzn-prime", "published_at": "2024-10-27", "source": "Bloomberg"},
            {"title": "Amazon faces labor pressure ahead of holiday season", "description": "Warehouse workers unionization efforts may increase operational costs.", "url": "https://example.com/amzn-labor", "published_at": "2024-10-20", "source": "CNBC"},
        ],
    }
    return articles.get(ticker.upper(), articles["AAPL"])

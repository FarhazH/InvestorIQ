"""
Agent 1 — Data Gathering Agent.
Fetches price history, financial statements, news, and analyst data for
the primary ticker and up to 3 competitors. Caches results in IBM COS.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.market_data import RawMarketData, TickerData, NewsItem
from services.data_service import fetch_ticker_data, fetch_sec_filings
from services.news_service import fetch_news
from services.storage_service import save_to_cos, load_from_cos


def _cos_key(ticker: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"raw/{ticker.upper()}_{today}.json"


class DataGatheringAgent(BaseAgent):
    name = "data_gathering"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        try:
            primary_data = self._with_retry(self._fetch_one, state.ticker)
            competitor_data = []
            for comp in state.competitors[:3]:
                try:
                    cd = self._with_retry(self._fetch_one, comp)
                    competitor_data.append(cd)
                except Exception as exc:
                    self.logger.warning("Skipping competitor %s: %s", comp, exc)

            state.raw_data = RawMarketData(
                primary=primary_data,
                competitors=competitor_data,
            )

            # Cache to IBM COS
            key = _cos_key(state.ticker)
            payload = state.raw_data.model_dump()
            if save_to_cos(key, payload):
                state.raw_data.cached = True
                self.logger.info("Raw data cached to COS: %s", key)

            self._finish_step(state, f"Fetched data for {state.ticker} + {len(competitor_data)} peers")
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

    def _fetch_one(self, ticker: str) -> TickerData:
        raw = fetch_ticker_data(ticker)
        # Fetch news separately
        company_name = raw.get("info", {}).get("shortName", ticker)
        news_raw = fetch_news(ticker, company_name=company_name, limit=30)
        news_items = [
            NewsItem(
                title=n.get("title", ""),
                description=n.get("description", ""),
                url=n.get("url", ""),
                published_at=n.get("published_at", ""),
                source=n.get("source", ""),
            )
            for n in news_raw
        ]
        sec = fetch_sec_filings(ticker)
        return TickerData(
            ticker=ticker.upper(),
            fetched_at=raw.get("fetched_at", datetime.now(timezone.utc).isoformat()),
            info=raw.get("info", {}),
            price_history=raw.get("price_history", {}),
            income_stmt=raw.get("income_stmt", {}),
            balance_sheet=raw.get("balance_sheet", {}),
            cash_flow=raw.get("cash_flow", {}),
            analyst_ratings=raw.get("analyst_ratings", {}),
            analyst_targets=raw.get("analyst_targets", {}),
            news=news_items,
            sec_filings=sec,
        )

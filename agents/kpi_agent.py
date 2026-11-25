"""
Agent 3 — KPI Extraction Agent.
Extracts revenue, net income, EBITDA, EPS, FCF, margins, debt,
and YoY/QoQ growth rates from raw financial statements.
"""
from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.market_data import TickerData
from models.kpi import KPIResult, TickerKPIs, YoYMetric


def _safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _first_val(d: dict, *keys: str) -> float | None:
    """Return the first non-None value from a nested dict for the first available year."""
    for k in keys:
        sub = d.get(k)
        if isinstance(sub, dict) and sub:
            # Return first (most recent) value
            vals = [v for v in sub.values() if v is not None]
            if vals:
                f = _safe_float(vals[0])
                if f is not None:
                    return f
    return None


def _yoy_pair(d: dict, key: str) -> tuple[float | None, float | None]:
    """Return (current_year, prior_year) for a statement line item."""
    sub = d.get(key, {})
    if not isinstance(sub, dict):
        return None, None
    vals = [(str(k), v) for k, v in sub.items() if v is not None]
    if not vals:
        return None, None
    vals.sort(key=lambda x: x[0], reverse=True)
    current = _safe_float(vals[0][1]) if len(vals) >= 1 else None
    prior = _safe_float(vals[1][1]) if len(vals) >= 2 else None
    return current, prior


def _yoy_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def _extract_kpis(td: TickerData) -> TickerKPIs:
    info = td.info or {}
    income = td.income_stmt or {}
    balance = td.balance_sheet or {}
    cashflow = td.cash_flow or {}

    # Revenue
    rev_cur, rev_prior = _yoy_pair(income, "Total Revenue")
    rev_yoy = _yoy_change(rev_cur, rev_prior)
    revenue = YoYMetric(current=rev_cur, prior_year=rev_prior, yoy_change_pct=rev_yoy)

    # Net Income
    ni_cur, ni_prior = _yoy_pair(income, "Net Income")
    ni_yoy = _yoy_change(ni_cur, ni_prior)
    net_income = YoYMetric(current=ni_cur, prior_year=ni_prior, yoy_change_pct=ni_yoy)

    # EBITDA (from info or statement)
    ebitda = _safe_float(info.get("ebitda")) or _first_val(income, "EBITDA", "Ebitda")

    # EPS TTM
    eps_ttm = _safe_float(info.get("trailingEps"))
    eps_prior = _safe_float(info.get("trailingEps"))  # yfinance has limited historical EPS
    # EPS YoY from earnings growth
    eps_yoy = _safe_float(info.get("earningsQuarterlyGrowth"))
    if eps_yoy is not None:
        eps_yoy = round(eps_yoy * 100, 2)

    # Free Cash Flow
    fcf_cur, fcf_prior = _yoy_pair(cashflow, "Free Cash Flow")
    if fcf_cur is None:
        fcf_cur = _safe_float(info.get("freeCashflow"))
    fcf_yoy = _yoy_change(fcf_cur, fcf_prior)

    # Margins (prefer info over calculated)
    gross_margin = _safe_float(info.get("grossMargins"))
    operating_margin = _safe_float(info.get("operatingMargins"))
    net_margin = _safe_float(info.get("profitMargins"))

    # Calculate from statements if info missing
    if gross_margin is None and rev_cur:
        gross_profit = _first_val(income, "Gross Profit")
        if gross_profit and rev_cur:
            gross_margin = gross_profit / rev_cur

    if operating_margin is None and rev_cur:
        op_income = _first_val(income, "Operating Income", "EBIT")
        if op_income and rev_cur:
            operating_margin = op_income / rev_cur

    if net_margin is None and rev_cur and ni_cur:
        net_margin = ni_cur / rev_cur

    # Debt
    total_debt = _safe_float(info.get("totalDebt")) or _first_val(balance, "Total Debt", "Long Term Debt")
    total_cash = _safe_float(info.get("totalCash")) or _first_val(balance, "Cash And Cash Equivalents", "Cash")
    net_debt = None
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash

    return TickerKPIs(
        ticker=td.ticker,
        currency=info.get("currency", "USD"),
        revenue=revenue,
        net_income=net_income,
        ebitda=ebitda,
        eps_ttm=eps_ttm,
        eps_yoy_change_pct=eps_yoy,
        free_cash_flow=fcf_cur,
        fcf_yoy_change_pct=fcf_yoy,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        total_debt=total_debt,
        total_cash=total_cash,
        net_debt=net_debt,
        revenue_growth_yoy=rev_yoy if rev_yoy is not None else (
            _safe_float(info.get("revenueGrowth")) * 100
            if _safe_float(info.get("revenueGrowth")) else None
        ),
        earnings_growth_yoy=eps_yoy,
    )


class KPIAgent(BaseAgent):
    name = "kpi_extraction"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        if state.raw_data is None:
            self._fail_step(state, "No raw data for KPI extraction")
            return state

        try:
            primary_kpis = _extract_kpis(state.raw_data.primary)
            competitor_kpis = [_extract_kpis(c) for c in state.raw_data.competitors]
            state.kpis = KPIResult(primary=primary_kpis, competitors=competitor_kpis)
            self._finish_step(state, f"Revenue: {primary_kpis.revenue.current}, NetMargin: {primary_kpis.net_margin}")
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

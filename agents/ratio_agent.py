"""
Agent 4 — Ratio Analysis Agent.
Computes valuation, profitability, leverage, liquidity, and technical ratios
including RSI, moving averages, beta, volatility, max drawdown, and Sharpe.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.market_data import TickerData
from models.ratios import RatioResult, TickerRatios


def _safe(val: Any) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _price_series(td: TickerData) -> pd.Series:
    ph = td.price_history or {}
    close = ph.get("Close", {})
    if not close:
        return pd.Series(dtype=float)
    s = pd.Series(close, dtype=float).dropna()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()
    return s


def _rsi(prices: pd.Series, period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    # When loss == 0, RSI is 100 (all gains, no losses)
    last_loss = float(loss.iloc[-1])
    last_gain = float(gain.iloc[-1])
    if math.isnan(last_gain):
        return None
    if last_loss == 0.0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    val = 100 - (100 / (1 + rs))
    return round(val, 2) if not math.isnan(val) else None


def _max_drawdown(prices: pd.Series) -> float | None:
    if prices.empty:
        return None
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return round(float(drawdown.min()) * 100, 2)


def _annualised_volatility(prices: pd.Series) -> float | None:
    if len(prices) < 20:
        return None
    returns = prices.pct_change().dropna()
    vol = returns.std() * math.sqrt(252)
    return round(float(vol) * 100, 2)


def _sharpe(prices: pd.Series, risk_free: float = 0.045) -> float | None:
    if len(prices) < 20:
        return None
    returns = prices.pct_change().dropna()
    ann_ret = float(returns.mean()) * 252
    ann_vol = float(returns.std()) * math.sqrt(252)
    if ann_vol == 0:
        return None
    return round((ann_ret - risk_free) / ann_vol, 3)


def _first_stmt_val(d: dict, *keys: str) -> float | None:
    for k in keys:
        sub = d.get(k)
        if isinstance(sub, dict) and sub:
            vals = [v for v in sub.values() if v is not None]
            if vals:
                f = _safe(vals[0])
                if f is not None:
                    return f
    return None


def _compute_ratios(td: TickerData) -> TickerRatios:
    info = td.info or {}
    income = td.income_stmt or {}
    balance = td.balance_sheet or {}

    prices = _price_series(td)
    current_price = float(prices.iloc[-1]) if not prices.empty else None

    # ── Valuation ────────────────────────────────────────────────────────────
    pe = _safe(info.get("trailingPE"))
    fpe = _safe(info.get("forwardPE"))
    pb = _safe(info.get("priceToBook"))

    # P/S ratio
    mkt_cap = _safe(info.get("marketCap"))
    revenue = _first_stmt_val(income, "Total Revenue") or _safe(info.get("totalRevenue"))
    ps = round(mkt_cap / revenue, 2) if mkt_cap and revenue else None

    # ── Profitability ────────────────────────────────────────────────────────
    roe = _safe(info.get("returnOnEquity"))
    roa = _safe(info.get("returnOnAssets"))
    gross_m = _safe(info.get("grossMargins"))
    op_m = _safe(info.get("operatingMargins"))
    net_m = _safe(info.get("profitMargins"))

    # ── Leverage ─────────────────────────────────────────────────────────────
    de = _safe(info.get("debtToEquity"))
    if de is not None:
        de = round(de / 100, 3)  # yfinance returns as percentage points

    # Interest coverage = EBIT / Interest Expense
    ebit = _first_stmt_val(income, "Operating Income", "EBIT")
    interest_exp = _first_stmt_val(income, "Interest Expense")
    interest_coverage = None
    if ebit and interest_exp and interest_exp != 0:
        # Interest expense can be negative in yfinance
        interest_coverage = round(ebit / abs(interest_exp), 2)

    # ── Liquidity ────────────────────────────────────────────────────────────
    current_ratio = _safe(info.get("currentRatio"))
    if current_ratio is None:
        curr_assets = _first_stmt_val(balance, "Current Assets")
        curr_liab = _first_stmt_val(balance, "Current Liabilities")
        if curr_assets and curr_liab and curr_liab != 0:
            current_ratio = round(curr_assets / curr_liab, 3)

    # ── Technical ────────────────────────────────────────────────────────────
    beta = _safe(info.get("beta"))
    rsi_14 = _rsi(prices) if not prices.empty else None
    ma50 = round(float(prices.tail(50).mean()), 2) if len(prices) >= 50 else None
    ma200 = round(float(prices.tail(200).mean()), 2) if len(prices) >= 200 else None

    pma50 = round((current_price / ma50 - 1) * 100, 2) if current_price and ma50 else None
    pma200 = round((current_price / ma200 - 1) * 100, 2) if current_price and ma200 else None

    vol = _annualised_volatility(prices)
    mdd = _max_drawdown(prices)
    sharpe = _sharpe(prices)

    div_yield = _safe(info.get("dividendYield"))

    return TickerRatios(
        ticker=td.ticker,
        pe_ratio=pe,
        forward_pe=fpe,
        pb_ratio=pb,
        ps_ratio=ps,
        roe=roe,
        roa=roa,
        gross_margin=gross_m,
        operating_margin=op_m,
        net_margin=net_m,
        debt_to_equity=de,
        interest_coverage=interest_coverage,
        current_ratio=current_ratio,
        beta=beta,
        rsi_14=rsi_14,
        ma_50=ma50,
        ma_200=ma200,
        price_vs_ma50_pct=pma50,
        price_vs_ma200_pct=pma200,
        volatility_annualised=vol,
        max_drawdown=mdd,
        sharpe_ratio=sharpe,
        dividend_yield=div_yield,
    )


class RatioAgent(BaseAgent):
    name = "ratio_analysis"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        if state.raw_data is None:
            self._fail_step(state, "No raw data for ratio analysis")
            return state

        try:
            primary_ratios = _compute_ratios(state.raw_data.primary)
            competitor_ratios = [_compute_ratios(c) for c in state.raw_data.competitors]
            state.ratios = RatioResult(primary=primary_ratios, competitors=competitor_ratios)
            self._finish_step(
                state,
                f"P/E={primary_ratios.pe_ratio}, ROE={primary_ratios.roe}, RSI={primary_ratios.rsi_14}"
            )
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

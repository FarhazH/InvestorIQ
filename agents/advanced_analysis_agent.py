"""
Advanced Analysis Agent (Phase 2).

Runs between ratio_analysis and benchmarking in the pipeline and computes:
1. DCF-lite fair value range (5-year FCF projection + terminal value)
2. Altman Z-score bankruptcy risk
3. Peer-relative valuation percentiles
4. Weighted composite scoring model (fundamentals, valuation, momentum,
   sentiment, risk) with adjustable weights
5. Simple 2-year backtest: how would the weighted signal have performed
   vs buy-and-hold?
6. Forecast quality (MAPE on hold-out) for price and revenue ARIMA models
"""
from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.advanced_analysis import (
    AdvancedAnalysisResult,
    DCFResult,
    AltmanZResult,
    ValuationPercentile,
    WeightedScore,
    BacktestResult,
    BacktestTrade,
    ForecastQuality,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DCF-LITE
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_dcf(state: PipelineState) -> DCFResult | None:
    """
    5-year FCF projection using the current FCF + revenue growth rate,
    discounted at WACC, plus a terminal value at g=3%.

    WACC estimate: risk-free=4.5%, equity_risk_premium=5.5%, beta from ratios.
    """
    if not state.kpis or not state.ratios:
        return None

    kpis = state.kpis.primary
    ratios = state.ratios.primary
    info = (state.raw_data.primary.info or {}) if state.raw_data else {}

    fcf = kpis.free_cash_flow
    if not fcf or fcf <= 0:
        return None

    # Growth rate: blend of revenue growth and analyst estimates
    growth = (kpis.revenue_growth_yoy or 5.0) / 100
    growth = min(max(growth, 0.0), 0.30)  # cap at 30%

    # WACC = Rf + β × ERP  (simplified, unlevered)
    beta = ratios.beta or 1.0
    rf = 0.045       # risk-free (10y UST)
    erp = 0.055      # equity risk premium
    wacc = rf + beta * erp
    wacc = round(min(max(wacc, 0.06), 0.20), 4)

    terminal_g = 0.03  # 3% perpetuity growth

    # 5-year projected FCF
    projected = []
    cf = fcf
    for year in range(1, 6):
        # Growth rate decays linearly to terminal_g over 5 years
        yr_growth = growth - (growth - terminal_g) * (year / 5)
        cf = cf * (1 + yr_growth)
        projected.append(cf)

    # Discount each year's FCF
    pv_fcfs = [cf_i / ((1 + wacc) ** i) for i, cf_i in enumerate(projected, start=1)]
    pv_sum = sum(pv_fcfs)

    # Terminal value (Gordon Growth) at end of year 5
    tv = projected[-1] * (1 + terminal_g) / (wacc - terminal_g)
    pv_tv = tv / ((1 + wacc) ** 5)

    # Enterprise value
    ev = pv_sum + pv_tv
    net_debt = kpis.net_debt or 0.0
    equity_value = ev - net_debt

    # Shares outstanding
    shares = _safe_float(info.get("sharesOutstanding")) or _safe_float(info.get("impliedSharesOutstanding"))
    if not shares:
        mkt_cap = _safe_float(info.get("marketCap"))
        current_price_info = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if mkt_cap and current_price_info:
            shares = mkt_cap / current_price_info
    if not shares or shares <= 0:
        return None

    fair_mid = equity_value / shares

    # Bear/Bull cases: shift WACC ±1%
    def _fair(wacc_adj: float) -> float:
        pv_b = sum(
            projected[i] / ((1 + wacc_adj) ** (i + 1))
            for i in range(5)
        )
        tv_b = projected[-1] * (1 + terminal_g) / (wacc_adj - terminal_g)
        pv_tv_b = tv_b / ((1 + wacc_adj) ** 5)
        ev_b = pv_b + pv_tv_b - net_debt
        return ev_b / shares

    fair_low = _fair(wacc + 0.01)
    fair_high = _fair(max(wacc - 0.01, terminal_g + 0.001))

    # Current price from price history
    ph = (state.raw_data.primary.price_history or {}) if state.raw_data else {}
    close_data = ph.get("Close", {})
    current_price = None
    if close_data:
        try:
            current_price = float(list(close_data.values())[-1])
        except Exception:
            pass

    upside_pct = None
    if current_price and current_price > 0:
        upside_pct = round((fair_mid / current_price - 1) * 100, 1)

    return DCFResult(
        ticker=state.ticker,
        wacc=round(wacc, 4),
        terminal_growth_rate=terminal_g,
        projected_fcf=projected,
        terminal_value=round(tv, 0),
        enterprise_value=round(ev, 0),
        equity_value=round(equity_value, 0),
        shares_outstanding=shares,
        fair_value_low=round(max(fair_low, 0), 2),
        fair_value_mid=round(max(fair_mid, 0), 2),
        fair_value_high=round(max(fair_high, 0), 2),
        current_price=round(current_price, 2) if current_price else None,
        upside_pct=upside_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ALTMAN Z-SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _compute_altman_z(state: PipelineState) -> AltmanZResult | None:
    """
    Altman Z-score for public companies:
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets
    """
    if not state.kpis or not state.raw_data:
        return None

    kpis = state.kpis.primary
    info = state.raw_data.primary.info or {}
    balance = state.raw_data.primary.balance_sheet or {}

    def _first(d: dict, *keys: str) -> float | None:
        for k in keys:
            sub = d.get(k)
            if isinstance(sub, dict) and sub:
                vals = [v for v in sub.values() if v is not None]
                if vals:
                    f = _safe_float(vals[0])
                    if f is not None:
                        return f
        return None

    total_assets = _first(balance, "Total Assets") or _safe_float(info.get("totalAssets"))
    current_assets = _first(balance, "Current Assets")
    current_liabilities = _first(balance, "Current Liabilities")
    retained_earnings = _first(balance, "Retained Earnings", "Retained Earnings (Deficit)")
    total_liabilities = _first(balance, "Total Liabilities Net Minority Interest", "Total Liabilities")

    # Fallbacks from info
    if total_liabilities is None:
        total_debt = kpis.total_debt or 0
        equity = _safe_float(info.get("bookValue"))
        shares = _safe_float(info.get("sharesOutstanding"))
        if equity and shares:
            book_equity = equity * shares
            total_liabilities = (total_assets or 0) - book_equity if total_assets else None

    if not total_assets or total_assets <= 0:
        return None

    # X1: Working Capital / Total Assets
    wc = None
    if current_assets is not None and current_liabilities is not None:
        wc = current_assets - current_liabilities
    if wc is None:
        # Approximate from currentRatio
        cr = _safe_float(info.get("currentRatio"))
        cash = kpis.total_cash or 0
        if cr and cash:
            wc = cash * (cr - 1) / cr if cr != 0 else 0
    x1 = (wc / total_assets) if wc is not None else 0.0

    # X2: Retained Earnings / Total Assets
    x2 = (retained_earnings / total_assets) if retained_earnings is not None else 0.1

    # X3: EBIT / Total Assets
    ebit = None
    if kpis.ebitda:
        # Approximate EBIT as EBITDA × 0.85
        ebit = kpis.ebitda * 0.85
    x3 = (ebit / total_assets) if ebit is not None else (
        (kpis.operating_margin or 0) * (kpis.revenue.current or 0) / total_assets
    )

    # X4: Market Cap / Total Liabilities
    mkt_cap = _safe_float(info.get("marketCap")) or 0
    x4 = (mkt_cap / total_liabilities) if total_liabilities and total_liabilities > 0 else 1.0

    # X5: Revenue / Total Assets
    rev = kpis.revenue.current or 0
    x5 = rev / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    z = round(z, 3)

    if z > 2.99:
        zone = "Safe"
        interpretation = f"Z={z:.2f} — Company appears financially healthy (Z > 2.99)."
    elif z >= 1.81:
        zone = "Grey"
        interpretation = f"Z={z:.2f} — Grey zone; moderate financial stress (1.81–2.99)."
    else:
        zone = "Distress"
        interpretation = f"Z={z:.2f} — Distress zone; elevated bankruptcy risk (Z < 1.81)."

    return AltmanZResult(
        ticker=state.ticker,
        z_score=z,
        zone=zone,
        components={"X1": round(x1, 4), "X2": round(x2, 4),
                    "X3": round(x3, 4), "X4": round(x4, 4), "X5": round(x5, 4)},
        interpretation=interpretation,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PEER-RELATIVE VALUATION PERCENTILES
# ═══════════════════════════════════════════════════════════════════════════════

_VALUATION_METRICS = [
    ("pe_ratio", "P/E Ratio", True),          # lower = cheaper
    ("forward_pe", "Forward P/E", True),
    ("pb_ratio", "P/B Ratio", True),
    ("ps_ratio", "P/S Ratio", True),
    ("ev_to_ebitda", "EV/EBITDA", True),      # computed below if missing
]


def _compute_percentiles(state: PipelineState) -> list[ValuationPercentile]:
    if not state.ratios:
        return []

    all_ratios = [state.ratios.primary] + state.ratios.competitors
    results: list[ValuationPercentile] = []

    for attr, _label, _lower_is_better in _VALUATION_METRICS:
        peer_values: dict[str, float | None] = {}
        for r in all_ratios:
            val = getattr(r, attr, None)
            peer_values[r.ticker] = val

        ticker_val = peer_values.get(state.ticker)
        numeric_vals = [(t, v) for t, v in peer_values.items() if v is not None]

        if len(numeric_vals) < 2:
            percentile = None
            rank = None
        else:
            sorted_vals = sorted(numeric_vals, key=lambda x: x[1])
            pos = next((i for i, (t, _) in enumerate(sorted_vals) if t == state.ticker), None)
            if pos is not None:
                percentile = round(pos / (len(sorted_vals) - 1) * 100, 1)
                rank = pos + 1  # 1 = cheapest
            else:
                percentile = None
                rank = None

        results.append(ValuationPercentile(
            metric=attr,
            ticker_value=ticker_val,
            peer_values=peer_values,
            percentile=percentile,
            rank=rank,
            n_peers=len(all_ratios),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WEIGHTED COMPOSITE SCORING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _score_fundamentals(state: PipelineState) -> float:
    """Score 0–100 based on KPIs: margins, growth, FCF."""
    if not state.kpis:
        return 50.0
    k = state.kpis.primary
    score = 0.0
    # Net margin (0-30 pts)
    nm = (k.net_margin or 0) * 100
    score += min(nm / 0.35 * 30, 30)
    # Revenue growth (0-25 pts)
    rg = k.revenue_growth_yoy or 0
    score += min(max(rg / 20 * 25, 0), 25)
    # FCF yield proxy (0-25 pts): positive FCF = 15, strong FCF margin = bonus
    if k.free_cash_flow and k.free_cash_flow > 0:
        score += 15
        if k.revenue.current and k.revenue.current > 0:
            fcf_margin = k.free_cash_flow / k.revenue.current
            score += min(fcf_margin / 0.20 * 10, 10)
    # EPS growth (0-20 pts)
    eg = k.earnings_growth_yoy or 0
    score += min(max(eg / 20 * 20, 0), 20)
    return round(min(score, 100), 1)


def _score_valuation(state: PipelineState) -> float:
    """Score 0–100: lower P/E and P/B = higher score."""
    if not state.ratios:
        return 50.0
    r = state.ratios.primary
    score = 50.0
    if r.pe_ratio:
        # PE < 15 = max pts, PE > 40 = 0 pts
        score += (40 - min(r.pe_ratio, 40)) / 40 * 25
    if r.pb_ratio:
        score += (10 - min(r.pb_ratio, 10)) / 10 * 15
    if r.forward_pe:
        score += (35 - min(r.forward_pe, 35)) / 35 * 10
    return round(min(max(score, 0), 100), 1)


def _score_momentum(state: PipelineState) -> float:
    """Score 0–100 based on price momentum signals."""
    if not state.ratios:
        return 50.0
    r = state.ratios.primary
    score = 50.0
    if r.price_vs_ma200_pct is not None:
        score += min(max(r.price_vs_ma200_pct / 20 * 20, -20), 20)
    if r.price_vs_ma50_pct is not None:
        score += min(max(r.price_vs_ma50_pct / 10 * 10, -10), 10)
    if r.rsi_14 is not None:
        # RSI near 50-60 = good momentum
        rsi_score = 10 - abs(r.rsi_14 - 55) / 55 * 10
        score += max(rsi_score, 0)
    if r.sharpe_ratio is not None:
        score += min(max(r.sharpe_ratio * 10, -10), 10)
    return round(min(max(score, 0), 100), 1)


def _score_sentiment(state: PipelineState) -> float:
    """Score 0–100 from sentiment aggregate score (-1 to +1)."""
    if not state.sentiment:
        return 50.0
    agg = state.sentiment.aggregate_score  # -1 to +1
    return round((agg + 1) / 2 * 100, 1)


def _score_risk(state: PipelineState) -> float:
    """Score 0–100: lower risk = higher score."""
    if not state.risk:
        return 50.0
    return round(100 - state.risk.overall_risk_score, 1)


def _compute_weighted_score(
    state: PipelineState,
    weights: dict[str, float] | None = None,
) -> WeightedScore:
    default_weights = {
        "fundamentals": 0.30,
        "valuation":    0.20,
        "momentum":     0.20,
        "sentiment":    0.15,
        "risk":         0.15,
    }
    w = {**default_weights, **(weights or {})}
    # Normalise in case weights don't sum to 1
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}

    f_score = _score_fundamentals(state)
    v_score = _score_valuation(state)
    m_score = _score_momentum(state)
    s_score = _score_sentiment(state)
    r_score = _score_risk(state)

    composite = (
        f_score * w.get("fundamentals", 0.3)
        + v_score * w.get("valuation", 0.2)
        + m_score * w.get("momentum", 0.2)
        + s_score * w.get("sentiment", 0.15)
        + r_score * w.get("risk", 0.15)
    )
    composite = round(composite, 1)

    if composite >= 65:
        signal, confidence = "Overweight", round(0.5 + (composite - 65) / 70, 2)
    elif composite >= 45:
        signal, confidence = "Hold", round(0.4 + (composite - 45) / 100, 2)
    else:
        signal, confidence = "Underweight", round(0.5 + (45 - composite) / 90, 2)

    return WeightedScore(
        ticker=state.ticker,
        fundamentals_score=f_score,
        valuation_score=v_score,
        momentum_score=m_score,
        sentiment_score=s_score,
        risk_score=r_score,
        weights=w,
        composite_score=composite,
        signal=signal,
        signal_confidence=round(min(confidence, 1.0), 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_backtest(state: PipelineState) -> BacktestResult | None:
    """
    Simple rule-based backtest on 2-year price history.
    Signal logic mirrors the weighted scoring model, using only
    price-based features available at each point in time:
      - MA50 > MA200 → Overweight (buy)
      - MA50 < MA200 → Underweight (sell/hold cash)
    Rebalanced monthly. Buy-and-hold baseline.
    """
    if not state.raw_data:
        return None

    ph = state.raw_data.primary.price_history or {}
    close_data = ph.get("Close", {})
    if not close_data or len(close_data) < 60:
        return None

    try:
        prices = pd.Series(close_data, dtype=float).dropna()
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index()

        if len(prices) < 200:
            return None

        # Monthly resampled signals
        monthly = prices.resample("ME").last().dropna()
        if len(monthly) < 6:
            return None

        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()

        strategy_equity = [100.0]
        buyhold_equity = [100.0]
        in_market = False
        trades: list[BacktestTrade] = []
        entry_price: float | None = None
        entry_date: str | None = None
        dates_out: list[str] = []

        for i in range(1, len(monthly)):
            dt = monthly.index[i]
            px = float(monthly.iloc[i])
            px_prev = float(monthly.iloc[i - 1])

            # Signal at this date
            try:
                ma50_val = float(ma50.loc[:dt].iloc[-1])
                ma200_val = float(ma200.loc[:dt].iloc[-1])
                bullish = ma50_val > ma200_val
            except Exception:
                bullish = False

            ret = (px / px_prev - 1)
            bh_ret = ret

            # Strategy return: only participate when bullish
            str_ret = ret if in_market else 0.0

            if bullish and not in_market:
                in_market = True
                entry_price = px
                entry_date = dt.strftime("%Y-%m-%d")
            elif not bullish and in_market:
                in_market = False
                if entry_price and entry_price > 0:
                    trades.append(BacktestTrade(
                        date=entry_date or "",
                        signal="Overweight",
                        entry_price=entry_price,
                        exit_price=px,
                        pnl_pct=round((px / entry_price - 1) * 100, 2),
                    ))
                entry_price = None

            strategy_equity.append(round(strategy_equity[-1] * (1 + str_ret), 2))
            buyhold_equity.append(round(buyhold_equity[-1] * (1 + bh_ret), 2))
            dates_out.append(dt.strftime("%Y-%m-%d"))

        # Close any open trade
        if in_market and entry_price and entry_price > 0:
            px_last = float(monthly.iloc[-1])
            trades.append(BacktestTrade(
                date=entry_date or "",
                signal="Overweight",
                entry_price=entry_price,
                exit_price=px_last,
                pnl_pct=round((px_last / entry_price - 1) * 100, 2),
            ))

        strat_ret = round((strategy_equity[-1] / 100 - 1) * 100, 2)
        bh_ret_tot = round((buyhold_equity[-1] / 100 - 1) * 100, 2)
        excess = round(strat_ret - bh_ret_tot, 2)
        wins = sum(1 for t in trades if (t.pnl_pct or 0) > 0)
        win_rate = round(wins / len(trades), 3) if trades else 0.0

        # Max drawdown on strategy
        eq = pd.Series(strategy_equity)
        mdd = round(float((eq / eq.cummax() - 1).min()) * 100, 2)

        # Insert start date
        dates_out = [monthly.index[0].strftime("%Y-%m-%d")] + dates_out

        return BacktestResult(
            ticker=state.ticker,
            start_date=prices.index[0].strftime("%Y-%m-%d"),
            end_date=prices.index[-1].strftime("%Y-%m-%d"),
            strategy_return_pct=strat_ret,
            buyhold_return_pct=bh_ret_tot,
            excess_return_pct=excess,
            n_trades=len(trades),
            win_rate=win_rate,
            max_drawdown_pct=mdd,
            trades=trades[:50],
            dates=dates_out,
            strategy_equity=strategy_equity,
            buyhold_equity=buyhold_equity,
            note="Signal: MA50 > MA200 → Overweight. Monthly rebalancing.",
        )
    except Exception as exc:
        logger.warning("Backtest failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FORECAST QUALITY (MAPE on hold-out)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_price_forecast_mape(state: PipelineState) -> ForecastQuality | None:
    """Hold out last 8 weekly prices, fit ARIMA on the rest, compute MAPE."""
    if not state.raw_data:
        return None
    ph = state.raw_data.primary.price_history or {}
    close_data = ph.get("Close", {})
    if not close_data:
        return None
    try:
        prices = pd.Series(close_data, dtype=float).dropna()
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index()
        weekly = prices.resample("W").last().dropna()
        series = weekly.values.tolist()
        if len(series) < 20:
            return ForecastQuality(ticker=state.ticker, forecast_type="price", method="N/A")

        holdout = 8
        train = series[:-holdout]
        test = series[-holdout:]

        try:
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train, order=(1, 1, 1)).fit()
                pred = model.forecast(steps=holdout)
            method = "ARIMA"
        except Exception:
            x = np.arange(len(train))
            coeffs = np.polyfit(x, train, 1)
            pred = [coeffs[0] * (len(train) + i) + coeffs[1] for i in range(holdout)]
            method = "Linear"

        mape = np.mean([abs((a - f) / a) for a, f in zip(test, pred) if abs(a) > 1e-9]) * 100
        return ForecastQuality(
            ticker=state.ticker,
            forecast_type="price",
            method=method,
            mape=round(float(mape), 2),
            holdout_periods=holdout,
        )
    except Exception as exc:
        logger.warning("Price MAPE computation failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedAnalysisAgent(BaseAgent):
    name = "advanced_analysis"

    def run(self, state: PipelineState, weights: dict[str, float] | None = None) -> PipelineState:
        self._start_step(state)
        self._log_input(ticker=state.ticker, has_kpis=state.kpis is not None,
                        has_ratios=state.ratios is not None)
        try:
            dcf = _compute_dcf(state)
            altman = _compute_altman_z(state)
            percentiles = _compute_percentiles(state)
            wscore = _compute_weighted_score(state, weights)
            backtest = _compute_backtest(state)
            pfq = _compute_price_forecast_mape(state)

            state.advanced = AdvancedAnalysisResult(
                ticker=state.ticker,
                dcf=dcf,
                altman_z=altman,
                valuation_percentiles=percentiles,
                weighted_score=wscore,
                backtest=backtest,
                price_forecast_quality=pfq,
            )
            self._log_output(
                dcf=f"fair_mid={dcf.fair_value_mid if dcf else 'N/A'}",
                altman_z=f"{altman.z_score if altman else 'N/A'} ({altman.zone if altman else ''})",
                composite=wscore.composite_score,
                signal=wscore.signal,
                backtest_excess=f"{backtest.excess_return_pct if backtest else 'N/A'}%",
                mape=f"{pfq.mape if pfq else 'N/A'}%",
            )
            self._finish_step(state,
                f"DCF fair={dcf.fair_value_mid if dcf else 'N/A'} | "
                f"Z={altman.z_score if altman else 'N/A'} ({altman.zone if altman else ''}) | "
                f"Score={wscore.composite_score} ({wscore.signal})"
            )
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

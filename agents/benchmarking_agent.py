"""
Agent 5 — Competitor Benchmarking Agent.
Ranks the primary ticker vs peers across key ratios and flags
over/under-performance.
"""
from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.ratios import TickerRatios
from models.benchmarking import BenchmarkResult, PeerRanking

# (metric_attr, label, lower_is_better)
BENCHMARK_METRICS: list[tuple[str, str, bool]] = [
    ("pe_ratio",           "P/E Ratio",          True),
    ("forward_pe",         "Forward P/E",         True),
    ("pb_ratio",           "P/B Ratio",           True),
    ("roe",                "ROE",                 False),
    ("roa",                "ROA",                 False),
    ("net_margin",         "Net Margin",          False),
    ("operating_margin",   "Operating Margin",    False),
    ("gross_margin",       "Gross Margin",        False),
    ("debt_to_equity",     "Debt/Equity",         True),
    ("current_ratio",      "Current Ratio",       False),
    ("volatility_annualised", "Volatility",       True),
    ("max_drawdown",       "Max Drawdown",        True),  # more negative = worse
]


def _rank_metric(
    primary: TickerRatios,
    peers: list[TickerRatios],
    attr: str,
    lower_is_better: bool,
) -> list[PeerRanking]:
    all_tickers = [primary] + peers
    vals: list[tuple[str, float]] = []
    for t in all_tickers:
        v = getattr(t, attr, None)
        if v is not None:
            vals.append((t.ticker, float(v)))

    if not vals:
        return []

    vals.sort(key=lambda x: x[1], reverse=(not lower_is_better))
    total = len(vals)
    rankings = []
    for rank, (ticker, value) in enumerate(vals, start=1):
        rankings.append(PeerRanking(
            metric=attr,
            ticker=ticker,
            value=round(value, 4),
            rank=rank,
            total_peers=total,
        ))
    return rankings


class BenchmarkingAgent(BaseAgent):
    name = "benchmarking"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        if state.ratios is None:
            self._fail_step(state, "No ratio data for benchmarking")
            return state

        try:
            primary = state.ratios.primary
            peers = state.ratios.competitors
            all_tickers = [primary] + peers

            all_rankings: list[PeerRanking] = []
            for attr, _label, lower_is_better in BENCHMARK_METRICS:
                all_rankings.extend(_rank_metric(primary, peers, attr, lower_is_better))

            # Count how many times primary ranks #1
            primary_ranks = [r for r in all_rankings if r.ticker == primary.ticker]
            rank_sum = sum(r.rank for r in primary_ranks)
            avg_rank = rank_sum / len(primary_ranks) if primary_ranks else None

            # Flag over/under performing metrics
            over = [r.metric for r in primary_ranks if r.rank == 1]
            under = [r.metric for r in primary_ranks if r.rank == r.total_peers]

            # Build comparison table (one row per ticker)
            table_rows = []
            for t in all_tickers:
                row: dict = {"ticker": t.ticker}
                for attr, label, _ in BENCHMARK_METRICS:
                    v = getattr(t, attr, None)
                    row[label] = round(v, 4) if v is not None else None
                table_rows.append(row)

            # Overall rank (1 = best)
            overall_rank: int | None = None
            if avg_rank is not None:
                sorted_tickers = sorted(
                    [t.ticker for t in all_tickers],
                    key=lambda tk: sum(
                        r.rank for r in all_rankings if r.ticker == tk
                    )
                )
                overall_rank = sorted_tickers.index(primary.ticker) + 1 if primary.ticker in sorted_tickers else None

            state.benchmark = BenchmarkResult(
                primary_ticker=primary.ticker,
                peer_tickers=[p.ticker for p in peers],
                rankings=all_rankings,
                overall_rank=overall_rank,
                over_performing_metrics=over,
                under_performing_metrics=under,
                peer_comparison_table=table_rows,
            )
            self._finish_step(
                state,
                f"Overall rank: {overall_rank}/{len(all_tickers)}, Over-performing: {over}"
            )
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

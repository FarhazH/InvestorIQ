"""
InvestIQ — 5-Ticker Evaluation Script
======================================
Runs the full 11-agent pipeline on 5 reference tickers (demo mode) and
measures:
  - Pipeline success rate (steps completed without critical error)
  - Grounding rate (% of AI reasons with verified metrics)
  - Price forecast MAPE (hold-out accuracy)
  - End-to-end wall-clock time
  - Recommendation distribution

Usage (from investiq/):
    python scripts/eval_pipeline.py

Always runs in DEMO_MODE=true so no API keys are needed.
Results are printed to stdout as a formatted table and saved as JSON.

Exit code 0 if success_rate >= 80%, else 1 (useful in CI).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Always demo mode for the eval
os.environ["DEMO_MODE"] = "true"
os.environ.setdefault("LOG_LEVEL", "WARNING")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.orchestrator import PipelineOrchestrator

# ── Evaluation config ─────────────────────────────────────────────────────────

EVAL_TICKERS = [
    ("AAPL", ["MSFT", "GOOGL", "AMZN"]),
    ("MSFT", ["AAPL", "GOOGL", "AMZN"]),
    ("GOOGL", ["AAPL", "MSFT", "META"]),
    ("AMZN", ["AAPL", "MSFT", "GOOGL"]),
    ("NVDA", ["AMD", "INTC", "QCOM"]),
]

PASS_THRESHOLD_SUCCESS_RATE = 0.80  # 80% of tickers must complete fully
PASS_THRESHOLD_GROUNDING_RATE = 0.50  # ≥50% of AI reasons must be verified


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(val: float | None, denom: int) -> str:
    if denom == 0 or val is None:
        return "N/A"
    return f"{val / denom:.0%}"


def _fmt(val: float | None, unit: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}{unit}"


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_eval() -> dict:
    results = []
    total = len(EVAL_TICKERS)
    n_success = 0
    n_total_steps = 0
    n_done_steps = 0
    grounding_ok = 0
    grounding_total = 0
    mape_values: list[float] = []

    print("=" * 72)
    print("InvestIQ — Pipeline Evaluation (Demo Mode)")
    print(f"Tickers: {[t for t, _ in EVAL_TICKERS]}")
    print("=" * 72)

    for ticker, competitors in EVAL_TICKERS:
        print(f"\n▶ Running: {ticker} vs {', '.join(competitors)} ...")
        t0 = time.perf_counter()
        try:
            orch = PipelineOrchestrator()
            state = orch.run(ticker, competitors)
            elapsed = time.perf_counter() - t0

            # Step completion
            n_steps = len(state.steps)
            done = sum(1 for s in state.steps if s.status == "done")
            n_total_steps += n_steps
            n_done_steps += done
            success = not state.has_critical_error and done >= n_steps * 0.8

            if success:
                n_success += 1

            # Grounding
            if state.recommendation and state.recommendation.grounding_flags:
                for f in state.recommendation.grounding_flags:
                    grounding_total += 1
                    if f.match:
                        grounding_ok += 1

            # MAPE
            mape = None
            if state.advanced and state.advanced.price_forecast_quality:
                mape = state.advanced.price_forecast_quality.mape
                if mape is not None:
                    mape_values.append(mape)

            rec = state.recommendation.recommendation if state.recommendation else "N/A"
            conf = state.recommendation.confidence if state.recommendation else None
            risk = state.risk.overall_risk_level if state.risk else "N/A"
            composite = (
                state.advanced.weighted_score.composite_score
                if state.advanced and state.advanced.weighted_score else None
            )

            row = {
                "ticker": ticker,
                "success": success,
                "steps_done": done,
                "steps_total": n_steps,
                "recommendation": rec,
                "confidence": f"{conf:.0%}" if conf else "N/A",
                "composite_score": f"{composite:.1f}" if composite else "N/A",
                "risk": risk,
                "mape": f"{mape:.1f}%" if mape else "N/A",
                "elapsed_s": f"{elapsed:.1f}s",
                "errors": [e.error if hasattr(e, "error") else str(e) for e in state.errors],
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row = {
                "ticker": ticker,
                "success": False,
                "steps_done": 0,
                "steps_total": 11,
                "recommendation": "ERROR",
                "confidence": "N/A",
                "composite_score": "N/A",
                "risk": "N/A",
                "mape": "N/A",
                "elapsed_s": f"{elapsed:.1f}s",
                "errors": [str(exc)],
            }

        results.append(row)
        status_icon = "✅" if row["success"] else "❌"
        print(
            f"  {status_icon} {ticker}: {row['recommendation']} "
            f"(conf={row['confidence']}, score={row['composite_score']}, "
            f"risk={row['risk']}, MAPE={row['mape']}, {row['elapsed_s']})"
        )
        if row["errors"]:
            for e in row["errors"]:
                print(f"      ⚠ {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    success_rate = n_success / total
    step_completion = n_done_steps / n_total_steps if n_total_steps > 0 else 0
    grounding_rate = grounding_ok / grounding_total if grounding_total > 0 else None
    avg_mape = sum(mape_values) / len(mape_values) if mape_values else None

    rec_dist: dict[str, int] = {}
    for r in results:
        rec_dist[r["recommendation"]] = rec_dist.get(r["recommendation"], 0) + 1

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tickers": total,
        "success_count": n_success,
        "success_rate": round(success_rate, 4),
        "step_completion_rate": round(step_completion, 4),
        "grounding_rate": round(grounding_rate, 4) if grounding_rate is not None else None,
        "avg_price_mape": round(avg_mape, 2) if avg_mape is not None else None,
        "recommendation_distribution": rec_dist,
        "passed": success_rate >= PASS_THRESHOLD_SUCCESS_RATE,
        "results": results,
    }

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Tickers evaluated  : {total}")
    print(f"  Success rate       : {success_rate:.0%} ({n_success}/{total})"
          f"  {'✅ PASS' if success_rate >= PASS_THRESHOLD_SUCCESS_RATE else '❌ FAIL'}")
    print(f"  Step completion    : {step_completion:.0%}")
    print(f"  Grounding rate     : {f'{grounding_rate:.0%}' if grounding_rate is not None else 'N/A'}")
    print(f"  Avg price MAPE     : {f'{avg_mape:.1f}%' if avg_mape is not None else 'N/A'}")
    print(f"  Rec distribution   : {rec_dist}")

    # Save results
    out_path = Path(__file__).parent.parent / ".bob" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to: {out_path}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    summary = run_eval()
    sys.exit(0 if summary["passed"] else 1)

"""
IBM watsonx.ai client — wraps ibm-watsonx-ai SDK for Granite inference.

Improvements over v1:
- Wall-clock timeout on every Granite call (GRANITE_CALL_TIMEOUT, default 60s)
- Exponential backoff on transport failures
- Independent JSON-retry loop (up to GRANITE_JSON_RETRIES=2):
    on invalid JSON, re-calls Granite with a "fix the JSON" addendum
- Optional Pydantic validator callable passed to call_granite_json
- Structured per-call log line: {model, attempt, latency_ms, json_valid, mode}
- granite_mode return value: "live" | "demo"
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from typing import Any, Callable

from config.settings import settings

logger = logging.getLogger(__name__)

_client_cache: Any = None

# ── Fix-up addendum appended on JSON parse failure ────────────────────────────
_JSON_FIXUP = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY the JSON object — no markdown, no explanation, no extra text. "
    "Start with { and end with }."
)


# ── Client initialisation ─────────────────────────────────────────────────────

def _get_client() -> Any:
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    try:
        from ibm_watsonx_ai import APIClient, Credentials  # type: ignore
        creds = Credentials(
            url=settings.watsonx_url,
            api_key=settings.watsonx_api_key,
        )
        _client_cache = APIClient(creds)
        logger.info("watsonx.ai client initialised (url=%s)", settings.watsonx_url)
        return _client_cache
    except Exception as exc:
        logger.error("Failed to init watsonx client: %s", exc)
        raise


def _build_params(max_tokens: int = 1024) -> dict:
    return {
        "decoding_method": "greedy",
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "stop_sequences": [],
    }


# ── Core call with timeout ────────────────────────────────────────────────────

def _call_once(prompt: str, max_tokens: int) -> str:
    """Execute a single Granite inference call inside a thread for timeout control."""
    client = _get_client()
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
    model = ModelInference(
        model_id=settings.granite_model_id,
        api_client=client,
        project_id=settings.watsonx_project_id,
    )
    return model.generate_text(
        prompt=prompt,
        params=_build_params(max_tokens),
    )


def call_granite(
    prompt: str,
    max_tokens: int = 1024,
    retries: int | None = None,
) -> tuple[str, str]:
    """
    Call IBM Granite and return (text, mode).
    mode is "live" or "demo".

    Uses exponential backoff: delay, 2*delay, 4*delay, ...
    Applies wall-clock timeout per attempt.
    """
    if settings.demo_mode:
        logger.info("DEMO MODE — skipping real Granite call")
        return _demo_response(prompt), "demo"

    max_retries = retries if retries is not None else settings.max_retries
    delay = settings.retry_delay
    timeout = settings.granite_call_timeout

    for attempt in range(1, max_retries + 1):
        t0 = time.monotonic()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_once, prompt, max_tokens)
                response = future.result(timeout=timeout)

            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "Granite call OK | model=%s attempt=%d latency_ms=%d tokens=%d",
                settings.granite_model_id, attempt, latency_ms, max_tokens,
            )
            return response, "live"

        except concurrent.futures.TimeoutError:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "Granite call TIMEOUT | attempt=%d/%d timeout=%ds latency_ms=%d",
                attempt, max_retries, timeout, latency_ms,
            )
            exc: Exception = TimeoutError(f"Granite call timed out after {timeout}s")

        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "Granite call FAILED | attempt=%d/%d error=%s latency_ms=%d",
                attempt, max_retries, e, latency_ms,
            )
            exc = e

        if attempt < max_retries:
            backoff = delay * (2 ** (attempt - 1))   # exponential: 1s, 2s, 4s
            logger.info("Retrying in %.1fs (attempt %d/%d)", backoff, attempt + 1, max_retries)
            time.sleep(backoff)

    raise RuntimeError(f"All {max_retries} Granite attempts failed") from exc


def call_granite_json(
    prompt: str,
    max_tokens: int = 1024,
    retries: int | None = None,
    validator: Callable[[dict], None] | None = None,
) -> tuple[dict, str]:
    """
    Call Granite and return (parsed_dict, mode).

    JSON-retry loop (up to GRANITE_JSON_RETRIES):
      - On invalid JSON: re-calls with fixup addendum
      - On Pydantic validation failure: re-calls with schema addendum
    Returns (dict, mode) where mode is "live" | "demo".
    """
    json_retries = settings.granite_json_retries
    current_prompt = prompt

    for json_attempt in range(1, json_retries + 2):  # +1 original + N retries
        raw, mode = call_granite(current_prompt, max_tokens=max_tokens, retries=retries)

        try:
            parsed = _parse_json_response(raw)
        except ValueError as parse_err:
            if json_attempt <= json_retries:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s — retrying with fixup",
                    json_attempt, json_retries + 1, parse_err,
                )
                current_prompt = prompt + _JSON_FIXUP
                continue
            # Exhausted JSON retries — raise
            logger.error("JSON parse failed after all retries. Raw response: %.300s", raw)
            raise

        # Optional Pydantic / structural validation
        if validator is not None:
            try:
                validator(parsed)
            except Exception as val_err:
                if json_attempt <= json_retries:
                    logger.warning(
                        "JSON validation failed (attempt %d/%d): %s — retrying",
                        json_attempt, json_retries + 1, val_err,
                    )
                    current_prompt = (
                        prompt
                        + f"\n\nYour previous JSON was structurally invalid: {val_err}. "
                        "Return the corrected JSON only."
                    )
                    continue
                logger.error("JSON validation failed after all retries: %s", val_err)
                raise

        logger.debug(
            "Granite JSON OK | mode=%s attempt=%d keys=%s",
            mode, json_attempt, list(parsed.keys())[:8],
        )
        return parsed, mode

    # Should never reach here but satisfy type checker
    raise RuntimeError("call_granite_json: unexpected exit")


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """Strip markdown code fences and parse JSON. Raises ValueError on failure."""
    # Remove ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Granite returned invalid JSON: {exc}") from exc


# ── Validators for each LLM-powered agent ─────────────────────────────────────

def validate_sentiment_json(d: dict) -> None:
    """Structural check for sentiment output."""
    if "items" not in d:
        raise ValueError("Missing 'items' key in sentiment JSON")
    if not isinstance(d["items"], list):
        raise ValueError("'items' must be a list")
    if "aggregate_score" not in d:
        raise ValueError("Missing 'aggregate_score'")


def validate_recommendation_json(d: dict) -> None:
    """Structural check for recommendation output."""
    required = {"recommendation", "confidence", "reasons"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if d["recommendation"] not in ("Overweight", "Hold", "Underweight"):
        raise ValueError(f"Invalid recommendation value: {d['recommendation']!r}")
    if not isinstance(d.get("reasons"), list) or len(d["reasons"]) == 0:
        raise ValueError("'reasons' must be a non-empty list")


# ── Demo mode stubs ───────────────────────────────────────────────────────────

def _demo_response(prompt: str) -> str:
    """Return canned responses keyed by prompt type."""
    is_recommendation = (
        "senior equity research analyst" in prompt.lower()
        or "recommendation" in prompt[:200].lower()
    )
    is_sentiment = ("sentiment analyst" in prompt.lower()) and not is_recommendation

    if is_sentiment:
        return json.dumps({
            "ticker": "AAPL",
            "items": [
                {"headline": "Apple reports record revenue",
                 "sentiment": "positive", "confidence": 0.95, "key_theme": "earnings beat"},
                {"headline": "iPhone sales growth slows in China",
                 "sentiment": "negative", "confidence": 0.80, "key_theme": "market headwinds"},
                {"headline": "Apple expands services segment",
                 "sentiment": "positive", "confidence": 0.88, "key_theme": "diversification"},
            ],
            "aggregate_score": 0.63,
            "sentiment_label": "Bullish",
            "top_themes": ["earnings beat", "diversification", "market headwinds"],
            "total_articles": 3,
        })

    # Default: recommendation stub
    return json.dumps({
        "ticker": "AAPL",
        "recommendation": "Overweight",
        "confidence": 0.82,
        "target_horizon": "12 months",
        "reasons": [
            {"point": "Strong revenue growth YoY",
             "metric": "revenue_growth_yoy", "value": "8.1%"},
            {"point": "High net margin vs peers",
             "metric": "net_margin", "value": "25.3%"},
            {"point": "Positive sentiment index",
             "metric": "sentiment_score", "value": "0.63"},
        ],
        "key_risks": ["China market slowdown", "Regulatory scrutiny"],
        "forward_outlook": (
            "AAPL shows strong fundamentals with expanding margins and a robust services business. "
            "Near-term risks include macro headwinds and China exposure."
        ),
        "metrics_cited": {
            "pe_ratio": 28.5,
            "revenue_growth_yoy": "8.1%",
            "net_margin": "25.3%",
            "debt_to_equity": 1.73,
            "sentiment_score": 0.63,
            "overall_risk_score": 42,
        },
        "disclaimer": (
            "This output is generated by an AI model and is NOT financial advice. "
            "Always consult a licensed financial advisor before making investment decisions."
        ),
    })

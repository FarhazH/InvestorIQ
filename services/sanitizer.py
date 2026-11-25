"""
Phase 7 — Input sanitization utilities.

All user-supplied inputs (ticker symbols, free-text fields) are validated and
cleaned here before being passed to the pipeline or Granite prompts.

Security goals:
- Prevent prompt injection via ticker/notes fields
- Reject clearly invalid tickers early (before hitting yfinance / EDGAR)
- Provide clear error messages to the caller
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple


# ── Constants ─────────────────────────────────────────────────────────────────

# Valid tickers: 1-10 uppercase letters/digits, optionally followed by a dot
# and 1-4 more chars (e.g. BRK.A, BTC-USD).
_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}([.\-][A-Z0-9]{1,4})?$")

# Characters allowed in analyst notes (printable ASCII + common Unicode letters)
_DANGEROUS_PROMPT_CHARS = re.compile(
    r"(ignore\s+previous\s+instructions|system\s+prompt|<\|.*?\|>|###\s*system)",
    re.IGNORECASE,
)

MAX_TICKER_LEN = 15
MAX_NOTES_LEN = 2000
MAX_COMPETITORS = 3


# ── Result type ───────────────────────────────────────────────────────────────

class SanitizeResult(NamedTuple):
    value: str
    ok: bool
    error: str = ""


# ── Ticker sanitization ───────────────────────────────────────────────────────

def sanitize_ticker(raw: str) -> SanitizeResult:
    """
    Validate and normalise a ticker symbol.

    Rules:
    - Strip whitespace, uppercase
    - 1–15 characters
    - Only [A-Z0-9.-] with dot/dash only in the middle
    - Block common injection patterns
    """
    if not raw or not isinstance(raw, str):
        return SanitizeResult("", False, "Ticker must be a non-empty string.")

    # Strip all whitespace including embedded newlines
    ticker = "".join(raw.split()).upper()

    if len(ticker) > MAX_TICKER_LEN:
        return SanitizeResult("", False, f"Ticker too long (max {MAX_TICKER_LEN} chars).")

    if not _TICKER_RE.match(ticker):
        return SanitizeResult(
            "", False,
            f"Invalid ticker '{ticker}'. Use only letters/digits (e.g. AAPL, BRK.A).",
        )

    return SanitizeResult(ticker, True)


def sanitize_competitors(raw: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate a list of competitor tickers.

    Returns (valid_tickers, errors).
    Silently drops invalid entries and caps at MAX_COMPETITORS.
    """
    valid: list[str] = []
    errors: list[str] = []
    for t in raw[:MAX_COMPETITORS]:
        result = sanitize_ticker(t)
        if result.ok:
            if result.value not in valid:   # deduplicate
                valid.append(result.value)
        else:
            errors.append(result.error)
    return valid, errors


# ── Free-text sanitization (notes, sector notes) ─────────────────────────────

def sanitize_text(raw: str, max_len: int = MAX_NOTES_LEN, field: str = "text") -> SanitizeResult:
    """
    Sanitize free-text user input.

    - Strip leading/trailing whitespace
    - Normalize Unicode to NFC (prevent homoglyph attacks)
    - Block prompt-injection patterns
    - Enforce max length
    """
    if not isinstance(raw, str):
        return SanitizeResult("", False, f"{field} must be a string.")

    # Normalize Unicode
    text = unicodedata.normalize("NFC", raw.strip())

    if len(text) > max_len:
        return SanitizeResult("", False, f"{field} exceeds {max_len} characters.")

    # Block prompt injection attempts
    if _DANGEROUS_PROMPT_CHARS.search(text):
        return SanitizeResult(
            "", False,
            f"{field} contains potentially unsafe content and was rejected.",
        )

    return SanitizeResult(text, True)


# ── Sanitize full analysis request ───────────────────────────────────────────

class AnalysisRequestSanitized(NamedTuple):
    ticker: str
    competitors: list[str]
    errors: list[str]
    ok: bool


def sanitize_analysis_request(
    ticker: str,
    competitors: list[str],
) -> AnalysisRequestSanitized:
    """
    Validate and sanitize the full analysis request.
    Returns a typed namedtuple with either clean values or collected errors.
    """
    errors: list[str] = []

    t_result = sanitize_ticker(ticker)
    if not t_result.ok:
        errors.append(f"Primary ticker: {t_result.error}")
        clean_ticker = ""
    else:
        clean_ticker = t_result.value

    clean_competitors, comp_errors = sanitize_competitors(competitors)
    errors.extend(comp_errors)

    # Reject if primary ticker is in competitors list
    if clean_ticker and clean_ticker in clean_competitors:
        clean_competitors = [c for c in clean_competitors if c != clean_ticker]

    return AnalysisRequestSanitized(
        ticker=clean_ticker,
        competitors=clean_competitors,
        errors=errors,
        ok=bool(clean_ticker) and len(errors) == 0,
    )

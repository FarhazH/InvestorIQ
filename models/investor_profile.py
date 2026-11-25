"""
Pydantic models for Phase 5 — Investor Profile (client-facing personalization).

The InvestorProfile is captured in the Streamlit sidebar and passed to the
recommendation prompt so IBM Granite can tailor the output to the specific
investor's risk tolerance, horizon, and context.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Investor Profile ──────────────────────────────────────────────────────────

class InvestorProfile(BaseModel):
    """
    Client profile that personalizes the investment analysis.
    Captured once per session in the sidebar; stored in session state and
    serialized into the report and PDF.
    """

    # Risk tolerance
    risk_tolerance: Literal["Conservative", "Moderate", "Aggressive"] = "Moderate"

    # Investment horizon
    horizon: Literal["Short-term (< 1 year)", "Medium-term (1–3 years)", "Long-term (3+ years)"] = (
        "Medium-term (1–3 years)"
    )

    # Purpose of this analysis
    intent: Literal[
        "New position", "Adding to existing position", "Portfolio review", "Research only"
    ] = "Research only"

    # Optional sector bias / exclusions (free text, max 200 chars)
    sector_notes: str = Field(default="", max_length=200)

    # Optional portfolio size bucket (purely for context, not used in math)
    portfolio_size: Literal[
        "< $10K", "$10K – $100K", "$100K – $1M", "$1M+"
    ] = "$10K – $100K"

    # ESG preference
    esg_preference: Literal["No preference", "ESG-aware", "ESG-only"] = "No preference"

    @property
    def risk_level_num(self) -> int:
        """Numeric risk level: Conservative=1, Moderate=2, Aggressive=3."""
        return {"Conservative": 1, "Moderate": 2, "Aggressive": 3}.get(self.risk_tolerance, 2)

    @property
    def horizon_years(self) -> float:
        """Approximate years for the horizon."""
        return {"Short-term (< 1 year)": 0.5, "Medium-term (1–3 years)": 2.0, "Long-term (3+ years)": 5.0}.get(
            self.horizon, 2.0
        )

    def to_prompt_context(self) -> str:
        """
        Render the profile as a compact text block for injection into the
        Granite recommendation prompt.
        """
        lines = [
            "=== INVESTOR PROFILE ===",
            f"Risk tolerance  : {self.risk_tolerance}",
            f"Investment horizon: {self.horizon}",
            f"Intent          : {self.intent}",
            f"Portfolio size  : {self.portfolio_size}",
            f"ESG preference  : {self.esg_preference}",
        ]
        if self.sector_notes.strip():
            lines.append(f"Sector notes    : {self.sector_notes}")
        lines.append(
            "\nPlease tailor your recommendation to this investor profile. "
            "A Conservative investor should receive more weight on downside risk "
            "and capital preservation; an Aggressive investor can tolerate higher risk "
            "for growth potential."
        )
        return "\n".join(lines)


# ── Watchlist entry ───────────────────────────────────────────────────────────

class WatchlistEntry(BaseModel):
    """
    One entry saved to the watchlist after a completed pipeline run.
    Stored in Streamlit session state (and optionally serialized to disk).
    """
    ticker: str
    run_id: str
    recommendation: str           # "Overweight" | "Hold" | "Underweight"
    confidence: float = 0.0
    composite_score: float | None = None
    risk_level: str = "Medium"    # "Low" | "Medium" | "High"
    generated_at: str = ""
    competitors: list[str] = Field(default_factory=list)
    granite_mode: str = "demo"
    notes: str = ""               # optional analyst note at save-time


# ── Portfolio position ────────────────────────────────────────────────────────

class PortfolioPosition(BaseModel):
    """
    One position in a user-defined portfolio for portfolio-mode analysis.
    Weight is a percentage (0–100); the UI normalises to 100%.
    """
    ticker: str
    weight_pct: float = Field(ge=0.0, le=100.0)  # % of portfolio
    description: str = ""


class PortfolioConfig(BaseModel):
    """
    User-defined multi-ticker portfolio for blended analysis.
    """
    positions: list[PortfolioPosition] = Field(default_factory=list)
    name: str = "My Portfolio"

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @property
    def total_weight(self) -> float:
        return sum(p.weight_pct for p in self.positions)

    def normalised_weights(self) -> dict[str, float]:
        """Return weight fractions that sum to 1.0."""
        total = self.total_weight
        if total <= 0:
            return {p.ticker: 0.0 for p in self.positions}
        return {p.ticker: p.weight_pct / total for p in self.positions}


# ── Blended portfolio result ──────────────────────────────────────────────────

class PortfolioAnalysisResult(BaseModel):
    """
    Aggregated metrics across all positions in a portfolio.
    Computed after running the pipeline on each individual ticker.
    """
    portfolio_name: str = "My Portfolio"
    positions: list[PortfolioPosition] = Field(default_factory=list)

    # Weighted-average composite score
    weighted_composite_score: float | None = None

    # Blended recommendation (derived from composite_score)
    blended_recommendation: str = "Hold"
    blended_confidence: float = 0.5

    # Weighted-average risk score (0–100)
    weighted_risk_score: float | None = None
    blended_risk_level: str = "Medium"

    # Per-ticker results (ticker → {recommendation, score, risk})
    ticker_results: dict[str, dict] = Field(default_factory=dict)

    # Diversification note
    diversification_note: str = ""

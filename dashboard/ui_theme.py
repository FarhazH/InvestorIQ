"""
Phase 6 — UI/UX Polish helpers.

Centralises design tokens (IBM Carbon-inspired), chart palette, and
formatting helpers so every part of the dashboard uses a consistent look.
These are pure-Python utilities — no Streamlit import required — which makes
them fully unit-testable.
"""
from __future__ import annotations

from datetime import datetime, timezone


# ── IBM Carbon-inspired colour tokens ─────────────────────────────────────────
# Derived from IBM Carbon Design System v11 (https://carbondesignsystem.com/)
# Using the Gray-10 / Blue-60 / Green-50 / Red-60 / Warm-Gray ramp.

class C:
    """Carbon design token namespace."""

    # Blues (primary)
    BLUE_60   = "#0f62fe"   # IBM Blue 60 — primary action
    BLUE_40   = "#78a9ff"   # IBM Blue 40 — hover / secondary
    BLUE_10   = "#edf5ff"   # IBM Blue 10 — light background tint

    # Greens (positive / overweight)
    GREEN_60  = "#198038"   # IBM Green 60
    GREEN_40  = "#42be65"   # IBM Green 40
    GREEN_10  = "#defbe6"   # IBM Green 10

    # Reds (negative / underweight / high risk)
    RED_60    = "#da1e28"   # IBM Red 60
    RED_40    = "#ff8389"   # IBM Red 40
    RED_10    = "#fff1f1"   # IBM Red 10

    # Yellows (hold / medium risk / warning)
    YELLOW_30 = "#f1c21b"   # IBM Yellow 30
    YELLOW_10 = "#fcf4d6"   # IBM Yellow 10

    # Purples (secondary accent)
    PURPLE_60 = "#8a3ffc"   # IBM Purple 60
    PURPLE_40 = "#be95ff"   # IBM Purple 40

    # Neutrals / UI Shell
    GRAY_100  = "#161616"   # IBM Black / body text
    GRAY_90   = "#262626"
    GRAY_80   = "#393939"
    GRAY_60   = "#6f6f6f"   # IBM muted text
    GRAY_30   = "#c6c6c6"   # border
    GRAY_20   = "#e0e0e0"   # subtle border
    GRAY_10   = "#f4f4f4"   # surface / card background
    WHITE     = "#ffffff"


# ── Semantic aliases ───────────────────────────────────────────────────────────

REC_COLORS = {
    "Overweight":  C.GREEN_60,
    "Hold":        "#b45309",   # amber-700, more legible than Yellow
    "Underweight": C.RED_60,
}

RISK_COLORS = {
    "Low":    C.GREEN_60,
    "Medium": "#b45309",
    "High":   C.RED_60,
}

ZONE_COLORS = {
    "Safe":     C.GREEN_60,
    "Grey":     "#b45309",
    "Distress": C.RED_60,
}


# ── Chart palette ──────────────────────────────────────────────────────────────
# Ordered list of hex colours — used as `color_discrete_sequence` in Plotly
# and manually for trace colours.

CHART_PALETTE = [
    C.BLUE_60,     # primary series
    C.PURPLE_60,   # secondary series
    C.GREEN_60,    # tertiary / positive
    "#f46800",     # orange — quaternary
    C.RED_60,      # quinary / negative
    C.GRAY_60,     # neutral
]

# Shorthand accessors
PRIMARY_COLOR   = CHART_PALETTE[0]
SECONDARY_COLOR = CHART_PALETTE[1]
SUCCESS_COLOR   = CHART_PALETTE[2]
WARNING_COLOR   = "#b45309"
DANGER_COLOR    = CHART_PALETTE[4]
NEUTRAL_COLOR   = CHART_PALETTE[5]


# ── Chart layout defaults ──────────────────────────────────────────────────────

def chart_layout(title: str = "", height: int = 400, **kwargs) -> dict:
    """
    Return a Plotly layout dict with consistent IBM Carbon styling.
    Pass extra kwargs to override individual properties.
    """
    base = {
        "title": {
            "text": title,
            "font": {"size": 14, "color": C.GRAY_100, "family": "IBM Plex Sans, Segoe UI, sans-serif"},
        },
        "height": height,
        "plot_bgcolor": C.WHITE,
        "paper_bgcolor": C.WHITE,
        "font": {"family": "IBM Plex Sans, Segoe UI, sans-serif", "size": 12, "color": C.GRAY_80},
        "legend": {"orientation": "h", "y": -0.15, "font": {"size": 11}},
        "margin": {"t": 50, "b": 40, "l": 40, "r": 20},
        "xaxis": {
            "gridcolor": C.GRAY_20,
            "linecolor": C.GRAY_30,
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": C.GRAY_20,
            "linecolor": C.GRAY_30,
            "zeroline": False,
        },
    }
    base.update(kwargs)
    return base


# ── Gauge layout ──────────────────────────────────────────────────────────────

def risk_gauge_steps() -> list[dict]:
    """Standard colour steps for a 0–100 risk gauge."""
    return [
        {"range": [0, 35],  "color": C.GREEN_10},
        {"range": [35, 65], "color": C.YELLOW_10},
        {"range": [65, 100],"color": C.RED_10},
    ]


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def time_ago(iso_ts: str) -> str:
    """
    Return a human-readable relative timestamp, e.g. "3 minutes ago".
    Accepts an ISO 8601 string (UTC, with or without timezone suffix).
    """
    if not iso_ts:
        return "unknown time ago"
    try:
        # Normalise: remove trailing Z, add UTC offset if bare
        ts_str = iso_ts.strip()
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        elif "+" not in ts_str and ts_str.count("-") <= 2:
            ts_str += "+00:00"
        ts = datetime.fromisoformat(ts_str)
        now = datetime.now(timezone.utc)
        # Make ts timezone-aware if it isn't
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta_s = (now - ts).total_seconds()
        if delta_s < 60:
            return f"{int(delta_s)}s ago"
        if delta_s < 3600:
            return f"{int(delta_s / 60)}m ago"
        if delta_s < 86400:
            return f"{int(delta_s / 3600)}h ago"
        return f"{int(delta_s / 86400)}d ago"
    except Exception:
        return iso_ts[:19].replace("T", " ") + " UTC"


def fmt_ts(iso_ts: str) -> str:
    """Format an ISO timestamp as 'YYYY-MM-DD HH:MM UTC'."""
    try:
        return iso_ts[:19].replace("T", " ") + " UTC"
    except Exception:
        return iso_ts


# ── CSS helper ────────────────────────────────────────────────────────────────

def build_css() -> str:
    """
    Return the full custom CSS block for the Streamlit dashboard.
    Applies IBM Carbon-inspired tokens throughout.
    """
    return f"""
<style>
/* ── IBM Carbon token overrides ─────────────────────────────────────────── */
:root {{
    --ibm-blue-60:   {C.BLUE_60};
    --ibm-green-60:  {C.GREEN_60};
    --ibm-red-60:    {C.RED_60};
    --ibm-gray-10:   {C.GRAY_10};
    --ibm-gray-20:   {C.GRAY_20};
    --ibm-gray-60:   {C.GRAY_60};
    --ibm-gray-100:  {C.GRAY_100};
}}

/* ── Recommendation cards ────────────────────────────────────────────────── */
.rec-card {{
    border-radius: 8px;
    padding: 18px 20px;
    margin: 10px 0;
    font-weight: 700;
    font-size: 1.25em;
    text-align: center;
    letter-spacing: 0.01em;
}}
.overweight  {{ background: {C.GREEN_10};  color: {C.GREEN_60};  border: 2px solid {C.GREEN_60}; }}
.hold        {{ background: #fef9c3;        color: #92400e;        border: 2px solid #d97706; }}
.underweight {{ background: {C.RED_10};    color: {C.RED_60};    border: 2px solid {C.RED_60}; }}

/* ── Pipeline step status colours ───────────────────────────────────────── */
.step-done    {{ color: {C.GREEN_60}; font-weight: 600; }}
.step-running {{ color: {C.BLUE_60};  font-weight: 600; }}
.step-failed  {{ color: {C.RED_60};   font-weight: 600; }}
.step-warning {{ color: #d97706;      font-weight: 600; }}
.step-pending {{ color: {C.GRAY_60}; }}
.step-skipped {{ color: {C.GRAY_60}; font-style: italic; }}

/* ── Mode / status badges ────────────────────────────────────────────────── */
.badge-live     {{ background:{C.GREEN_10};  color:{C.GREEN_60}; padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }}
.badge-demo     {{ background:{C.GRAY_10};   color:{C.GRAY_80};  padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }}
.badge-fallback {{ background:#fef9c3;        color:#92400e;       padding:3px 10px; border-radius:12px; font-size:0.82em; font-weight:600; }}

/* ── Data quality text ───────────────────────────────────────────────────── */
.quality-high {{ color: {C.GREEN_60}; font-weight: 600; }}
.quality-med  {{ color: #d97706;      font-weight: 600; }}
.quality-low  {{ color: {C.RED_60};   font-weight: 600; }}

/* ── Grounding status ────────────────────────────────────────────────────── */
.grounding-ok   {{ color: {C.GREEN_60}; }}
.grounding-warn {{ color: #d97706; }}
.grounding-fail {{ color: {C.RED_60}; }}

/* ── Disclaimer callout ──────────────────────────────────────────────────── */
.disclaimer {{
    background: #fef9c3;
    border-left: 4px solid #d97706;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 0.85em;
    color: #78350f;
}}

/* ── Hero welcome card ───────────────────────────────────────────────────── */
.hero-card {{
    background: linear-gradient(135deg, {C.BLUE_10} 0%, #f0f4ff 100%);
    border: 1px solid {C.GRAY_20};
    border-radius: 12px;
    padding: 32px 36px;
    margin: 24px 0 32px;
}}
.hero-title {{
    font-size: 1.6em;
    font-weight: 700;
    color: {C.BLUE_60};
    margin-bottom: 8px;
}}
.hero-subtitle {{
    color: {C.GRAY_60};
    font-size: 1em;
    margin-bottom: 20px;
}}
.feature-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}}
.feature-item {{
    background: {C.WHITE};
    border: 1px solid {C.GRAY_20};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.88em;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}}
.feature-icon {{ font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }}

/* ── Pipeline stepper ────────────────────────────────────────────────────── */
.stepper-wrap {{
    display: flex;
    align-items: center;
    overflow-x: auto;
    gap: 0;
    padding: 12px 0 8px;
    margin-bottom: 8px;
}}
.step-node {{
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 76px;
    max-width: 90px;
    font-size: 0.72em;
    text-align: center;
    position: relative;
    flex: 1;
}}
.step-circle {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1em;
    font-weight: 700;
    margin-bottom: 4px;
    border: 2px solid {C.GRAY_20};
    background: {C.WHITE};
    z-index: 1;
}}
.step-circle-done    {{ background: {C.GREEN_60}; color: white; border-color: {C.GREEN_60}; }}
.step-circle-running {{ background: {C.BLUE_60};  color: white; border-color: {C.BLUE_60};  }}
.step-circle-failed  {{ background: {C.RED_60};   color: white; border-color: {C.RED_60};   }}
.step-circle-warning {{ background: #d97706;      color: white; border-color: #d97706;      }}
.step-circle-pending {{ background: {C.WHITE};    color: {C.GRAY_60}; border-color: {C.GRAY_30}; }}
.step-connector {{
    flex: 1;
    height: 2px;
    background: {C.GRAY_20};
    margin-top: -18px;
    min-width: 10px;
}}
.step-connector-done {{ background: {C.GREEN_60}; }}
.step-label {{ color: {C.GRAY_60}; line-height: 1.3; word-break: break-word; }}
.step-label-done    {{ color: {C.GREEN_60}; font-weight: 600; }}
.step-label-running {{ color: {C.BLUE_60};  font-weight: 600; }}
.step-label-failed  {{ color: {C.RED_60};   font-weight: 600; }}
.step-dur {{ color: {C.GRAY_60}; font-size: 0.85em; }}

/* ── Metric card skeleton ────────────────────────────────────────────────── */
.skeleton {{
    background: linear-gradient(90deg, {C.GRAY_10} 25%, {C.GRAY_20} 50%, {C.GRAY_10} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 6px;
    height: 70px;
    margin-bottom: 10px;
}}
@keyframes shimmer {{
    0%   {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* ── Last-updated chip ───────────────────────────────────────────────────── */
.last-updated {{
    display: inline-block;
    background: {C.GRAY_10};
    border: 1px solid {C.GRAY_20};
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 0.79em;
    color: {C.GRAY_60};
}}

/* ── Misc layout ─────────────────────────────────────────────────────────── */
.section-header {{
    font-size: 1.0em;
    font-weight: 700;
    color: {C.GRAY_100};
    border-bottom: 2px solid {C.BLUE_60};
    padding-bottom: 4px;
    margin: 20px 0 12px;
}}
</style>
"""

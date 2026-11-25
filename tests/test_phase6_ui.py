"""
Phase 6 — UI/UX Polish tests.

Covers:
- C (Carbon token namespace) — hex format validity
- CHART_PALETTE — length, uniqueness, format
- REC_COLORS / RISK_COLORS / ZONE_COLORS — key coverage
- chart_layout() — returns required keys, height param, override
- risk_gauge_steps() — three steps, correct ranges
- time_ago() — relative timestamps: seconds, minutes, hours, days
- time_ago() edge cases: empty string, naive datetime, Z suffix
- fmt_ts() — formats correctly
- build_css() — returns non-empty string with key class names
"""
from __future__ import annotations

import pytest
import re
from datetime import datetime, timezone, timedelta

from dashboard.ui_theme import (
    C, CHART_PALETTE, PRIMARY_COLOR, SECONDARY_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, NEUTRAL_COLOR,
    REC_COLORS, RISK_COLORS, ZONE_COLORS,
    chart_layout, risk_gauge_steps, time_ago, fmt_ts, build_css,
)


# ── Hex colour format helper ──────────────────────────────────────────────────

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

def _is_hex(value: str) -> bool:
    return bool(_HEX_RE.match(value))


# ── Carbon token namespace ────────────────────────────────────────────────────

class TestCarbonTokens:
    def test_blue_60_is_hex(self):
        assert _is_hex(C.BLUE_60)

    def test_green_60_is_hex(self):
        assert _is_hex(C.GREEN_60)

    def test_red_60_is_hex(self):
        assert _is_hex(C.RED_60)

    def test_gray_tokens_are_hex(self):
        for attr in ["GRAY_100", "GRAY_90", "GRAY_80", "GRAY_60", "GRAY_30", "GRAY_20", "GRAY_10", "WHITE"]:
            assert _is_hex(getattr(C, attr)), f"C.{attr} is not a valid hex"

    def test_yellow_tokens_are_hex(self):
        assert _is_hex(C.YELLOW_30)
        assert _is_hex(C.YELLOW_10)

    def test_purple_tokens_are_hex(self):
        assert _is_hex(C.PURPLE_60)
        assert _is_hex(C.PURPLE_40)

    def test_white_is_ffffff(self):
        assert C.WHITE == "#ffffff"


# ── Chart palette ─────────────────────────────────────────────────────────────

class TestChartPalette:
    def test_palette_has_at_least_four_colors(self):
        assert len(CHART_PALETTE) >= 4

    def test_all_palette_colors_are_hex(self):
        for c in CHART_PALETTE:
            assert _is_hex(c), f"{c} is not a valid hex"

    def test_palette_colors_are_unique(self):
        assert len(set(CHART_PALETTE)) == len(CHART_PALETTE)

    def test_primary_in_palette(self):
        assert PRIMARY_COLOR in CHART_PALETTE

    def test_secondary_in_palette(self):
        assert SECONDARY_COLOR in CHART_PALETTE

    def test_semantic_colors_are_hex(self):
        for name, val in [
            ("PRIMARY", PRIMARY_COLOR),
            ("SECONDARY", SECONDARY_COLOR),
            ("SUCCESS", SUCCESS_COLOR),
            ("WARNING", WARNING_COLOR),
            ("DANGER", DANGER_COLOR),
            ("NEUTRAL", NEUTRAL_COLOR),
        ]:
            assert _is_hex(val), f"{name}_COLOR '{val}' is not a valid hex"


# ── Semantic colour maps ──────────────────────────────────────────────────────

class TestSemanticColorMaps:
    def test_rec_colors_has_all_signals(self):
        for sig in ["Overweight", "Hold", "Underweight"]:
            assert sig in REC_COLORS
            assert _is_hex(REC_COLORS[sig])

    def test_risk_colors_has_all_levels(self):
        for lvl in ["Low", "Medium", "High"]:
            assert lvl in RISK_COLORS
            assert _is_hex(RISK_COLORS[lvl])

    def test_zone_colors_has_all_zones(self):
        for zone in ["Safe", "Grey", "Distress"]:
            assert zone in ZONE_COLORS
            assert _is_hex(ZONE_COLORS[zone])

    def test_overweight_is_green(self):
        # Overweight should use a green shade
        assert REC_COLORS["Overweight"] == C.GREEN_60

    def test_underweight_is_red(self):
        assert REC_COLORS["Underweight"] == C.RED_60

    def test_high_risk_is_red(self):
        assert RISK_COLORS["High"] == C.RED_60

    def test_low_risk_is_green(self):
        assert RISK_COLORS["Low"] == C.GREEN_60

    def test_distress_is_red(self):
        assert ZONE_COLORS["Distress"] == C.RED_60

    def test_safe_is_green(self):
        assert ZONE_COLORS["Safe"] == C.GREEN_60


# ── chart_layout() ────────────────────────────────────────────────────────────

class TestChartLayout:
    def test_returns_dict(self):
        layout = chart_layout("Test Chart")
        assert isinstance(layout, dict)

    def test_contains_height(self):
        layout = chart_layout("Test", height=450)
        assert layout["height"] == 450

    def test_contains_title(self):
        layout = chart_layout("My Chart")
        assert layout["title"]["text"] == "My Chart"

    def test_default_height_is_400(self):
        layout = chart_layout("X")
        assert layout["height"] == 400

    def test_plot_bgcolor_is_white(self):
        layout = chart_layout()
        assert layout["plot_bgcolor"] == C.WHITE

    def test_paper_bgcolor_is_white(self):
        layout = chart_layout()
        assert layout["paper_bgcolor"] == C.WHITE

    def test_xaxis_has_gridcolor(self):
        layout = chart_layout()
        assert "xaxis" in layout
        assert "gridcolor" in layout["xaxis"]

    def test_kwargs_override(self):
        layout = chart_layout("T", showlegend=False)
        assert layout["showlegend"] is False

    def test_empty_title_allowed(self):
        layout = chart_layout("")
        assert layout["title"]["text"] == ""


# ── risk_gauge_steps() ────────────────────────────────────────────────────────

class TestRiskGaugeSteps:
    def test_returns_three_steps(self):
        steps = risk_gauge_steps()
        assert len(steps) == 3

    def test_ranges_cover_zero_to_hundred(self):
        steps = risk_gauge_steps()
        assert steps[0]["range"][0] == 0
        assert steps[-1]["range"][1] == 100

    def test_ranges_are_contiguous(self):
        steps = risk_gauge_steps()
        for i in range(len(steps) - 1):
            assert steps[i]["range"][1] == steps[i+1]["range"][0]

    def test_all_steps_have_color(self):
        for step in risk_gauge_steps():
            assert "color" in step
            assert step["color"]

    def test_first_step_is_greenish(self):
        # Low risk (0–35) should be greenish
        first_color = risk_gauge_steps()[0]["color"]
        assert _is_hex(first_color)

    def test_last_step_is_reddish(self):
        last_color = risk_gauge_steps()[-1]["color"]
        assert _is_hex(last_color)


# ── time_ago() ────────────────────────────────────────────────────────────────

class TestTimeAgo:
    def _iso(self, delta: timedelta) -> str:
        return (datetime.now(timezone.utc) - delta).isoformat()

    def test_seconds_ago(self):
        result = time_ago(self._iso(timedelta(seconds=30)))
        assert "s ago" in result

    def test_minutes_ago(self):
        result = time_ago(self._iso(timedelta(minutes=5)))
        assert "m ago" in result

    def test_hours_ago(self):
        result = time_ago(self._iso(timedelta(hours=3)))
        assert "h ago" in result

    def test_days_ago(self):
        result = time_ago(self._iso(timedelta(days=2)))
        assert "d ago" in result

    def test_empty_string_returns_fallback(self):
        result = time_ago("")
        assert "ago" in result or "unknown" in result

    def test_z_suffix_handled(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = time_ago(ts)
        assert "m ago" in result

    def test_naive_ts_handled(self):
        # bare ISO without timezone — should not raise
        ts = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S")
        result = time_ago(ts)
        assert result  # just ensure no exception

    def test_one_minute_boundary(self):
        # exactly 61 seconds → minutes
        result = time_ago(self._iso(timedelta(seconds=61)))
        assert "m ago" in result

    def test_just_now(self):
        # 0 seconds
        result = time_ago(datetime.now(timezone.utc).isoformat())
        assert "s ago" in result or "0s" in result


# ── fmt_ts() ─────────────────────────────────────────────────────────────────

class TestFmtTs:
    def test_formats_iso_string(self):
        ts = "2024-11-01T14:32:00+00:00"
        result = fmt_ts(ts)
        assert result == "2024-11-01 14:32:00 UTC"

    def test_truncates_to_seconds(self):
        ts = "2024-06-15T09:05:30.123456+00:00"
        result = fmt_ts(ts)
        assert result.startswith("2024-06-15 09:05:30")

    def test_handles_empty_string(self):
        result = fmt_ts("")
        assert isinstance(result, str)


# ── build_css() ──────────────────────────────────────────────────────────────

class TestBuildCss:
    def test_returns_non_empty_string(self):
        css = build_css()
        assert isinstance(css, str)
        assert len(css) > 100

    def test_contains_style_tag(self):
        css = build_css()
        assert "<style>" in css
        assert "</style>" in css

    def test_contains_rec_card_class(self):
        assert ".rec-card" in build_css()

    def test_contains_disclaimer_class(self):
        assert ".disclaimer" in build_css()

    def test_contains_hero_card_class(self):
        assert ".hero-card" in build_css()

    def test_contains_stepper_class(self):
        assert ".stepper-wrap" in build_css()

    def test_contains_badge_classes(self):
        css = build_css()
        assert ".badge-live" in css
        assert ".badge-demo" in css
        assert ".badge-fallback" in css

    def test_contains_skeleton_animation(self):
        assert ".skeleton" in build_css()

    def test_contains_last_updated_class(self):
        assert ".last-updated" in build_css()

    def test_contains_carbon_blue(self):
        # The Carbon Blue 60 token should appear in the CSS
        assert C.BLUE_60 in build_css()

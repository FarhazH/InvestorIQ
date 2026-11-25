"""
InvestIQ — Streamlit Dashboard (Phase 6: IBM Carbon UI/UX Polish)
Tabs: Summary | KPIs | Ratios | Price | Sentiment | Risks | Advanced |
      Filings | Chat | Watchlist | Portfolio | Report
"""
from __future__ import annotations

import sys
import os
import json
import time
import threading
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Phase 6: import theme tokens ──────────────────────────────────────────────
from dashboard.ui_theme import (
    C, CHART_PALETTE, PRIMARY_COLOR, SECONDARY_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, NEUTRAL_COLOR,
    REC_COLORS, RISK_COLORS, ZONE_COLORS,
    chart_layout, risk_gauge_steps, time_ago, fmt_ts, build_css,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InvestIQ — AI Investment Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Carbon CSS ─────────────────────────────────────────────────────────
st.markdown(build_css(), unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "running" not in st.session_state:
    st.session_state.running = False
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []          # list[WatchlistEntry dicts]
if "portfolio_results" not in st.session_state:
    st.session_state.portfolio_results = {}  # ticker → PipelineState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_num(val: Any, suffix: str = "", billions: bool = False) -> str:
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if billions and abs(f) >= 1e9:
            return f"${f/1e9:.2f}B{suffix}"
        if abs(f) >= 1e6:
            return f"${f/1e6:.1f}M{suffix}"
        return f"{f:.2f}{suffix}"
    except Exception:
        return str(val)


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f}%"
    except Exception:
        return str(val)


def _status_badge(status: str) -> str:
    badges = {
        "done": "✅",
        "running": "🔄",
        "failed": "❌",
        "warning": "⚠️",
        "pending": "⏳",
        "skipped": "⏭️",
    }
    return badges.get(status, "❓")


def _run_pipeline_sync(
    ticker: str,
    competitors: list[str],
    investor_profile: "Any | None" = None,
) -> None:
    """Run pipeline synchronously in Streamlit (runs in same thread for simplicity)."""
    from pipeline.orchestrator import PipelineOrchestrator

    def on_step(state):
        st.session_state.pipeline_state = state

    orch = PipelineOrchestrator(step_callback=on_step)
    state = orch.run(ticker, competitors)
    # Attach investor profile (Phase 5)
    if investor_profile is not None:
        state.investor_profile = investor_profile
    st.session_state.pipeline_state = state
    st.session_state.running = False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
    st.title("📈 InvestIQ")
    st.caption("AI Investment Analyst · Powered by IBM Granite")
    st.divider()

    ticker_input = st.text_input("Primary Ticker", value="AAPL", placeholder="e.g. AAPL").upper().strip()
    competitors_raw = st.text_input(
        "Competitors (comma-separated, max 3)",
        value="MSFT, GOOGL, AMZN",
        placeholder="e.g. MSFT, GOOGL",
    )
    competitors = [c.strip().upper() for c in competitors_raw.split(",") if c.strip()][:3]

    # ── Investor Profile (Phase 5) ────────────────────────────────────────────
    st.divider()
    with st.expander("👤 Investor Profile", expanded=False):
        sb_risk = st.selectbox(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            index=1,
            key="sb_risk",
        )
        sb_horizon = st.selectbox(
            "Investment Horizon",
            options=["Short-term (< 1 year)", "Medium-term (1–3 years)", "Long-term (3+ years)"],
            index=1,
            key="sb_horizon",
        )
        sb_intent = st.selectbox(
            "Intent",
            options=["New position", "Adding to existing position", "Portfolio review", "Research only"],
            index=3,
            key="sb_intent",
        )
        sb_port_size = st.selectbox(
            "Portfolio Size",
            options=["< $10K", "$10K – $100K", "$100K – $1M", "$1M+"],
            index=1,
            key="sb_port_size",
        )
        sb_esg = st.selectbox(
            "ESG Preference",
            options=["No preference", "ESG-aware", "ESG-only"],
            index=0,
            key="sb_esg",
        )
        sb_sector = st.text_input(
            "Sector Notes (optional)",
            placeholder="e.g. Avoid tobacco, focus on tech",
            key="sb_sector",
            max_chars=200,
        )

    # Build investor profile object from sidebar values
    from models.investor_profile import InvestorProfile as _InvestorProfile
    _investor_profile = _InvestorProfile(
        risk_tolerance=sb_risk,
        horizon=sb_horizon,
        intent=sb_intent,
        portfolio_size=sb_port_size,
        esg_preference=sb_esg,
        sector_notes=sb_sector,
    )

    st.divider()
    demo_toggle = st.toggle("Demo Mode (no API keys needed)", value=True)
    if demo_toggle:
        os.environ["DEMO_MODE"] = "true"
    else:
        os.environ["DEMO_MODE"] = "false"

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=st.session_state.running)

    if run_btn and ticker_input:
        st.session_state.running = True
        st.session_state.pipeline_state = None
        st.session_state.analyst_reviews = []   # reset reviews for new run
        # Store profile in session so trigger block can access it
        st.session_state._investor_profile = _investor_profile

    # Save to Watchlist button (Phase 5)
    _ps = st.session_state.get("pipeline_state")
    if _ps is not None and _ps.recommendation:
        if st.button("⭐ Save to Watchlist", use_container_width=True):
            from models.investor_profile import WatchlistEntry as _WLE
            _ws_entry = _WLE(
                ticker=_ps.ticker,
                run_id=_ps.run_id,
                recommendation=_ps.recommendation.recommendation,
                confidence=_ps.recommendation.confidence,
                composite_score=(
                    _ps.advanced.weighted_score.composite_score
                    if _ps.advanced and _ps.advanced.weighted_score else None
                ),
                risk_level=_ps.risk.overall_risk_level if _ps.risk else "Medium",
                generated_at=_ps.created_at[:19],
                competitors=_ps.competitors,
                granite_mode=_ps.granite_mode,
            )
            # Avoid duplicates by run_id
            existing_ids = {e.get("run_id") for e in st.session_state.watchlist}
            if _ps.run_id not in existing_ids:
                st.session_state.watchlist.append(_ws_entry.model_dump())
                st.success(f"⭐ {_ps.ticker} added to watchlist!")
            else:
                st.info("Already in watchlist.")

    # Show Granite mode badge if a run has completed
    if _ps is not None:
        gmode = getattr(_ps, "granite_mode", "demo")
        badge_map = {
            "live":     ('<span class="badge-live">🤖 Granite (live)</span>', ""),
            "demo":     ('<span class="badge-demo">🗂️ Demo data</span>', ""),
            "fallback": ('<span class="badge-fallback">⚙️ Rule-based fallback</span>', ""),
        }
        badge_html, _ = badge_map.get(gmode, badge_map["demo"])
        st.markdown(f"**Powered by:** {badge_html}", unsafe_allow_html=True)

    st.divider()
    st.caption("⚠️ NOT financial advice. For educational purposes only.")
    st.caption("Built with IBM Bob · IBM Granite · watsonx.ai")


# ── Trigger pipeline ──────────────────────────────────────────────────────────

if st.session_state.running and st.session_state.pipeline_state is None:
    _profile = st.session_state.get("_investor_profile", None)
    with st.spinner("Initialising pipeline..."):
        _run_pipeline_sync(ticker_input, competitors, investor_profile=_profile)
    st.rerun()


# ── Pipeline progress tracker ─────────────────────────────────────────────────

state = st.session_state.pipeline_state

STEP_LABELS = {
    "data_gathering":    "1. Data Gathering",
    "validation":        "2. Validation",
    "kpi_extraction":    "3. KPI Extraction",
    "ratio_analysis":    "4. Ratio Analysis",
    "advanced_analysis": "5. Advanced Analysis",
    "benchmarking":      "6. Benchmarking",
    "sentiment":         "7. Sentiment (Granite)",
    "risk_trend":        "8. Risk & Trend",
    "recommendation":    "9. Rec. (Granite)",
    "report":           "10. Report",
    "rag_ingestion":    "11. RAG (Filings)",
}

# ── Phase 6: Carbon stepper ───────────────────────────────────────────────────

def _render_stepper(steps: list) -> None:
    """Render a horizontal IBM Carbon-style pipeline stepper."""
    ICONS = {"done": "✓", "running": "⟳", "failed": "✕", "warning": "!", "pending": "○", "skipped": "⤳"}
    nodes_html = []
    for i, step in enumerate(steps):
        st_name = step.status if isinstance(step.status, str) else step.status.value
        label   = STEP_LABELS.get(step.name, step.name)
        icon    = ICONS.get(st_name, "○")
        dur     = f'<div class="step-dur">({step.duration_s:.1f}s)</div>' if step.duration_s else ""
        connector = ""
        if i > 0:
            conn_cls = "step-connector-done" if steps[i-1].status in ("done", "warning") else ""
            connector = f'<div class="step-connector {conn_cls}"></div>'
        nodes_html.append(
            f'{connector}'
            f'<div class="step-node">'
            f'  <div class="step-circle step-circle-{st_name}">{icon}</div>'
            f'  <div class="step-label step-label-{st_name}">{label}</div>'
            f'  {dur}'
            f'</div>'
        )
    stepper = (
        '<div class="stepper-wrap">'
        + "".join(nodes_html)
        + '</div>'
    )
    st.markdown(stepper, unsafe_allow_html=True)


if state is not None:
    st.markdown('<div class="section-header">Pipeline Progress</div>', unsafe_allow_html=True)
    _render_stepper(state.steps)
    if state.errors:
        with st.expander("⚠️ Pipeline Errors"):
            for err in state.errors:
                agent_name = err["agent"] if isinstance(err, dict) else getattr(err, "agent", "?")
                err_msg    = err["error"] if isinstance(err, dict) else getattr(err, "error", str(err))
                st.error(f"[{agent_name}] {err_msg}")
    st.divider()


# ── Main tabs ─────────────────────────────────────────────────────────────────

if state is None:
    # Phase 6: Carbon hero welcome card
    st.markdown("""
<div class="hero-card">
  <div class="hero-title">📈 InvestIQ — AI Investment Analyst</div>
  <div class="hero-subtitle">Powered by IBM Granite · watsonx.ai · Sequential 11-Agent Pipeline</div>
  <div class="feature-grid">
    <div class="feature-item"><span class="feature-icon">🤖</span><div><b>IBM Granite LLM</b><br>Sentiment scoring &amp; grounded investment recommendations via watsonx.ai</div></div>
    <div class="feature-item"><span class="feature-icon">📊</span><div><b>11-Stage Pipeline</b><br>Data → Validation → KPIs → Ratios → Advanced → Benchmarking → Sentiment → Risk → Rec → Report → RAG</div></div>
    <div class="feature-item"><span class="feature-icon">🔬</span><div><b>Advanced Analysis</b><br>DCF-lite fair value, Altman Z-score, weighted composite scoring, MA backtest</div></div>
    <div class="feature-item"><span class="feature-icon">📑</span><div><b>SEC Filing RAG</b><br>Semantic search over 10-K filings with grounded Q&amp;A and risk factor extraction</div></div>
    <div class="feature-item"><span class="feature-icon">🔎</span><div><b>Explainability</b><br>Factor contribution waterfall, explanation trace, grounding check on every AI claim</div></div>
    <div class="feature-item"><span class="feature-icon">👤</span><div><b>Investor Profile</b><br>Personalises Granite recommendations to your risk tolerance, horizon &amp; intent</div></div>
    <div class="feature-item"><span class="feature-icon">💼</span><div><b>Portfolio Mode</b><br>Multi-ticker weighted portfolio with blended composite signal &amp; risk score</div></div>
    <div class="feature-item"><span class="feature-icon">📄</span><div><b>PDF &amp; JSON Reports</b><br>Download full structured reports with investor profile, audit trail &amp; colour-coded risk tables</div></div>
  </div>
</div>
<div class="disclaimer" style="margin-top:0;">
⚠️ <b>Disclaimer:</b> InvestIQ is an AI-powered educational tool and is <b>NOT financial advice</b>.
Always consult a licensed financial advisor before making investment decisions.
</div>
    """, unsafe_allow_html=True)
else:
    tabs = st.tabs([
        "📋 Summary",
        "💰 Financial KPIs",
        "📊 Ratios & Peers",
        "📈 Price Trend",
        "📰 Sentiment",
        "⚠️ Risks",
        "🔬 Advanced",
        "📑 Filings & RAG",
        "💬 Ask the Analyst",
        "⭐ Watchlist",
        "💼 Portfolio",
        "📄 Full Report",
    ])

    # ── TAB 1: Summary ────────────────────────────────────────────────────────
    with tabs[0]:
        rec = state.recommendation
        risk = state.risk

        # ── Top banner: data quality + granite mode + investor profile ────────
        banner_cols = st.columns([3, 1, 1])
        with banner_cols[0]:
            if state.validation:
                qs = state.validation.primary.quality_score
                q_cls = "quality-high" if qs >= 0.8 else ("quality-med" if qs >= 0.6 else "quality-low")
                q_flags = len(state.validation.primary.flags)
                st.markdown(
                    f'📊 **Data Quality:** <span class="{q_cls}">{qs:.0%}</span>'
                    f' &nbsp; ({q_flags} flag{"s" if q_flags != 1 else ""})',
                    unsafe_allow_html=True,
                )
            # Investor profile pill (Phase 5)
            if state.investor_profile:
                p = state.investor_profile
                risk_color = {
                    "Conservative": C.BLUE_60,
                    "Moderate": WARNING_COLOR,
                    "Aggressive": DANGER_COLOR,
                }.get(p.risk_tolerance, C.GRAY_60)
                st.markdown(
                    f'👤 <span style="background:{C.GRAY_10};border:1px solid {C.GRAY_20};padding:2px 8px;'
                    f'border-radius:10px;font-size:0.82em;">'
                    f'<span style="color:{risk_color};font-weight:600;">{p.risk_tolerance}</span>'
                    f' · {p.horizon.split("(")[0].strip()} · {p.intent}</span>',
                    unsafe_allow_html=True,
                )
            # Phase 6: last-updated chip
            ts_ago = time_ago(state.created_at)
            st.markdown(
                f'<span class="last-updated">🕐 Last analysed: {ts_ago}</span>',
                unsafe_allow_html=True,
            )
        with banner_cols[1]:
            gmode = getattr(state, "granite_mode", "demo")
            badge_map2 = {
                "live":     '<span class="badge-live">🤖 Granite (live)</span>',
                "demo":     '<span class="badge-demo">🗂️ Demo data</span>',
                "fallback": '<span class="badge-fallback">⚙️ Fallback</span>',
            }
            st.markdown(badge_map2.get(gmode, badge_map2["demo"]), unsafe_allow_html=True)
        with banner_cols[2]:
            if state.validation and state.validation.primary.flags:
                with st.expander("⚠️ Quality Flags"):
                    for fl in state.validation.primary.flags:
                        st.caption(f"[{fl.severity.upper()}] {fl.field}: {fl.issue}")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Investment Recommendation: {state.ticker}")
            if rec:
                css_class = {
                    "Overweight": "overweight",
                    "Hold": "hold",
                    "Underweight": "underweight",
                }.get(rec.recommendation, "hold")
                # Granite mode badge on the card
                gmode = getattr(rec, "granite_mode", "demo")
                mode_badge = {
                    "live":     '&nbsp;<span class="badge-live" style="font-size:0.65em">🤖 live</span>',
                    "demo":     '&nbsp;<span class="badge-demo" style="font-size:0.65em">🗂️ demo</span>',
                    "fallback": '&nbsp;<span class="badge-fallback" style="font-size:0.65em">⚙️ fallback</span>',
                }.get(gmode, "")
                st.markdown(
                    f'<div class="rec-card {css_class}">{rec.recommendation} '
                    f'| Confidence: {rec.confidence*100:.0f}%{mode_badge}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Horizon:** {rec.target_horizon}")
                st.markdown("**Reasons:**")
                for r in rec.reasons:
                    metric_str = f" `{r.metric}={r.value}`" if r.metric else ""
                    st.markdown(f"- {r.point}{metric_str}")
                if rec.forward_outlook:
                    st.info(f"🔮 **Outlook:** {rec.forward_outlook}")
                if rec.key_risks:
                    st.warning(f"🚨 **Key Risks:** {', '.join(rec.key_risks)}")

                # Grounding check panel
                if rec.grounding_flags:
                    n_ok = sum(1 for f in rec.grounding_flags if f.match)
                    n_total = len(rec.grounding_flags)
                    with st.expander(
                        f"🔍 Grounding Check: {n_ok}/{n_total} metrics verified "
                        f"({'✅ All OK' if n_ok == n_total else '⚠️ Mismatches found'})"
                    ):
                        import pandas as pd
                        gdf = pd.DataFrame([
                            {
                                "Metric": f.metric,
                                "Cited": f.cited_value,
                                "Computed": f.computed_value or "—",
                                "Match": "✅" if f.match else ("⚠️" if f.computed_value else "❓"),
                            }
                            for f in rec.grounding_flags
                        ])
                        st.dataframe(gdf, use_container_width=True, hide_index=True)
            else:
                st.warning("Recommendation not yet available.")

        with col2:
            st.subheader("Risk Gauge")
            if risk:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk.overall_risk_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": f"Risk: {risk.overall_risk_level}",
                           "font": {"size": 13, "color": C.GRAY_80}},
                    number={"font": {"size": 28, "color": C.GRAY_100}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": C.GRAY_60},
                        "bar": {"color": PRIMARY_COLOR},
                        "steps": risk_gauge_steps(),
                        "threshold": {
                            "line": {"color": DANGER_COLOR, "width": 3},
                            "thickness": 0.75,
                            "value": risk.overall_risk_score,
                        },
                    },
                ))
                fig.update_layout(height=250, margin=dict(t=40, b=0, l=10, r=10),
                                  paper_bgcolor=C.WHITE, plot_bgcolor=C.WHITE)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk data not available yet.")

        # Quick KPI snapshot with YoY deltas (Phase 6)
        kpis = state.kpis
        if kpis:
            st.divider()
            st.markdown('<div class="section-header">Key Metrics Snapshot</div>', unsafe_allow_html=True)
            pk = kpis.primary
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Revenue", _fmt_num(pk.revenue.current, billions=True))
            m2.metric("Net Income", _fmt_num(pk.net_income.current, billions=True),
                      delta=_fmt_pct(pk.net_income.yoy_change_pct) if pk.net_income.yoy_change_pct else None)
            m3.metric("Net Margin", _fmt_pct(pk.net_margin and pk.net_margin * 100))
            m4.metric("Revenue Growth", _fmt_pct(pk.revenue_growth_yoy),
                      delta=_fmt_pct(pk.revenue_growth_yoy) if pk.revenue_growth_yoy else None)
            m5.metric("EPS (TTM)", _fmt_num(pk.eps_ttm),
                      delta=_fmt_pct(pk.eps_yoy_change_pct) if pk.eps_yoy_change_pct else None)
            m6.metric("Free Cash Flow", _fmt_num(pk.free_cash_flow, billions=True))

        # ── PHASE 4: Factor Contribution Waterfall ────────────────────────────
        if state.advanced and state.advanced.weighted_score:
            st.divider()
            st.subheader("🔬 Explainability: What's Driving the Signal?")
            ws = state.advanced.weighted_score

            try:
                from models.analyst_review import (
                    build_factor_contributions,
                    build_confidence_breakdown,
                    build_explanation_trace,
                )
                from agents.recommendation_agent import _build_computed_lookup

                factors = build_factor_contributions(ws)
                cbd = build_confidence_breakdown(ws)

                # Factor contribution waterfall chart
                col_wf, col_br = st.columns([3, 2])
                with col_wf:
                    st.markdown("**Factor Contribution to Composite Score** (relative to 50 baseline)")
                    st.caption("Each bar = how much a dimension shifts the score above/below neutral (50).")
                    factor_names = [f.factor for f in factors]
                    contribs = [f.contribution for f in factors]
                    raw_scores = [f.raw_score for f in factors]
                    colors = [
                        "#28a745" if c > 0.5 else ("#dc3545" if c < -0.5 else "#6c757d")
                        for c in contribs
                    ]
                    wf_fig = go.Figure()
                    wf_fig.add_trace(go.Bar(
                        x=factor_names,
                        y=contribs,
                        marker_color=colors,
                        text=[f"{c:+.1f}" for c in contribs],
                        textposition="outside",
                        name="Contribution",
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Contribution: %{y:+.2f}<br>"
                            "Raw score: " + "<br>".join(
                                f"{n}: {s:.1f}" for n, s in zip(factor_names, raw_scores)
                            )
                            + "<extra></extra>"
                        ),
                    ))
                    wf_fig.add_hline(y=0, line_dash="dash", line_color="#6c757d", line_width=1)
                    wf_fig.update_layout(
                        height=320,
                        margin=dict(t=30, b=20, l=10, r=10),
                        yaxis_title="Contribution to score",
                        showlegend=False,
                        plot_bgcolor="white",
                        yaxis=dict(gridcolor="#e5e7eb"),
                    )
                    st.plotly_chart(wf_fig, use_container_width=True)
                    st.caption(f"💡 {cbd.explanation}")

                with col_br:
                    st.markdown("**Confidence Breakdown by Dimension**")
                    st.caption(f"Composite: **{ws.composite_score:.1f}/100** → Signal: **{ws.signal}** ({ws.signal_confidence:.0%} confidence)")
                    # Horizontal bar chart: each dimension's raw_score vs 50 baseline
                    br_fig = go.Figure()
                    br_fig.add_trace(go.Bar(
                        y=[f.factor for f in factors],
                        x=[f.raw_score for f in factors],
                        orientation="h",
                        marker_color=[
                            "#28a745" if f.raw_score >= 60 else
                            ("#ffc107" if f.raw_score >= 40 else "#dc3545")
                            for f in factors
                        ],
                        text=[f"{f.raw_score:.0f}" for f in factors],
                        textposition="auto",
                        name="Score",
                    ))
                    br_fig.add_vline(x=50, line_dash="dash", line_color="#6c757d", line_width=1.5,
                                     annotation_text="Neutral (50)", annotation_position="top")
                    br_fig.update_layout(
                        height=320,
                        margin=dict(t=30, b=20, l=10, r=10),
                        xaxis=dict(range=[0, 100], title="Score (0–100)"),
                        showlegend=False,
                        plot_bgcolor="white",
                        xaxis_gridcolor="#e5e7eb",
                    )
                    st.plotly_chart(br_fig, use_container_width=True)

            except Exception as _exp_err:
                st.warning(f"Explainability charts unavailable: {_exp_err}")

        # ── PHASE 4: Explanation Trace ────────────────────────────────────────
        if state.recommendation and state.recommendation.reasons:
            st.divider()
            st.markdown("#### 🔎 Explanation Trace — AI Reasons vs. Computed Metrics")
            st.caption(
                "Each row shows an AI-cited reason alongside the pipeline-computed metric value, "
                "so you can verify the AI's claims against real data."
            )
            try:
                from models.analyst_review import build_explanation_trace
                from agents.recommendation_agent import _build_computed_lookup
                trace = build_explanation_trace(state.recommendation, _build_computed_lookup(state))
                if trace:
                    trace_rows = []
                    for e in trace:
                        status = "✅ Verified" if e.grounded else ("⚠️ Mismatch" if e.computed_value else "❓ Unverifiable")
                        trace_rows.append({
                            "Reason": e.reason,
                            "Metric": e.metric or "—",
                            "AI Cited": e.cited_value or "—",
                            "Pipeline Computed": e.computed_value or "—",
                            "Status": status,
                            "Source": e.source,
                        })
                    trace_df = pd.DataFrame(trace_rows)
                    st.dataframe(trace_df, use_container_width=True, hide_index=True)
            except Exception as _trace_err:
                st.warning(f"Explanation trace unavailable: {_trace_err}")

        # ── PHASE 4: Analyst Review Panel ─────────────────────────────────────
        st.divider()
        st.subheader("👩‍💼 Analyst Review & Override")
        st.caption(
            "Override the AI recommendation, add notes, and flag concerns. "
            "Every review is logged in the audit trail below."
        )

        # Initialise review state
        if "analyst_reviews" not in st.session_state:
            st.session_state.analyst_reviews = []

        with st.expander("✍️ Submit Analyst Review", expanded=False):
            ar_col1, ar_col2 = st.columns([2, 1])
            with ar_col1:
                ar_name = st.text_input("Your Name / ID", value="Analyst", key="ar_name")
                ar_notes = st.text_area(
                    "Notes / Rationale",
                    placeholder="Enter your analysis notes, caveats, or additional context here…",
                    key="ar_notes",
                    max_chars=2000,
                    height=100,
                )
                ar_flags = st.multiselect(
                    "Raise Flags",
                    options=[
                        "Data quality concern",
                        "Stale data",
                        "Sentiment appears skewed",
                        "DCF assumptions too aggressive",
                        "Risk score underestimated",
                        "Competitor selection insufficient",
                        "Regulatory risk not reflected",
                        "AI reasoning appears hallucinated",
                    ],
                    key="ar_flags",
                )
            with ar_col2:
                ai_rec = state.recommendation.recommendation if state.recommendation else "Hold"
                ar_override = st.selectbox(
                    "Override Recommendation",
                    options=["(Keep AI recommendation)", "Overweight", "Hold", "Underweight"],
                    index=0,
                    key="ar_override",
                )
                ar_conf = st.slider(
                    "Override Confidence",
                    min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                    key="ar_conf",
                    disabled=(ar_override == "(Keep AI recommendation)"),
                )
                st.markdown(f"**Current AI Rec:** `{ai_rec}`")
                if ar_override != "(Keep AI recommendation)":
                    css_map = {"Overweight": "overweight", "Hold": "hold", "Underweight": "underweight"}
                    _ar_css = css_map.get(ar_override, "hold")
                    st.markdown(
                        f'<div class="rec-card {_ar_css}" '
                        f'style="font-size:0.9em;padding:10px;">Override: {ar_override} '
                        f'({ar_conf:.0%})</div>',
                        unsafe_allow_html=True,
                    )

            if st.button("💾 Submit Review", key="ar_submit", type="primary"):
                if not ar_notes.strip() and ar_override == "(Keep AI recommendation)" and not ar_flags:
                    st.warning("Please add notes, an override, or at least one flag before submitting.")
                else:
                    from models.analyst_review import AnalystReview
                    review = AnalystReview(
                        run_id=state.run_id,
                        ticker=state.ticker,
                        override_recommendation=(
                            "" if ar_override == "(Keep AI recommendation)" else ar_override
                        ),
                        override_confidence=ar_conf if ar_override != "(Keep AI recommendation)" else 0.0,
                        notes=ar_notes.strip(),
                        analyst_name=ar_name.strip() or "Anonymous",
                        flags=ar_flags,
                    )
                    st.session_state.analyst_reviews.append(review.model_dump())
                    st.session_state.pipeline_state.analyst_reviews.append(review)
                    st.success(f"✅ Review submitted by {review.analyst_name} at {review.reviewed_at}")

        # ── Audit Trail ───────────────────────────────────────────────────────
        reviews = st.session_state.get("analyst_reviews", [])
        if reviews:
            st.markdown(f"#### 📋 Audit Trail ({len(reviews)} review{'s' if len(reviews) != 1 else ''})")
            for i, rv in enumerate(reviews):
                override_txt = (
                    f"**Override:** {rv.get('override_recommendation')} ({rv.get('override_confidence', 0):.0%})"
                    if rv.get("override_recommendation")
                    else "No override (AI recommendation accepted)"
                )
                flags_txt = ", ".join(rv.get("flags", [])) or "None"
                with st.expander(
                    f"Review #{i+1} — {rv.get('analyst_name', '?')} · "
                    f"{rv.get('reviewed_at', '?')[:19].replace('T', ' ')} UTC",
                    expanded=(i == len(reviews) - 1),
                ):
                    st.markdown(f"**Ticker:** `{rv.get('ticker')}` | **Run ID:** `{rv.get('run_id', '?')[:8]}…`")
                    st.markdown(override_txt)
                    if rv.get("notes"):
                        st.markdown(f"**Notes:** {rv['notes']}")
                    st.markdown(f"**Flags:** {flags_txt}")
        else:
            st.caption("No analyst reviews yet. Submit a review above to start the audit trail.")

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This report is generated by an AI model (IBM Granite) and is
        <b>NOT financial advice</b>. Always consult a licensed financial advisor before making
        investment decisions.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Financial KPIs ─────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Financial KPIs")
        if state.kpis:
            pk = state.kpis.primary
            kpi_data = {
                "Metric": [
                    "Revenue", "Revenue YoY %", "Net Income", "Net Income YoY %",
                    "EBITDA", "EPS (TTM)", "EPS Growth %", "Free Cash Flow",
                    "FCF YoY %", "Gross Margin %", "Operating Margin %", "Net Margin %",
                    "Total Debt", "Total Cash", "Net Debt",
                ],
                state.ticker: [
                    _fmt_num(pk.revenue.current, billions=True),
                    _fmt_pct(pk.revenue_growth_yoy),
                    _fmt_num(pk.net_income.current, billions=True),
                    _fmt_pct(pk.net_income.yoy_change_pct),
                    _fmt_num(pk.ebitda, billions=True),
                    _fmt_num(pk.eps_ttm),
                    _fmt_pct(pk.eps_yoy_change_pct),
                    _fmt_num(pk.free_cash_flow, billions=True),
                    _fmt_pct(pk.fcf_yoy_change_pct),
                    _fmt_pct(pk.gross_margin and pk.gross_margin * 100),
                    _fmt_pct(pk.operating_margin and pk.operating_margin * 100),
                    _fmt_pct(pk.net_margin and pk.net_margin * 100),
                    _fmt_num(pk.total_debt, billions=True),
                    _fmt_num(pk.total_cash, billions=True),
                    _fmt_num(pk.net_debt, billions=True),
                ],
            }
            for ck in state.kpis.competitors:
                kpi_data[ck.ticker] = [
                    _fmt_num(ck.revenue.current, billions=True),
                    _fmt_pct(ck.revenue_growth_yoy),
                    _fmt_num(ck.net_income.current, billions=True),
                    _fmt_pct(ck.net_income.yoy_change_pct),
                    _fmt_num(ck.ebitda, billions=True),
                    _fmt_num(ck.eps_ttm),
                    _fmt_pct(ck.eps_yoy_change_pct),
                    _fmt_num(ck.free_cash_flow, billions=True),
                    _fmt_pct(ck.fcf_yoy_change_pct),
                    _fmt_pct(ck.gross_margin and ck.gross_margin * 100),
                    _fmt_pct(ck.operating_margin and ck.operating_margin * 100),
                    _fmt_pct(ck.net_margin and ck.net_margin * 100),
                    _fmt_num(ck.total_debt, billions=True),
                    _fmt_num(ck.total_cash, billions=True),
                    _fmt_num(ck.net_debt, billions=True),
                ]
            kpi_df = pd.DataFrame(kpi_data).set_index("Metric")
            st.dataframe(kpi_df, use_container_width=True)

            # Margin comparison bar chart
            if state.kpis.competitors:
                st.subheader("Margin Comparison")
                all_kpis = [state.kpis.primary] + state.kpis.competitors
                margin_df = pd.DataFrame([
                    {
                        "Ticker": k.ticker,
                        "Gross Margin %": round((k.gross_margin or 0) * 100, 1),
                        "Operating Margin %": round((k.operating_margin or 0) * 100, 1),
                        "Net Margin %": round((k.net_margin or 0) * 100, 1),
                    }
                    for k in all_kpis
                ])
                fig = px.bar(
                    margin_df.melt(id_vars="Ticker", var_name="Margin Type", value_name="Value"),
                    x="Ticker", y="Value", color="Margin Type", barmode="group",
                    title="Margin Comparison Across Peers",
                    labels={"Value": "Margin (%)"},
                    color_discrete_sequence=CHART_PALETTE,
                )
                fig.update_layout(**chart_layout("Margin Comparison Across Peers", height=380))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("KPI data not yet available.")

    # ── TAB 3: Ratios & Peers ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Ratio Analysis & Peer Benchmarking")
        if state.ratios and state.benchmark:
            # Comparison table
            tbl = state.benchmark.peer_comparison_table
            if tbl:
                df = pd.DataFrame(tbl).set_index("ticker")
                st.dataframe(
                    df.style
                      .highlight_max(color=C.GREEN_10)
                      .highlight_min(color=C.RED_10),
                    use_container_width=True,
                )

            # Radar chart
            st.subheader("Radar: Profitability vs Peers")
            all_ratios = [state.ratios.primary] + state.ratios.competitors
            categories = ["ROE", "ROA", "Net Margin", "Operating Margin", "Gross Margin"]
            fig = go.Figure()
            for i, r in enumerate(all_ratios):
                values = [
                    (r.roe or 0) * 100,
                    (r.roa or 0) * 100,
                    (r.net_margin or 0) * 100,
                    (r.operating_margin or 0) * 100,
                    (r.gross_margin or 0) * 100,
                ]
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=r.ticker,
                    opacity=0.65,
                    line=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)], width=2),
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, gridcolor=C.GRAY_20),
                           bgcolor=C.WHITE),
                title="Profitability Radar",
                height=450,
                paper_bgcolor=C.WHITE,
                font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Over/under-performing
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Over-performing:** {', '.join(state.benchmark.over_performing_metrics) or 'None'}")
            with col2:
                st.error(f"**Under-performing:** {', '.join(state.benchmark.under_performing_metrics) or 'None'}")
            st.info(f"**Overall Peer Rank:** {state.benchmark.overall_rank} of {len(all_ratios)}")
        else:
            st.info("Ratio data not yet available.")

    # ── TAB 4: Price Trend ────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Price History & Technical Analysis")
        if state.raw_data:
            ph = state.raw_data.primary.price_history
            if ph and ph.get("Close"):
                try:
                    close_data = ph["Close"]
                    high_data = ph.get("High", {})
                    low_data = ph.get("Low", {})
                    open_data = ph.get("Open", {})

                    df = pd.DataFrame({
                        "Close": close_data,
                        "High": high_data if high_data else close_data,
                        "Low": low_data if low_data else close_data,
                        "Open": open_data if open_data else close_data,
                    })
                    df.index = pd.to_datetime(df.index, errors="coerce")
                    df = df.sort_index().dropna()

                    # Candlestick + MAs
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"],
                        name=state.ticker, showlegend=True,
                    ))
                    if len(df) >= 50:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df["Close"].rolling(50).mean(),
                            name="MA50", line=dict(color="#f46800", width=1.8),
                        ))
                    if len(df) >= 200:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df["Close"].rolling(200).mean(),
                            name="MA200", line=dict(color=SECONDARY_COLOR, width=1.8),
                        ))

                    # Forecast overlay
                    if state.risk and state.risk.price_forecast:
                        last_date = df.index[-1]
                        fc_dates = pd.date_range(start=last_date, periods=len(state.risk.price_forecast) + 1, freq="W")[1:]
                        fc_vals = [p.value for p in state.risk.price_forecast]
                        fc_lower = [p.lower for p in state.risk.price_forecast if p.lower is not None]
                        fc_upper = [p.upper for p in state.risk.price_forecast if p.upper is not None]
                        fig.add_trace(go.Scatter(
                            x=fc_dates, y=fc_vals,
                            name="Forecast", line=dict(color=SUCCESS_COLOR, dash="dash", width=2),
                        ))
                        if fc_lower and fc_upper:
                            fig.add_trace(go.Scatter(
                                x=list(fc_dates) + list(fc_dates[::-1]),
                                y=fc_upper + fc_lower[::-1],
                                fill="toself", fillcolor="rgba(25,128,56,0.08)",
                                line=dict(color="rgba(255,255,255,0)"),
                                name="Forecast CI",
                            ))

                    fig.update_layout(
                        **chart_layout(f"{state.ticker} — 2-Year Price History", height=500),
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Technical stats
                    if state.ratios:
                        r = state.ratios.primary
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("RSI (14)", _fmt_num(r.rsi_14))
                        c2.metric("Beta", _fmt_num(r.beta))
                        c3.metric("Volatility (Ann.)", _fmt_pct(r.volatility_annualised))
                        c4.metric("Max Drawdown", _fmt_pct(r.max_drawdown))
                        c5.metric("Sharpe Ratio", _fmt_num(r.sharpe_ratio))

                        c1b, c2b, c3b = st.columns(3)
                        c1b.metric("50-DMA", _fmt_num(r.ma_50))
                        c2b.metric("200-DMA", _fmt_num(r.ma_200))
                        c3b.metric("Price vs 200-DMA", _fmt_pct(r.price_vs_ma200_pct))
                except Exception as e:
                    st.warning(f"Chart error: {e}")
            else:
                st.info("Price history not available.")
        else:
            st.info("Data not yet available.")

    # ── TAB 5: Sentiment ──────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("News Sentiment Analysis")
        if state.sentiment:
            sent = state.sentiment
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentiment", sent.sentiment_label)
            c2.metric("Score", f"{sent.aggregate_score:.2f}")
            c3.metric("Articles", sent.total_articles)
            c4.metric("Positive / Neg", f"{sent.positive_count} / {sent.negative_count}")

            if sent.top_themes:
                st.markdown(f"**Top Themes:** {' · '.join(['`' + t + '`' for t in sent.top_themes])}")

            # Sentiment pie chart
            pie_fig = px.pie(
                values=[sent.positive_count, sent.neutral_count, sent.negative_count],
                names=["Positive", "Neutral", "Negative"],
                color_discrete_map={
                    "Positive": SUCCESS_COLOR,
                    "Neutral":  C.YELLOW_30,
                    "Negative": DANGER_COLOR,
                },
                title="Sentiment Distribution",
                hole=0.35,
            )
            pie_fig.update_layout(height=300, paper_bgcolor=C.WHITE,
                                  font=dict(family="IBM Plex Sans, Segoe UI, sans-serif"))
            st.plotly_chart(pie_fig, use_container_width=True)

            # News list
            st.subheader("News Articles")
            for item in sent.items:
                badge = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(item.sentiment, "⚪")
                with st.expander(f"{badge} {item.headline[:100]} [{item.source}]"):
                    st.caption(f"Published: {item.published_at} · Confidence: {item.confidence:.0%} · Theme: {item.key_theme}")
                    if item.description:
                        st.write(item.description)
                    if item.url:
                        st.markdown(f"[Read more]({item.url})")
        else:
            st.info("Sentiment data not yet available.")

    # ── TAB 6: Risks ─────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Risk Assessment")
        if state.risk:
            risk = state.risk
            st.markdown(f"### Overall Risk: **{risk.overall_risk_level}** ({risk.overall_risk_score:.0f}/100)")
            st.write(risk.summary)

            # Risk indicator cards
            cols = st.columns(len(risk.indicators))
            level_colors = {
                "Low":    C.GREEN_10,
                "Medium": C.YELLOW_10,
                "High":   C.RED_10,
            }
            level_text_colors = {
                "Low":    C.GREEN_60,
                "Medium": WARNING_COLOR,
                "High":   C.RED_60,
            }
            for col, ind in zip(cols, risk.indicators):
                bg   = level_colors.get(ind.level, C.WHITE)
                txt  = level_text_colors.get(ind.level, C.GRAY_80)
                col.markdown(
                    f'<div style="background:{bg};border:1px solid {C.GRAY_20};'
                    f'padding:12px;border-radius:8px;text-align:center;">'
                    f'<b style="color:{C.GRAY_100}">{ind.category}</b><br>'
                    f'<span style="font-size:1.3em;font-weight:bold;color:{txt}">{ind.level}</span><br>'
                    f'<small style="color:{C.GRAY_60}">{ind.score:.0f}/100</small><br>'
                    f'<small style="color:{C.GRAY_60}">{ind.description[:60]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Risk bar chart
            risk_df = pd.DataFrame([
                {"Category": i.category, "Score": i.score, "Level": i.level}
                for i in risk.indicators
            ])
            fig = px.bar(
                risk_df, x="Category", y="Score", color="Level",
                color_discrete_map={
                    "Low":    SUCCESS_COLOR,
                    "Medium": C.YELLOW_30,
                    "High":   DANGER_COLOR,
                },
                title="Risk Scores by Category",
                range_y=[0, 100],
            )
            fig.update_layout(**chart_layout("Risk Scores by Category", height=360))
            st.plotly_chart(fig, use_container_width=True)

            # Revenue forecast
            if risk.revenue_forecast:
                st.subheader("Revenue Forecast")
                fc_df = pd.DataFrame([p.model_dump() for p in risk.revenue_forecast])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=fc_df["period"], y=fc_df["value"], name="Forecast",
                                     marker_color=PRIMARY_COLOR))
                if "lower" in fc_df.columns and fc_df["lower"].notna().any():
                    fig.add_trace(go.Scatter(
                        x=fc_df["period"], y=fc_df["upper"],
                        mode="lines", line=dict(width=0), name="Upper CI",
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc_df["period"], y=fc_df["lower"],
                        fill="tonexty", mode="lines", line=dict(width=0),
                        name="Lower CI", fillcolor=f"rgba(15,98,254,0.15)",
                    ))
                fig.update_layout(**chart_layout(f"Revenue Forecast ({risk.forecast_method})", height=350))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk data not yet available.")

    # ── TAB 7: Advanced Analysis ──────────────────────────────────────────────
    with tabs[6]:
        st.subheader("🔬 Advanced Analysis")
        adv = state.advanced

        if adv is None:
            st.info("Advanced analysis not yet available.")
        else:
            # ── Weighted Scoring Model ────────────────────────────────────────
            st.markdown("#### ⚖️ Weighted Composite Score")
            if adv.weighted_score:
                ws = adv.weighted_score
                # Weights slider in an expander
                with st.expander("🎚️ Adjust Scoring Weights (recalculates signal)"):
                    w_fund = st.slider("Fundamentals", 0, 100, int(ws.weights.get("fundamentals", 0.30) * 100), 5)
                    w_val = st.slider("Valuation", 0, 100, int(ws.weights.get("valuation", 0.20) * 100), 5)
                    w_mom = st.slider("Momentum", 0, 100, int(ws.weights.get("momentum", 0.20) * 100), 5)
                    w_sent = st.slider("Sentiment", 0, 100, int(ws.weights.get("sentiment", 0.15) * 100), 5)
                    w_risk = st.slider("Risk (inverse)", 0, 100, int(ws.weights.get("risk", 0.15) * 100), 5)
                    if st.button("Recalculate Score", key="recalc_score"):
                        try:
                            from agents.advanced_analysis_agent import _compute_weighted_score
                            new_w = {
                                "fundamentals": w_fund / 100,
                                "valuation": w_val / 100,
                                "momentum": w_mom / 100,
                                "sentiment": w_sent / 100,
                                "risk": w_risk / 100,
                            }
                            new_ws = _compute_weighted_score(state, new_w)
                            st.session_state.pipeline_state.advanced.weighted_score = new_ws
                            st.success(f"Recalculated: {new_ws.signal} ({new_ws.composite_score:.1f}/100)")
                        except Exception as e:
                            st.error(f"Recalculation failed: {e}")

                # Radar chart of component scores
                cats = ["Fundamentals", "Valuation", "Momentum", "Sentiment", "Risk (inv)"]
                vals = [
                    ws.fundamentals_score,
                    ws.valuation_score,
                    ws.momentum_score,
                    ws.sentiment_score,
                    ws.risk_score,
                ]
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    name="Score",
                    line=dict(color=PRIMARY_COLOR, width=2),
                    fillcolor=f"rgba(15,98,254,0.12)",
                ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100], gridcolor=C.GRAY_20),
                               bgcolor=C.WHITE),
                    title=f"Composite Score: {ws.composite_score:.1f}/100 → {ws.signal}",
                    height=350,
                    paper_bgcolor=C.WHITE,
                    font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=12),
                )
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(radar_fig, use_container_width=True)
                with col2:
                    css_class = {"Overweight": "overweight", "Hold": "hold", "Underweight": "underweight"}.get(ws.signal, "hold")
                    st.markdown(
                        f'<div class="rec-card {css_class}">{ws.signal}<br>'
                        f'<small>Score: {ws.composite_score:.1f}/100</small><br>'
                        f'<small>Confidence: {ws.signal_confidence:.0%}</small></div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── DCF Fair Value ────────────────────────────────────────────────
            st.markdown("#### 💲 DCF-Lite Fair Value")
            if adv.dcf:
                dcf = adv.dcf
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fair Value (Mid)", f"${dcf.fair_value_mid:.2f}")
                c2.metric("Fair Value (Low)", f"${dcf.fair_value_low:.2f}")
                c3.metric("Fair Value (High)", f"${dcf.fair_value_high:.2f}")
                c4.metric("Upside / Downside",
                          f"{dcf.upside_pct:+.1f}%" if dcf.upside_pct is not None else "N/A",
                          delta=f"{dcf.upside_pct:+.1f}%" if dcf.upside_pct is not None else None)
                st.caption(
                    f"WACC: {dcf.wacc:.1%} | Terminal Growth: {dcf.terminal_growth_rate:.1%} | "
                    f"Current Price: ${dcf.current_price or 'N/A'} | {dcf.method_note}"
                )
                # Projected FCF bar chart
                if dcf.projected_fcf:
                    fcf_fig = px.bar(
                        x=[f"Y+{i+1}" for i in range(len(dcf.projected_fcf))],
                        y=[v / 1e9 for v in dcf.projected_fcf],
                        labels={"x": "Year", "y": "FCF ($B)"},
                        title="Projected Free Cash Flow (5-Year)",
                        color_discrete_sequence=[PRIMARY_COLOR],
                    )
                    fcf_fig.update_layout(**chart_layout("Projected Free Cash Flow (5-Year)", height=320))
                    st.plotly_chart(fcf_fig, use_container_width=True)
            else:
                st.info("DCF not available (requires positive FCF and shares outstanding data).")

            st.divider()

            # ── Altman Z-Score ────────────────────────────────────────────────
            st.markdown("#### 🏦 Altman Z-Score (Bankruptcy Risk)")
            if adv.altman_z:
                az = adv.altman_z
                zone_color = {
                    "Safe":     C.GREEN_60,
                    "Grey":     WARNING_COLOR,
                    "Distress": C.RED_60,
                }.get(az.zone, C.GRAY_60)
                st.markdown(
                    f'<b>Z-Score: <span style="color:{zone_color};font-size:1.3em">{az.z_score:.2f}</span>'
                    f' — {az.zone} Zone</b>',
                    unsafe_allow_html=True,
                )
                st.caption(az.interpretation)
                comp_df = pd.DataFrame([{"Component": k, "Value": v} for k, v in az.components.items()])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Altman Z not available.")

            st.divider()

            # ── Backtest ──────────────────────────────────────────────────────
            st.markdown("#### 📉 Historical Signal Backtest (2-Year MA50/MA200 Strategy)")
            if adv.backtest:
                bt = adv.backtest
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Strategy Return", f"{bt.strategy_return_pct:+.1f}%")
                b2.metric("Buy & Hold", f"{bt.buyhold_return_pct:+.1f}%")
                b3.metric("Excess Return", f"{bt.excess_return_pct:+.1f}%",
                          delta=f"{bt.excess_return_pct:+.1f}%")
                b4.metric("Win Rate", f"{bt.win_rate:.0%} ({bt.n_trades} trades)")

                if bt.dates and bt.strategy_equity:
                    bt_fig = go.Figure()
                    bt_fig.add_trace(go.Scatter(
                        x=bt.dates, y=bt.strategy_equity,
                        name="MA Strategy", line=dict(color=PRIMARY_COLOR, width=2),
                    ))
                    bt_fig.add_trace(go.Scatter(
                        x=bt.dates, y=bt.buyhold_equity,
                        name="Buy & Hold", line=dict(color=SECONDARY_COLOR, width=2, dash="dash"),
                    ))
                    bt_fig.update_layout(
                        **chart_layout(
                            f"{state.ticker} Strategy vs Buy & Hold (Equity Curve, base=100)",
                            height=350,
                        ),
                        xaxis_title="Date",
                        yaxis_title="Equity (base 100)",
                    )
                    st.plotly_chart(bt_fig, use_container_width=True)
                st.caption(bt.note)
            else:
                st.info("Backtest not available (requires 200+ days of price data).")

            st.divider()

            # ── Forecast Quality ──────────────────────────────────────────────
            st.markdown("#### 📐 Forecast Model Quality (Hold-out MAPE)")
            if adv.price_forecast_quality:
                pfq = adv.price_forecast_quality
                st.metric(
                    f"Price Forecast MAPE ({pfq.method}, {pfq.holdout_periods}-period holdout)",
                    f"{pfq.mape:.1f}%" if pfq.mape is not None else "N/A",
                )
                if pfq.mape is not None:
                    if pfq.mape < 5:
                        st.success("Excellent forecast accuracy (MAPE < 5%)")
                    elif pfq.mape < 15:
                        st.warning(f"Moderate forecast accuracy (MAPE {pfq.mape:.1f}%)")
                    else:
                        st.error(f"Poor forecast accuracy (MAPE {pfq.mape:.1f}%) — treat price forecasts with caution")

            # ── Valuation Percentiles ────────────────────────────────────────
            if adv.valuation_percentiles:
                st.divider()
                st.markdown("#### 📊 Peer-Relative Valuation Percentiles")
                pct_df = pd.DataFrame([
                    {
                        "Metric": vp.metric,
                        f"{state.ticker} Value": round(vp.ticker_value, 2) if vp.ticker_value else "N/A",
                        "Percentile (0=cheapest)": f"{vp.percentile:.0f}%" if vp.percentile is not None else "N/A",
                        "Rank (1=cheapest)": vp.rank if vp.rank else "N/A",
                        "# Peers": vp.n_peers,
                    }
                    for vp in adv.valuation_percentiles
                    if vp.ticker_value is not None
                ])
                if not pct_df.empty:
                    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    # ── TAB 8: Filings & RAG ──────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("📑 SEC Filings & Risk Factor Analysis")
        rag = state.rag

        if rag is None:
            st.info("Filing analysis not yet available. Run the pipeline to ingest SEC filings.")
        else:
            # Document metadata
            if rag.documents:
                doc = rag.documents[0]
                st.markdown(
                    f"**Filing:** {doc.filing_type} | **Date:** {doc.filing_date} | "
                    f"**Chunks:** {doc.n_chunks} | **Index:** "
                    f"{'✅ FAISS' if rag.index_built else '🔤 Keyword'}"
                )
                if doc.sections_found:
                    st.caption(f"Sections found: {', '.join(doc.sections_found)}")

            st.divider()

            # Risk factors
            if rag.risk_summary:
                rs = rag.risk_summary
                gmode = getattr(rs, "granite_mode", "demo")
                badge_map3 = {
                    "live": '<span class="badge-live">🤖 Granite (live)</span>',
                    "demo": '<span class="badge-demo">🗂️ Demo data</span>',
                    "fallback": '<span class="badge-fallback">⚙️ Fallback</span>',
                }
                st.markdown(
                    f"#### ⚠️ Key Risk Factors &nbsp; {badge_map3.get(gmode, '')}",
                    unsafe_allow_html=True,
                )
                for i, rf in enumerate(rs.risk_factors, 1):
                    st.markdown(f"{i}. {rf}")

                st.divider()
                st.markdown("#### 🔭 Management Outlook")
                st.info(rs.management_outlook or "Not available.")

                if rs.key_themes:
                    st.markdown(f"**Key Themes:** {' · '.join(['`' + t + '`' for t in rs.key_themes])}")

                st.markdown("""
                <div class="disclaimer">
                ⚠️ <b>Disclaimer:</b> Risk factors are extracted by AI from SEC filing excerpts.
                Always read the full original filing for complete and accurate information.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Risk factor extraction not yet available.")

    # ── TAB 9: Ask the Analyst ────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("💬 Ask the Analyst")
        st.caption(
            "Ask questions about the company's SEC filings. Answers are grounded in filing excerpts "
            "using Retrieval-Augmented Generation (RAG) with IBM Granite."
        )

        # Initialise chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Suggested questions
        st.markdown("**Suggested questions:**")
        suggestions = [
            f"What are the key risks in {state.ticker}'s latest 10-K?",
            f"What is {state.ticker}'s management outlook for next year?",
            f"How does {state.ticker} describe its competitive position?",
            f"What does {state.ticker}'s filing say about liquidity and capital?",
        ]
        sug_cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if sug_cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": sug})
                st.session_state._pending_question = sug

        st.divider()

        # Chat input
        user_q = st.chat_input(f"Ask about {state.ticker}'s filings…")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state._pending_question = user_q

        # Answer pending question
        if getattr(st.session_state, "_pending_question", None):
            q = st.session_state._pending_question
            st.session_state._pending_question = None
            with st.spinner("Retrieving from filings and generating answer…"):
                try:
                    from services.analyst_chat import ask_analyst
                    answer = ask_analyst(
                        question=q,
                        ticker=state.ticker,
                        rag_state=state.rag,
                        filing_type="10-K",
                        filing_date=(
                            state.rag.documents[0].filing_date
                            if state.rag and state.rag.documents else "2024-10-31"
                        ),
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer.answer,
                        "sources": [{"section": s.section, "date": s.filing_date, "score": s.score} for s in answer.sources],
                        "mode": answer.granite_mode,
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error generating answer: {e}",
                        "sources": [],
                        "mode": "error",
                    })

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["content"])
                    sources = msg.get("sources", [])
                    mode = msg.get("mode", "demo")
                    if sources:
                        badge_map4 = {
                            "live": '<span class="badge-live">🤖 Granite (live)</span>',
                            "demo": '<span class="badge-demo">🗂️ Demo data</span>',
                            "fallback": '<span class="badge-fallback">⚙️ Fallback</span>',
                        }
                        _src_parts = " | ".join(
                            f'"{s["section"]}" ({s["date"]})' for s in sources[:3]
                        )
                        st.caption(
                            f"{badge_map4.get(mode, '')} · Sources: {_src_parts}",
                            unsafe_allow_html=True,
                        )

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Answers are generated by an AI model (IBM Granite) using SEC filing excerpts.
        They are <b>NOT financial advice</b>. Always read the original SEC filings for complete information.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 10: Watchlist ─────────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("⭐ Watchlist")
        st.caption("Save completed analyses here for quick comparison. Use the sidebar '⭐ Save to Watchlist' button after running analysis.")

        watchlist = st.session_state.get("watchlist", [])
        if not watchlist:
            st.info("Your watchlist is empty. Run an analysis and click **⭐ Save to Watchlist** in the sidebar.")
        else:
            # Clear watchlist button
            if st.button("🗑️ Clear Watchlist", key="clear_wl"):
                st.session_state.watchlist = []
                st.rerun()

            # Build comparison table
            wl_rows = []
            for entry in watchlist:
                rec = entry.get("recommendation", "Hold")
                rec_emoji = {"Overweight": "🟢", "Hold": "🟡", "Underweight": "🔴"}.get(rec, "⚪")
                risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(entry.get("risk_level", ""), "⚪")
                wl_rows.append({
                    "Ticker": entry.get("ticker", ""),
                    "Recommendation": f"{rec_emoji} {rec}",
                    "Confidence": f"{entry.get('confidence', 0):.0%}",
                    "Score": f"{entry.get('composite_score', 'N/A'):.1f}" if entry.get('composite_score') else "N/A",
                    "Risk": f"{risk_emoji} {entry.get('risk_level', 'N/A')}",
                    "Peers": ", ".join(entry.get("competitors", [])) or "—",
                    "Mode": entry.get("granite_mode", "demo"),
                    "Date": entry.get("generated_at", "")[:10],
                })
            wl_df = pd.DataFrame(wl_rows)
            st.dataframe(wl_df, use_container_width=True, hide_index=True)

            # Recommendation distribution pie
            if len(watchlist) >= 2:
                st.divider()
                st.markdown("#### Portfolio-Level Recommendation Mix")
                from collections import Counter
                rec_counts = Counter(e.get("recommendation", "Hold") for e in watchlist)
                pie_fig = go.Figure(go.Pie(
                    labels=list(rec_counts.keys()),
                    values=list(rec_counts.values()),
                    marker_colors=[SUCCESS_COLOR, C.YELLOW_30, DANGER_COLOR],
                    hole=0.4,
                ))
                pie_fig.update_layout(
                    **chart_layout("Watchlist Recommendations", height=300),
                )
                st.plotly_chart(pie_fig, use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Watchlist data is for reference only and is <b>NOT financial advice</b>.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 11: Portfolio Mode ────────────────────────────────────────────────
    with tabs[10]:
        st.subheader("💼 Portfolio Mode")
        st.caption(
            "Define a multi-ticker portfolio with custom weights. "
            "The pipeline runs on each ticker and computes a blended analysis."
        )

        # Portfolio builder
        if "portfolio_positions" not in st.session_state:
            st.session_state.portfolio_positions = [
                {"ticker": "AAPL", "weight_pct": 40.0},
                {"ticker": "MSFT", "weight_pct": 30.0},
                {"ticker": "GOOGL", "weight_pct": 30.0},
            ]

        st.markdown("#### Define Portfolio Positions")
        pf_col1, pf_col2 = st.columns([2, 1])
        with pf_col1:
            pf_name = st.text_input("Portfolio Name", value="My Portfolio", key="pf_name")

        # Editable position table
        pf_rows = []
        for i, pos in enumerate(st.session_state.portfolio_positions):
            c1, c2, c3 = st.columns([2, 2, 1])
            t = c1.text_input(f"Ticker #{i+1}", value=pos["ticker"], key=f"pf_t_{i}").upper().strip()
            w = c2.number_input(f"Weight % #{i+1}", min_value=0.0, max_value=100.0,
                                value=pos["weight_pct"], step=5.0, key=f"pf_w_{i}")
            remove = c3.button("❌", key=f"pf_rm_{i}")
            if not remove and t:
                pf_rows.append({"ticker": t, "weight_pct": w})

        c_add, c_run = st.columns(2)
        if c_add.button("➕ Add Position"):
            st.session_state.portfolio_positions = pf_rows + [{"ticker": "", "weight_pct": 10.0}]
            st.rerun()
        else:
            st.session_state.portfolio_positions = pf_rows

        # Normalise weights display
        total_w = sum(p["weight_pct"] for p in pf_rows if p["ticker"])
        if total_w > 0 and abs(total_w - 100) > 0.1:
            st.warning(f"⚠️ Weights sum to {total_w:.1f}% — will be normalised to 100% at analysis time.")

        st.divider()

        if c_run.button("🚀 Run Portfolio Analysis", type="primary", disabled=st.session_state.running):
            valid_positions = [p for p in pf_rows if p["ticker"]]
            if len(valid_positions) < 2:
                st.error("Please add at least 2 tickers with non-zero weights.")
            else:
                # Run pipeline on each ticker sequentially
                from pipeline.orchestrator import PipelineOrchestrator
                from models.investor_profile import PortfolioAnalysisResult, PortfolioPosition, PortfolioConfig

                _total = sum(p["weight_pct"] for p in valid_positions)
                _normalised = [
                    PortfolioPosition(
                        ticker=p["ticker"],
                        weight_pct=round(p["weight_pct"] / _total * 100, 2),
                    )
                    for p in valid_positions
                ]
                _portfolio = PortfolioConfig(positions=_normalised, name=pf_name)
                _profile = st.session_state.get("_investor_profile")

                ticker_states = {}
                progress_bar = st.progress(0, text="Running portfolio analysis…")
                for idx, pos in enumerate(_normalised):
                    progress_bar.progress((idx) / len(_normalised), text=f"Analysing {pos.ticker}…")
                    try:
                        orch = PipelineOrchestrator()
                        t_state = orch.run(pos.ticker, [])
                        if _profile:
                            t_state.investor_profile = _profile
                        ticker_states[pos.ticker] = t_state
                    except Exception as exc:
                        st.warning(f"Pipeline failed for {pos.ticker}: {exc}")
                progress_bar.progress(1.0, text="Portfolio analysis complete!")

                # Compute blended metrics
                total_norm = sum(pos.weight_pct for pos in _normalised)
                w_composite = 0.0
                w_risk = 0.0
                ticker_results = {}
                for pos in _normalised:
                    t = pos.ticker
                    frac = pos.weight_pct / total_norm if total_norm > 0 else 0
                    ts = ticker_states.get(t)
                    if ts:
                        comp = (ts.advanced.weighted_score.composite_score
                                if ts.advanced and ts.advanced.weighted_score else 50.0)
                        rsk = ts.risk.overall_risk_score if ts.risk else 50.0
                        rec = ts.recommendation.recommendation if ts.recommendation else "Hold"
                        conf = ts.recommendation.confidence if ts.recommendation else 0.5
                        w_composite += comp * frac
                        w_risk += rsk * frac
                        ticker_results[t] = {
                            "recommendation": rec,
                            "confidence": conf,
                            "composite_score": comp,
                            "risk_score": rsk,
                        }

                blended_signal = "Overweight" if w_composite >= 65 else ("Hold" if w_composite >= 45 else "Underweight")
                blended_risk = "Low" if w_risk < 35 else ("High" if w_risk >= 65 else "Medium")
                portfolio_result = PortfolioAnalysisResult(
                    portfolio_name=pf_name,
                    positions=_normalised,
                    weighted_composite_score=round(w_composite, 1),
                    blended_recommendation=blended_signal,
                    blended_confidence=round(0.5 + abs(w_composite - 50) / 100, 2),
                    weighted_risk_score=round(w_risk, 1),
                    blended_risk_level=blended_risk,
                    ticker_results=ticker_results,
                    diversification_note=(
                        f"{len(_normalised)}-ticker portfolio; "
                        f"weights range {min(p.weight_pct for p in _normalised):.0f}% – "
                        f"{max(p.weight_pct for p in _normalised):.0f}%"
                    ),
                )
                st.session_state.portfolio_results = {
                    "result": portfolio_result.model_dump(),
                    "ticker_states": ticker_states,
                }

        # Show portfolio results
        pf_data = st.session_state.get("portfolio_results", {})
        if pf_data:
            pr = pf_data.get("result", {})
            st.divider()
            st.markdown(f"### 📊 Portfolio: {pr.get('portfolio_name', 'My Portfolio')}")

            # Blended metrics
            bc1, bc2, bc3, bc4 = st.columns(4)
            bl_rec = pr.get("blended_recommendation", "Hold")
            rec_emoji2 = {"Overweight": "🟢", "Hold": "🟡", "Underweight": "🔴"}.get(bl_rec, "⚪")
            bc1.metric("Blended Signal", f"{rec_emoji2} {bl_rec}")
            bc2.metric("Composite Score", f"{pr.get('weighted_composite_score', 'N/A')}/100")
            bc3.metric("Portfolio Risk", pr.get("blended_risk_level", "Medium"))
            bc4.metric("Risk Score", f"{pr.get('weighted_risk_score', 'N/A')}/100")

            st.caption(pr.get("diversification_note", ""))

            # Per-ticker breakdown table
            ticker_res = pr.get("ticker_results", {})
            positions = pr.get("positions", [])
            if ticker_res:
                st.markdown("#### Per-Ticker Breakdown")
                rows = []
                for pos in positions:
                    t = pos.get("ticker") if isinstance(pos, dict) else pos.ticker
                    w = pos.get("weight_pct") if isinstance(pos, dict) else pos.weight_pct
                    tr = ticker_res.get(t, {})
                    rec_e = {"Overweight": "🟢", "Hold": "🟡", "Underweight": "🔴"}.get(tr.get("recommendation", ""), "⚪")
                    rows.append({
                        "Ticker": t,
                        "Weight": f"{w:.1f}%",
                        "Recommendation": f"{rec_e} {tr.get('recommendation', 'N/A')}",
                        "Confidence": f"{tr.get('confidence', 0):.0%}",
                        "Composite": f"{tr.get('composite_score', 0):.1f}",
                        "Risk Score": f"{tr.get('risk_score', 0):.1f}",
                    })
                pf_df = pd.DataFrame(rows)
                st.dataframe(pf_df, use_container_width=True, hide_index=True)

                # Weight pie chart
                pie_w_fig = go.Figure(go.Pie(
                    labels=[r["Ticker"] for r in rows],
                    values=[float(r["Weight"].replace("%", "")) for r in rows],
                    hole=0.4,
                    marker_colors=CHART_PALETTE,
                ))
                pie_w_fig.update_layout(**chart_layout("Portfolio Weights", height=300))
                st.plotly_chart(pie_w_fig, use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Portfolio analysis is for educational purposes only and is <b>NOT financial advice</b>.
        Diversification does not guarantee profits or protect against losses.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 12: Full Report ───────────────────────────────────────────────────
    with tabs[11]:
        st.subheader("Full Report")
        if state.report:
            report = state.report
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Run ID:** `{report.run_id}`")
                st.markdown(f"**Generated:** {report.generated_at}")
                st.markdown(f"**Ticker:** {report.ticker}")
                st.markdown(f"**Peers:** {', '.join(report.competitors)}")
            with col2:
                if report.pdf_path and Path(report.pdf_path).exists():
                    with open(report.pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=f,
                            file_name=f"InvestIQ_{report.ticker}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.info("PDF not available (reportlab not installed)")

                json_str = json.dumps(report.model_dump(), indent=2, default=str)
                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=json_str,
                    file_name=f"InvestIQ_{report.ticker}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # Raw JSON viewer
            with st.expander("View Raw JSON Report"):
                st.json(report.model_dump())

            st.markdown(f"""
            <div class="disclaimer">
            ⚠️ <b>Disclaimer:</b> {report.disclaimer}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Report not yet generated.")

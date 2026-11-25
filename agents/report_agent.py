"""
Agent 9 — Report Agent.
Assembles all outputs into a final structured JSON report and exports
a PDF using reportlab.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.report import FinalReport
from services.storage_service import save_to_cos


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_v(val: Any) -> str:
    """Format a value for the PDF — billions/millions shorthand."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if abs(f) >= 1e9:
            return f"${f/1e9:.2f}B"
        if abs(f) >= 1e6:
            return f"${f/1e6:.1f}M"
        return f"{f:.4g}"
    except (TypeError, ValueError):
        return str(val)


def _safe_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _build_pdf(report: FinalReport, out_path: str, state: "PipelineState | None" = None) -> bool:  # noqa: F821
    """
    Build a PDF report using reportlab. Returns True on success.

    Phase 5 enhancements:
    - Investor profile header section
    - Advanced analysis (DCF fair value, Altman Z, composite score)
    - Analyst reviews / override audit trail
    - Colour-coded risk table (Low=green, Medium=amber, High=red)
    """
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.lib import colors  # type: ignore
        from reportlab.platypus import (  # type: ignore
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        h3 = styles["Heading3"]
        normal = styles["Normal"]
        small = ParagraphStyle("Small", parent=normal, fontSize=8)
        disclaimer_style = ParagraphStyle("Disclaimer", parent=normal, fontSize=7, textColor=colors.grey)

        doc = SimpleDocTemplate(out_path, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        story = []

        # ── Title ─────────────────────────────────────────────────────────────
        story.append(Paragraph(f"InvestIQ Analysis Report: {report.ticker}", h1))
        story.append(Paragraph(f"Generated: {report.generated_at[:19].replace('T', ' ')} UTC", normal))
        story.append(Paragraph(f"Peers: {', '.join(report.competitors) or 'None'}", normal))
        story.append(Paragraph(f"Run ID: {report.run_id[:8]}…", small))
        story.append(Spacer(1, 0.15 * inch))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82d4")))
        story.append(Spacer(1, 0.15 * inch))

        # ── Investor Profile ───────────────────────────────────────────────────
        profile = getattr(state, "investor_profile", None) if state else None
        if profile is not None:
            story.append(Paragraph("Investor Profile", h2))
            prof_data = [
                ["Risk Tolerance", profile.risk_tolerance],
                ["Horizon", profile.horizon],
                ["Intent", profile.intent],
                ["Portfolio Size", profile.portfolio_size],
                ["ESG Preference", profile.esg_preference],
            ]
            if profile.sector_notes:
                prof_data.append(["Sector Notes", profile.sector_notes])
            prof_tbl = Table(prof_data, colWidths=[2.5*inch, 4.5*inch])
            prof_tbl.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f7f8fa")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
            ]))
            story.append(prof_tbl)
            story.append(Spacer(1, 0.2 * inch))

        # ── Recommendation ────────────────────────────────────────────────────
        rec = report.recommendation
        if rec:
            story.append(Paragraph("Investment Recommendation", h2))
            rec_val = rec.get("recommendation", "Hold")
            rec_color = {
                "Overweight": colors.HexColor("#155724"),
                "Hold":       colors.HexColor("#856404"),
                "Underweight":colors.HexColor("#721c24"),
            }.get(rec_val, colors.black)
            story.append(Paragraph(
                f'<font color="#{rec_color.hexval() if hasattr(rec_color, "hexval") else "000000"}">'
                f'<b>{rec_val}</b></font>'
                f' &nbsp;|&nbsp; Confidence: {float(rec.get("confidence", 0)) * 100:.0f}%'
                f' &nbsp;|&nbsp; Horizon: {rec.get("target_horizon", "12 months")}',
                h3,
            ))
            granite_mode = rec.get("granite_mode", "demo")
            story.append(Paragraph(
                f"AI Mode: {granite_mode.upper()} | "
                f"Disclaimer: {rec.get('disclaimer', '')}",
                small,
            ))
            story.append(Spacer(1, 0.08 * inch))
            for r in rec.get("reasons", []):
                metric_txt = f" [{r.get('metric', '')}={r.get('value', '')}]" if r.get("metric") else ""
                story.append(Paragraph(f"• {r.get('point', '')}{metric_txt}", normal))
            if rec.get("forward_outlook"):
                story.append(Spacer(1, 0.08 * inch))
                story.append(Paragraph(f"Outlook: {rec['forward_outlook']}", normal))
            if rec.get("key_risks"):
                story.append(Paragraph(f"Key Risks: {', '.join(rec['key_risks'])}", small))
            story.append(Spacer(1, 0.2 * inch))

        # ── Advanced Analysis ─────────────────────────────────────────────────
        adv = getattr(state, "advanced", None) if state else None
        if adv is not None:
            story.append(Paragraph("Advanced Analysis", h2))

            # Composite score + signal
            ws = getattr(adv, "weighted_score", None)
            if ws:
                sig_color = {"Overweight": "#155724", "Hold": "#856404", "Underweight": "#721c24"}.get(ws.signal, "#000000")
                story.append(Paragraph(
                    f'Composite Score: <b>{ws.composite_score:.1f}/100</b> → '
                    f'<font color="#{sig_color}"><b>{ws.signal}</b></font> '
                    f'({ws.signal_confidence:.0%} confidence)',
                    normal,
                ))
                score_rows = [["Dimension", "Score", "Weight", "Contribution"]]
                for dim, raw, w in [
                    ("Fundamentals", ws.fundamentals_score, ws.weights.get("fundamentals", 0.3)),
                    ("Valuation",    ws.valuation_score,    ws.weights.get("valuation",    0.2)),
                    ("Momentum",     ws.momentum_score,     ws.weights.get("momentum",     0.2)),
                    ("Sentiment",    ws.sentiment_score,     ws.weights.get("sentiment",    0.15)),
                    ("Risk (inv.)",  ws.risk_score,          ws.weights.get("risk",         0.15)),
                ]:
                    contrib = (raw - 50) * w
                    score_rows.append([dim, f"{raw:.1f}", f"{w:.0%}", f"{contrib:+.1f}"])
                s_tbl = Table(score_rows, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 2.1*inch])
                s_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b82d4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
                ]))
                story.append(Spacer(1, 0.08 * inch))
                story.append(s_tbl)
                story.append(Spacer(1, 0.1 * inch))

            # DCF
            dcf = getattr(adv, "dcf", None)
            if dcf:
                story.append(Paragraph(
                    f"DCF Fair Value: Bear ${dcf.fair_value_low:.2f} / "
                    f"Base ${dcf.fair_value_mid:.2f} / "
                    f"Bull ${dcf.fair_value_high:.2f} "
                    f"(WACC {dcf.wacc:.1%}, g={dcf.terminal_growth_rate:.1%})"
                    + (f" | Upside: {dcf.upside_pct:+.1f}%" if dcf.upside_pct is not None else ""),
                    normal,
                ))

            # Altman Z
            az = getattr(adv, "altman_z", None)
            if az:
                zone_color = {"Safe": "#155724", "Grey": "#856404", "Distress": "#721c24"}.get(az.zone, "#000000")
                story.append(Paragraph(
                    f'Altman Z-Score: <font color="#{zone_color}"><b>{az.z_score:.2f} ({az.zone})</b></font> — {az.interpretation}',
                    normal,
                ))
            story.append(Spacer(1, 0.2 * inch))

        # ── Risk ──────────────────────────────────────────────────────────────
        risk = report.risk
        if risk:
            story.append(Paragraph("Risk Assessment", h2))
            story.append(Paragraph(
                f"Overall Risk: <b>{risk.get('overall_risk_level', 'N/A')}</b> "
                f"(Score: {risk.get('overall_risk_score', 0):.0f}/100)",
                normal,
            ))
            story.append(Spacer(1, 0.08 * inch))
            ind_data = [["Category", "Level", "Score", "Description"]]
            for ind in risk.get("indicators", []):
                ind_data.append([
                    ind.get("category", ""),
                    ind.get("level", ""),
                    f"{ind.get('score', 0):.0f}",
                    ind.get("description", "")[:60],
                ])
            if len(ind_data) > 1:
                tbl = Table(ind_data, colWidths=[1.5*inch, 0.9*inch, 0.7*inch, 3.9*inch])
                # Colour-code rows by risk level
                row_styles = [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b82d4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ]
                for i, ind in enumerate(risk.get("indicators", []), start=1):
                    lvl = ind.get("level", "")
                    if lvl == "High":
                        row_styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8d7da")))
                    elif lvl == "Medium":
                        row_styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff3cd")))
                    else:
                        row_styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#d4edda")))
                tbl.setStyle(TableStyle(row_styles))
                story.append(tbl)
            story.append(Spacer(1, 0.2 * inch))

        # ── Key KPIs ──────────────────────────────────────────────────────────
        kpis = report.kpis.get("primary", {})
        if kpis:
            story.append(Paragraph("Key Financial KPIs", h2))
            kpi_rows = [["Metric", report.ticker]]
            kpi_map = [
                ("Revenue",           kpis.get("revenue", {}).get("current")),
                ("Revenue Growth YoY", kpis.get("revenue_growth_yoy")),
                ("Net Income",        kpis.get("net_income", {}).get("current")),
                ("Net Margin",        kpis.get("net_margin")),
                ("EBITDA",            kpis.get("ebitda")),
                ("EPS (TTM)",         kpis.get("eps_ttm")),
                ("Free Cash Flow",    kpis.get("free_cash_flow")),
                ("Total Debt",        kpis.get("total_debt")),
                ("Total Cash",        kpis.get("total_cash")),
            ]
            for label, val in kpi_map:
                if val is not None:
                    kpi_rows.append([label, _fmt_v(val)])
            if len(kpi_rows) > 1:
                kpi_tbl = Table(kpi_rows, colWidths=[3*inch, 4*inch])
                kpi_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b82d4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
                ]))
                story.append(kpi_tbl)
            story.append(Spacer(1, 0.2 * inch))

        # ── Analyst Reviews ───────────────────────────────────────────────────
        reviews = getattr(state, "analyst_reviews", []) if state else []
        if reviews:
            story.append(Paragraph("Analyst Review Audit Trail", h2))
            for i, rv in enumerate(reviews, 1):
                override = rv.override_recommendation if hasattr(rv, "override_recommendation") else rv.get("override_recommendation", "")
                name = rv.analyst_name if hasattr(rv, "analyst_name") else rv.get("analyst_name", "?")
                ts = rv.reviewed_at if hasattr(rv, "reviewed_at") else rv.get("reviewed_at", "")
                notes = rv.notes if hasattr(rv, "notes") else rv.get("notes", "")
                flags = rv.flags if hasattr(rv, "flags") else rv.get("flags", [])
                story.append(Paragraph(
                    f"Review #{i} — {name} @ {ts[:19].replace('T', ' ')} UTC",
                    h3,
                ))
                if override:
                    story.append(Paragraph(f"Override: {override}", normal))
                if notes:
                    story.append(Paragraph(f"Notes: {notes}", normal))
                if flags:
                    story.append(Paragraph(f"Flags: {', '.join(flags)}", small))
                story.append(Spacer(1, 0.1 * inch))

        # ── Disclaimer ────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(report.disclaimer, disclaimer_style))

        doc.build(story)
        return True
    except ImportError:
        return False
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("PDF generation failed: %s", exc)
        return False


class ReportAgent(BaseAgent):
    name = "report"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        try:
            now = datetime.now(timezone.utc).isoformat()
            report = FinalReport(
                run_id=state.run_id,
                ticker=state.ticker,
                competitors=state.competitors,
                generated_at=now,
                kpis=_safe_dict(state.kpis),
                ratios=_safe_dict(state.ratios),
                benchmark=_safe_dict(state.benchmark),
                sentiment=_safe_dict(state.sentiment),
                risk=_safe_dict(state.risk),
                recommendation=_safe_dict(state.recommendation),
                quality_report=_safe_dict(state.validation),
            )

            # Save JSON to temp file
            report_dir = Path(tempfile.gettempdir()) / "investiq_reports"
            report_dir.mkdir(exist_ok=True)
            json_path = report_dir / f"{state.run_id}.json"
            pdf_path = report_dir / f"{state.run_id}.pdf"

            json_path.write_text(
                json.dumps(report.model_dump(), indent=2, default=str),
                encoding="utf-8"
            )
            report.json_path = str(json_path)

            # PDF export (Phase 5: pass state for investor profile + reviews)
            if _build_pdf(report, str(pdf_path), state=state):
                report.pdf_path = str(pdf_path)

            # Upload to COS
            cos_key = f"reports/{state.ticker}/{state.run_id}.json"
            save_to_cos(cos_key, report.model_dump())
            report.cos_key = cos_key

            state.report = report
            self._finish_step(
                state,
                f"Report saved: {json_path}, PDF: {report.pdf_path or 'N/A'}"
            )
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

"""
Agent 10 — RAG Agent (Phase 3).

Runs after the report agent (or as an optional enrichment step).
1. Fetches 10-K text from SEC EDGAR (or demo stubs)
2. Chunks and embeds into FAISS vector store
3. Uses Granite to extract risk factors and management outlook
   from the most relevant chunks

The vector store is kept in-process for the "Ask the Analyst" chat tab.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from models.pipeline_state import PipelineState
from models.rag import RAGResult, RAGDocument, RiskFactorSummary
from services.rag_service import (
    fetch_filing_text,
    chunk_text,
    get_or_build_index,
)
from services.watsonx_client import call_granite_json

RISK_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "risk_summary_prompt.txt"


def _load_risk_prompt() -> str:
    try:
        return RISK_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return "Summarize risk factors for {{TICKER}} from these excerpts: {{EXCERPTS}}"


def _build_risk_prompt(ticker: str, filing_type: str, filing_date: str, excerpts: list[str]) -> str:
    template = _load_risk_prompt()
    excerpts_text = "\n\n---\n\n".join(
        f"[Section excerpt {i+1}]\n{e[:600]}"
        for i, e in enumerate(excerpts[:6])
    )
    return (
        template
        .replace("{{TICKER}}", ticker)
        .replace("{{FILING_TYPE}}", filing_type)
        .replace("{{FILING_DATE}}", filing_date)
        .replace("{{EXCERPTS}}", excerpts_text)
    )


def _demo_risk_summary(ticker: str, filing_type: str, filing_date: str) -> RiskFactorSummary:
    """Pre-built risk summaries for demo mode."""
    summaries = {
        "AAPL": RiskFactorSummary(
            ticker="AAPL",
            filing_type=filing_type,
            filing_date=filing_date,
            risk_factors=[
                "Intense competition from Samsung, Google, and other hardware makers",
                "China revenue exposure and geopolitical trade restriction risk",
                "Antitrust investigations in EU and US over App Store policies",
                "Supply chain concentration risk with TSMC for semiconductor manufacturing",
                "Global macroeconomic headwinds may reduce consumer electronics spending",
            ],
            management_outlook=(
                "Management expects Services segment to continue growing at double-digit rates in FY2025, "
                "with gross margin guidance of 46–47%. iPhone volumes are expected to stabilize in China "
                "as competitive dynamics evolve. AI features (Apple Intelligence) are expected to drive "
                "upgrade cycles and ecosystem engagement."
            ),
            key_themes=["AI & Services Growth", "China Exposure", "Regulatory Risk"],
            granite_mode="demo",
        ),
        "MSFT": RiskFactorSummary(
            ticker="MSFT",
            filing_type=filing_type,
            filing_date=filing_date,
            risk_factors=[
                "Competition from AWS and Google Cloud in enterprise cloud market",
                "AI product race; competitors may deliver superior LLM capabilities",
                "Cybersecurity incidents risk customer trust and regulatory penalties",
                "DOJ antitrust investigation into Microsoft Teams bundling",
                "Macroeconomic slowdown could reduce enterprise IT spending",
            ],
            management_outlook=(
                "Azure growth expected to accelerate in FY2025 as new AI capacity comes online. "
                "Copilot enterprise adoption at 1M seats with significant expansion opportunity. "
                "Operating margins expected to remain above 44% driven by software and cloud mix."
            ),
            key_themes=["AI & Cloud Growth", "Security Risk", "Regulatory Risk"],
            granite_mode="demo",
        ),
        "GOOGL": RiskFactorSummary(
            ticker="GOOGL",
            filing_type=filing_type,
            filing_date=filing_date,
            risk_factors=[
                "Generative AI alternatives threatening Google Search market share",
                "DOJ antitrust case challenging default search agreement with Apple",
                "YouTube competition from TikTok in short-form video advertising",
                "Cloud business competing in a market dominated by AWS and Azure",
                "Privacy regulations limiting targeted advertising effectiveness",
            ],
            management_outlook=(
                "Google Search remains dominant with 12% revenue growth; AI Overviews enhancing user engagement. "
                "Google Cloud achieving profitability with 28% revenue growth. "
                "Gemini investments expected to defend and extend AI-driven product differentiation."
            ),
            key_themes=["AI Competition", "Regulatory Risk", "Cloud Growth"],
            granite_mode="demo",
        ),
        "AMZN": RiskFactorSummary(
            ticker="AMZN",
            filing_type=filing_type,
            filing_date=filing_date,
            risk_factors=[
                "Labor cost inflation and unionization of fulfillment center workers",
                "Regulatory scrutiny of Amazon marketplace and FBA practices",
                "AWS competition from Azure and Google Cloud gaining enterprise share",
                "Thin retail margins sensitive to consumer spending downturns",
                "Significant capital expenditure required for AI infrastructure buildout",
            ],
            management_outlook=(
                "AWS operating income expected to grow significantly on AI demand. "
                "North American retail returned to 5.3% operating margin. "
                "Advertising segment growing 20% remains a high-margin growth driver."
            ),
            key_themes=["AWS & AI Growth", "Labor Risk", "Regulatory Risk"],
            granite_mode="demo",
        ),
    }
    return summaries.get(ticker.upper(), summaries["AAPL"])


class RAGAgent(BaseAgent):
    name = "rag_ingestion"

    def run(self, state: PipelineState) -> PipelineState:
        self._start_step(state)
        self._log_input(ticker=state.ticker)

        try:
            filing_type = "10-K"

            # 1. Fetch filing text
            text, filing_date, source_url = fetch_filing_text(state.ticker, filing_type)

            # 2. Chunk
            chunks = chunk_text(
                text, state.ticker, filing_type, filing_date, source_url
            )

            # 3. Build / load FAISS index
            store = get_or_build_index(state.ticker, chunks)
            index_built = store._index is not None or bool(store._chunks)

            doc = RAGDocument(
                ticker=state.ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                source_url=source_url,
                total_chars=len(text),
                n_chunks=len(chunks),
                sections_found=list({c.section for c in chunks}),
            )

            # 4. Extract risk summary using Granite
            # Use Risk Factors and MD&A chunks
            priority_chunks = [c for c in chunks if c.section in ("Risk Factors", "MD&A", "Outlook", "Forward-Looking Statements")]
            if not priority_chunks:
                priority_chunks = chunks[:6]

            from config.settings import settings as _s
            if _s.demo_mode:
                risk_summary = _demo_risk_summary(state.ticker, filing_type, filing_date)
                granite_mode = "demo"
            else:
                try:
                    prompt = _build_risk_prompt(
                        state.ticker,
                        filing_type,
                        filing_date,
                        [c.text for c in priority_chunks[:6]],
                    )
                    result, granite_mode = call_granite_json(prompt, max_tokens=1200)
                    risk_summary = RiskFactorSummary(
                        ticker=state.ticker,
                        filing_type=filing_type,
                        filing_date=filing_date,
                        risk_factors=result.get("risk_factors", []),
                        management_outlook=result.get("management_outlook", ""),
                        key_themes=result.get("key_themes", []),
                        granite_mode=granite_mode,
                    )
                except Exception as exc:
                    self.logger.warning("Granite risk summary failed: %s — using demo", exc)
                    risk_summary = _demo_risk_summary(state.ticker, filing_type, filing_date)
                    granite_mode = "fallback"

            state.rag = RAGResult(
                ticker=state.ticker,
                documents=[doc],
                risk_summary=risk_summary,
                index_built=index_built,
                index_path=store.index_path.as_posix(),
                n_total_chunks=len(chunks),
            )

            self._log_output(
                ticker=state.ticker,
                chunks=len(chunks),
                index=index_built,
                risk_factors=len(risk_summary.risk_factors),
            )
            self._finish_step(
                state,
                f"Ingested {len(chunks)} chunks from {filing_type} ({filing_date}). "
                f"Risk factors: {len(risk_summary.risk_factors)}. Index: {'FAISS' if index_built else 'keyword'}."
            )
        except Exception as exc:
            self._fail_step(state, str(exc))
        return state

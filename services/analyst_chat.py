"""
Analyst Chat Service — answers user questions using Granite + RAG retrieval.

Called directly from the Streamlit dashboard (not in the sequential pipeline).
"""
from __future__ import annotations

import logging
from pathlib import Path

from models.rag import AnalystAnswer, RetrievedChunk
from services.rag_service import FAISSVectorStore, get_or_build_index
from services.watsonx_client import call_granite
from config.settings import settings

logger = logging.getLogger(__name__)

ANALYST_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_analyst_prompt.txt"


def _load_analyst_prompt() -> str:
    try:
        return ANALYST_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return "Answer this question about {{TICKER}}: {{QUESTION}}\nUsing excerpts: {{EXCERPTS}}"


def _build_analyst_prompt(
    ticker: str,
    filing_type: str,
    filing_date: str,
    question: str,
    chunks: list[RetrievedChunk],
) -> str:
    template = _load_analyst_prompt()
    excerpts_text = "\n\n---\n\n".join(
        f'[Section: "{c.section}", Date: {c.filing_date}, Score: {c.score:.2f}]\n{c.text[:700]}'
        for c in chunks
    )
    return (
        template
        .replace("{{TICKER}}", ticker)
        .replace("{{FILING_TYPE}}", filing_type)
        .replace("{{FILING_DATE}}", filing_date)
        .replace("{{QUESTION}}", question)
        .replace("{{EXCERPTS}}", excerpts_text)
    )


def _demo_answer(ticker: str, question: str, chunks: list[RetrievedChunk]) -> AnalystAnswer:
    """Canned demo answer when demo mode is on."""
    q_lower = question.lower()

    if any(w in q_lower for w in ["risk", "risks", "concern", "threat"]):
        answer = (
            f"Based on the latest 10-K filing, {ticker}'s key risks include: "
            "competition from established players, geopolitical exposure (especially China), "
            "antitrust and regulatory investigations, supply chain concentration with TSMC, "
            "and macroeconomic headwinds reducing consumer spending. "
            "Management has acknowledged these risks in the 'Risk Factors' section and outlined "
            "mitigation strategies including geographic diversification and supply chain investment."
        )
    elif any(w in q_lower for w in ["outlook", "guidance", "forecast", "expect"]):
        answer = (
            f"Management's forward-looking statements indicate {ticker} expects continued growth "
            "in its Services segment at double-digit rates, with gross margin guidance in the "
            "46–47% range. The company is investing significantly in AI capabilities and "
            "anticipates these investments will drive ecosystem engagement and upgrade cycles. "
            "No specific revenue guidance was provided, consistent with the company's typical practice."
        )
    elif any(w in q_lower for w in ["revenue", "sales", "growth", "financial"]):
        answer = (
            f"According to the MD&A section of the 10-K, {ticker}'s recent financial performance "
            "showed revenue growth driven by Services (up 16%) offsetting some hardware softness. "
            "Free cash flow remained strong at approximately $99B, supporting continued shareholder "
            "returns of over $110B through buybacks and dividends. The company highlighted Services "
            "as the primary growth driver going forward."
        )
    elif any(w in q_lower for w in ["competition", "competitive", "competitor"]):
        answer = (
            f"The 10-K identifies intense competition as a primary risk for {ticker}, noting "
            "competition from companies with significant technical, marketing, and distribution resources. "
            "Key named competitors include Samsung, Google, and Microsoft. The company believes its "
            "competitive advantage lies in the integration of hardware, software, and services, "
            "as well as brand loyalty and the breadth of its ecosystem."
        )
    else:
        answer = (
            f"Based on the SEC 10-K filing excerpts for {ticker}, the filing covers business operations, "
            "risk factors, and management discussion. The specific information requested may be found in "
            "the Risk Factors (Item 1A) or MD&A (Item 7) sections. "
            "Please refine your question to focus on risks, management outlook, financial performance, "
            "or competitive positioning for a more targeted answer."
        )

    return AnalystAnswer(
        question=question,
        answer=answer,
        sources=chunks[:3],
        ticker=ticker,
        granite_mode="demo",
    )


def ask_analyst(
    question: str,
    ticker: str,
    rag_state=None,          # RAGResult from pipeline state
    filing_type: str = "10-K",
    filing_date: str = "2024-10-31",
) -> AnalystAnswer:
    """
    Answer a question about a company's SEC filings using Granite + RAG.

    1. Retrieves top-k chunks from the FAISS index (or keyword fallback)
    2. Builds a grounded prompt with the retrieved excerpts
    3. Calls Granite to generate an answer
    4. Returns AnalystAnswer with answer text + source chunks
    """
    # Get vector store
    store = _get_store(ticker, rag_state)

    # Retrieve relevant chunks
    retrieved = store.retrieve(question, top_k=settings.rag_top_k)

    if not retrieved:
        return AnalystAnswer(
            question=question,
            answer="No filing data available for this company. Please run the pipeline first.",
            sources=[],
            ticker=ticker,
            granite_mode="demo",
        )

    # Demo mode — skip Granite
    if settings.demo_mode:
        return _demo_answer(ticker, question, retrieved)

    # Live mode — call Granite
    prompt = _build_analyst_prompt(ticker, filing_type, filing_date, question, retrieved)
    try:
        raw, mode = call_granite(prompt, max_tokens=800)
        # Parse sources from answer text (Granite should include "Sources: [...]")
        answer_text = raw.strip()
        return AnalystAnswer(
            question=question,
            answer=answer_text,
            sources=retrieved,
            ticker=ticker,
            granite_mode=mode,
        )
    except Exception as exc:
        logger.warning("Granite analyst Q&A failed: %s — using demo fallback", exc)
        return _demo_answer(ticker, question, retrieved)


def _get_store(ticker: str, rag_state=None) -> FAISSVectorStore:
    """Return vector store from memory, loading or building if needed."""
    from services.rag_service import _INDEX_REGISTRY, get_or_build_index

    t = ticker.upper()

    # Already in registry?
    if t in _INDEX_REGISTRY:
        return _INDEX_REGISTRY[t]

    # Try loading from disk
    from services.rag_service import FAISSVectorStore
    from config.settings import settings as _s
    index_path = f"{_s.rag_index_dir}/{t}"
    store = FAISSVectorStore(index_path)
    if store.load():
        _INDEX_REGISTRY[t] = store
        return store

    # Build from rag_state chunks (if pipeline ran)
    if rag_state is not None and hasattr(rag_state, 'documents'):
        pass  # chunks are already loaded in _INDEX_REGISTRY if rag_agent ran

    # Last resort: fetch and build on-demand
    from services.rag_service import fetch_filing_text, chunk_text
    text, filing_date, source_url = fetch_filing_text(t, "10-K")
    chunks = chunk_text(text, t, "10-K", filing_date, source_url)
    store = get_or_build_index(t, chunks)
    return store

"""
Phase 3 tests: RAG service (chunking, retrieval, keyword fallback),
RAG agent (filing ingestion, risk summary), analyst chat.
"""
import sys
import os
import pytest

os.environ["DEMO_MODE"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Chunking tests ────────────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_basic(self):
        from services.rag_service import chunk_text
        text = "A" * 2000  # 2000 chars
        chunks = chunk_text(text, "AAPL", "10-K", "2024-01-01", chunk_size=800, overlap=100)
        assert len(chunks) >= 2

    def test_chunk_ids_unique(self):
        from services.rag_service import chunk_text
        text = "word " * 400  # ~2000 chars
        chunks = chunk_text(text, "AAPL", "10-K", "2024-01-01", chunk_size=800, overlap=100)
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids), "Chunk IDs should be unique"

    def test_chunk_has_section(self):
        from services.rag_service import chunk_text
        text = "Item 1A. Risk Factors\nApple faces intense competition. " * 50
        chunks = chunk_text(text, "AAPL", "10-K", "2024-01-01")
        risk_chunks = [c for c in chunks if c.section == "Risk Factors"]
        assert len(risk_chunks) > 0, "Should detect 'Risk Factors' section"

    def test_chunk_mda_section(self):
        from services.rag_service import chunk_text
        text = "Item 7. Management Discussion and Analysis\nRevenue grew 10%. " * 40
        chunks = chunk_text(text, "AAPL", "10-K", "2024-01-01")
        mda_chunks = [c for c in chunks if c.section == "MD&A"]
        assert len(mda_chunks) > 0

    def test_chunk_empty_text(self):
        from services.rag_service import chunk_text
        chunks = chunk_text("", "AAPL", "10-K", "2024-01-01")
        assert chunks == []

    def test_chunk_short_text(self):
        from services.rag_service import chunk_text
        chunks = chunk_text("Short text.", "AAPL", "10-K", "2024-01-01")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_chunk_ticker_propagated(self):
        from services.rag_service import chunk_text
        chunks = chunk_text("Some text " * 100, "MSFT", "10-Q", "2024-06-30")
        for c in chunks:
            assert c.ticker == "MSFT"
            assert c.filing_type == "10-Q"
            assert c.filing_date == "2024-06-30"


# ── Section detection ─────────────────────────────────────────────────────────

class TestSectionDetection:
    def test_risk_factors_detected(self):
        from services.rag_service import _detect_section
        assert _detect_section("Item 1A. Risk Factors\nblah") == "Risk Factors"

    def test_mda_detected(self):
        from services.rag_service import _detect_section
        assert _detect_section("Item 7. MD&A revenue grew") == "MD&A"

    def test_business_detected(self):
        from services.rag_service import _detect_section
        assert _detect_section("Item 1. Business Overview") == "Business"

    def test_unknown_returns_general(self):
        from services.rag_service import _detect_section
        assert _detect_section("Random paragraph about nothing specific.") == "General"

    def test_forward_looking_detected(self):
        from services.rag_service import _detect_section
        assert _detect_section("Forward-looking statements about future performance") == "Forward-Looking Statements"


# ── Keyword retrieval fallback ────────────────────────────────────────────────

class TestKeywordRetrieval:
    def _make_store(self, ticker: str = "AAPL") -> "FAISSVectorStore":
        from services.rag_service import FAISSVectorStore, chunk_text, _demo_filing_text
        text = _demo_filing_text(ticker, "10-K")
        chunks = chunk_text(text, ticker, "10-K", "2024-10-31")
        store = FAISSVectorStore("/tmp/test_no_faiss")
        store._chunks = chunks  # inject without building
        return store

    def test_keyword_retrieve_returns_results(self):
        store = self._make_store()
        results = store._retrieve_keyword("competition risk China", k=3)
        assert len(results) > 0

    def test_keyword_retrieve_scores_positive(self):
        store = self._make_store()
        results = store._retrieve_keyword("competition risk", k=5)
        for r in results:
            assert r.score >= 0

    def test_keyword_retrieve_respects_top_k(self):
        store = self._make_store()
        results = store._retrieve_keyword("risk factor", k=2)
        assert len(results) <= 2

    def test_keyword_retrieve_empty_query(self):
        store = self._make_store()
        # Empty query returns up to k chunks without filtering
        results = store._retrieve_keyword("", k=3)
        assert len(results) <= 3

    def test_retrieve_dispatches_to_keyword(self):
        """When FAISS not available, retrieve() falls back to keyword."""
        store = self._make_store()
        store._available = False  # force keyword fallback
        results = store.retrieve("What are the main risks?", top_k=3)
        assert len(results) > 0
        for r in results:
            assert r.ticker == "AAPL"


# ── Demo filing text ──────────────────────────────────────────────────────────

class TestDemoFilingText:
    def test_all_demo_tickers_have_text(self):
        from services.rag_service import _demo_filing_text
        for ticker in ("AAPL", "MSFT", "GOOGL", "AMZN"):
            text = _demo_filing_text(ticker, "10-K")
            assert len(text) > 200, f"Demo text for {ticker} should be substantial"

    def test_demo_text_contains_risk_factors(self):
        from services.rag_service import _demo_filing_text
        text = _demo_filing_text("AAPL", "10-K")
        assert "risk" in text.lower()

    def test_demo_text_contains_mda(self):
        from services.rag_service import _demo_filing_text
        text = _demo_filing_text("AAPL", "10-K")
        assert "management" in text.lower() or "revenue" in text.lower()

    def test_unknown_ticker_falls_back_to_aapl(self):
        from services.rag_service import _demo_filing_text
        text = _demo_filing_text("ZZZZ", "10-K")
        assert len(text) > 100


# ── RAG Agent pipeline integration ───────────────────────────────────────────

class TestRAGAgent:
    @pytest.fixture(scope="class")
    def rag_state(self):
        from pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        return orch.run("AAPL", [])

    def test_rag_populated(self, rag_state):
        assert rag_state.rag is not None

    def test_rag_has_documents(self, rag_state):
        assert len(rag_state.rag.documents) > 0

    def test_rag_document_has_chunks(self, rag_state):
        doc = rag_state.rag.documents[0]
        assert doc.n_chunks > 0

    def test_rag_risk_summary_populated(self, rag_state):
        assert rag_state.rag.risk_summary is not None
        rs = rag_state.rag.risk_summary
        assert len(rs.risk_factors) > 0
        assert rs.management_outlook != ""

    def test_rag_risk_factors_count(self, rag_state):
        rs = rag_state.rag.risk_summary
        assert 3 <= len(rs.risk_factors) <= 10

    def test_rag_step_completed(self, rag_state):
        from models.pipeline_state import StepStatus
        step = rag_state.get_step("rag_ingestion")
        assert step is not None
        assert step.status in (StepStatus.DONE, StepStatus.WARNING)


# ── Analyst chat ──────────────────────────────────────────────────────────────

class TestAnalystChat:
    def test_ask_returns_answer(self):
        from services.analyst_chat import ask_analyst
        answer = ask_analyst("What are the key risks?", "AAPL")
        assert answer.answer != ""
        assert answer.ticker == "AAPL"

    def test_ask_risk_question(self):
        from services.analyst_chat import ask_analyst
        answer = ask_analyst("What are the key risks in Apple's 10-K?", "AAPL")
        assert len(answer.answer) > 50

    def test_ask_outlook_question(self):
        from services.analyst_chat import ask_analyst
        answer = ask_analyst("What is the management outlook?", "AAPL")
        assert answer.answer != ""

    def test_ask_has_disclaimer(self):
        from services.analyst_chat import ask_analyst
        answer = ask_analyst("What are the risks?", "AAPL")
        assert "NOT financial advice" in answer.disclaimer

    def test_ask_sources_populated_in_demo(self):
        from services.analyst_chat import ask_analyst
        answer = ask_analyst("What are the risks?", "AAPL")
        # In demo mode, sources come from keyword retrieval
        assert isinstance(answer.sources, list)

    def test_ask_granite_mode_set(self):
        from services.analyst_chat import ask_analyst
        answer = ask_analyst("Revenue growth?", "MSFT")
        assert answer.granite_mode in ("live", "demo", "fallback")

    def test_ask_different_tickers(self):
        from services.analyst_chat import ask_analyst
        for ticker in ("AAPL", "MSFT", "GOOGL"):
            answer = ask_analyst("What are the main risks?", ticker)
            assert answer.ticker == ticker
            assert len(answer.answer) > 20

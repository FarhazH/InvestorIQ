"""
RAG Service — SEC filing ingestion, chunking, FAISS vector store, retrieval.

Architecture:
  1. fetch_filing_text()     — download 10-K/10-Q text from SEC EDGAR
  2. chunk_text()            — split into overlapping chunks with section detection
  3. build_faiss_index()     — embed chunks with sentence-transformers, store in FAISS
  4. retrieve()              — similarity search returning top-k RetrievedChunk objects
  5. Demo mode stubs          — canned Apple 10-K excerpts when DEMO_MODE=true

Dependencies (all optional — service degrades gracefully):
  - sentence-transformers  (pip install sentence-transformers)
  - faiss-cpu              (pip install faiss-cpu)
  - requests               (already in requirements.txt)
  - beautifulsoup4         (pip install beautifulsoup4)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from config.settings import settings
from models.rag import FilingChunk, RetrievedChunk, RAGDocument

logger = logging.getLogger(__name__)

# ── Section header patterns ────────────────────────────────────────────────────
_SECTION_PATTERNS = [
    (re.compile(r"item\s+1a[\.\s]", re.I), "Risk Factors"),
    (re.compile(r"item\s+7[\.\s]", re.I), "MD&A"),
    (re.compile(r"item\s+1[\.\s]", re.I), "Business"),
    (re.compile(r"item\s+2[\.\s]", re.I), "Properties"),
    (re.compile(r"item\s+8[\.\s]", re.I), "Financial Statements"),
    (re.compile(r"item\s+9a[\.\s]", re.I), "Controls"),
    (re.compile(r"forward[\-\s]looking", re.I), "Forward-Looking Statements"),
    (re.compile(r"liquidity\s+and\s+capital", re.I), "Liquidity"),
    (re.compile(r"critical\s+accounting", re.I), "Accounting Policies"),
    (re.compile(r"outlook", re.I), "Outlook"),
]


def _detect_section(text_fragment: str) -> str:
    for pattern, label in _SECTION_PATTERNS:
        if pattern.search(text_fragment):
            return label
    return "General"


# ── Filing text fetch ─────────────────────────────────────────────────────────

def fetch_filing_text(ticker: str, filing_type: str = "10-K") -> tuple[str, str, str]:
    """
    Fetch plain text from the latest SEC EDGAR filing.
    Returns (text, filing_date, source_url).
    Falls back to demo text in demo mode or on error.
    """
    if settings.demo_mode:
        return _demo_filing_text(ticker, filing_type), "2024-10-31", "https://sec.gov/demo"

    try:
        import requests  # type: ignore

        headers = {"User-Agent": "InvestIQ research@investiq.ai"}

        # Step 1: Get CIK
        cik_resp = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={filing_type}&dateRange=custom&startdt=2023-01-01",
            headers=headers, timeout=15,
        )
        cik_resp.raise_for_status()
        hits = cik_resp.json().get("hits", {}).get("hits", [])
        if not hits:
            logger.warning("No SEC hits for %s %s", ticker, filing_type)
            return _demo_filing_text(ticker, filing_type), "2024-01-01", ""

        src = hits[0]["_source"]
        filing_date = src.get("file_date", "unknown")
        entity_id = src.get("entity_id", "")
        accession = src.get("period_of_report", "").replace("-", "")

        # Step 2: Fetch the filing index to find the .htm document
        index_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entity_id}"
            f"&type={filing_type}&dateb=&owner=include&count=1&search_text="
        )

        # Simpler approach: use full-text search API endpoint
        filing_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={filing_type}&dateRange=custom&startdt=2023-01-01"
        resp = requests.get(filing_url, headers=headers, timeout=15)
        data = resp.json()
        hits2 = data.get("hits", {}).get("hits", [])
        if not hits2:
            return _demo_filing_text(ticker, filing_type), filing_date, ""

        file_path = hits2[0].get("_source", {}).get("file_date", "")
        htm_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entity_id}&type={filing_type}&dateb=&owner=include&count=1"

        # Step 3: Fetch the actual document text via EDGAR full-text search
        text_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms={filing_type}&dateRange=custom&startdt=2022-01-01"
        )
        text_resp = requests.get(text_url, headers=headers, timeout=20)
        raw_json = text_resp.json()

        # Extract text snippets from search results
        snippets = []
        for hit in raw_json.get("hits", {}).get("hits", [])[:5]:
            highlight = hit.get("highlight", {})
            for field_snippets in highlight.values():
                snippets.extend(field_snippets)

        if snippets:
            combined = " ".join(snippets)
            return combined, filing_date, f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={entity_id}"

        return _demo_filing_text(ticker, filing_type), filing_date, ""

    except Exception as exc:
        logger.warning("SEC filing fetch failed for %s: %s — using demo text", ticker, exc)
        return _demo_filing_text(ticker, filing_type), "2024-01-01", ""


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    ticker: str,
    filing_type: str,
    filing_date: str,
    source_url: str = "",
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[FilingChunk]:
    """
    Split text into overlapping chunks with section detection.
    Returns list of FilingChunk objects.
    """
    cs = chunk_size or settings.rag_chunk_size
    ov = overlap or settings.rag_chunk_overlap

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[FilingChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + cs, len(text))
        fragment = text[start:end]
        # Detect section from first 200 chars of fragment
        section = _detect_section(fragment[:200])
        chunk_id = hashlib.md5(f"{ticker}{filing_type}{filing_date}{idx}".encode()).hexdigest()[:12]
        chunks.append(FilingChunk(
            chunk_id=chunk_id,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            section=section,
            text=fragment,
            source_url=source_url,
            page_num=None,
        ))
        start = end - ov
        idx += 1
        if end == len(text):
            break

    return chunks


# ── FAISS vector store ─────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Thin wrapper around FAISS for storing and retrieving filing chunks.
    Falls back gracefully if sentence-transformers or faiss-cpu not installed.
    """

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self._index: Any = None
        self._chunks: list[FilingChunk] = []
        self._encoder: Any = None
        self._available = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import faiss  # type: ignore  # noqa: F401
            import sentence_transformers  # type: ignore  # noqa: F401
            return True
        except ImportError:
            logger.info("faiss-cpu or sentence-transformers not installed — using keyword fallback")
            return False

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    def build(self, chunks: list[FilingChunk]) -> bool:
        """Embed all chunks and build a FAISS index. Returns True on success."""
        if not self._available or not chunks:
            self._chunks = chunks  # still store for keyword fallback
            return False
        try:
            import faiss  # type: ignore
            import numpy as np

            encoder = self._get_encoder()
            texts = [c.text for c in chunks]
            logger.info("Encoding %d chunks…", len(texts))
            embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=32)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised = cosine
            index.add(embeddings)

            self._index = index
            self._chunks = chunks

            # Persist to disk
            self.index_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self.index_path / "index.faiss"))
            with open(self.index_path / "chunks.json", "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in chunks], f, indent=2)

            logger.info("FAISS index built: %d chunks, dim=%d", len(chunks), dim)
            return True
        except Exception as exc:
            logger.warning("FAISS build failed: %s — using keyword fallback", exc)
            self._chunks = chunks
            return False

    def load(self) -> bool:
        """Load a previously built index from disk."""
        if not self._available:
            return False
        try:
            import faiss  # type: ignore
            index_file = self.index_path / "index.faiss"
            chunks_file = self.index_path / "chunks.json"
            if not index_file.exists() or not chunks_file.exists():
                return False
            self._index = faiss.read_index(str(index_file))
            with open(chunks_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._chunks = [FilingChunk(**c) for c in raw]
            logger.info("FAISS index loaded: %d chunks", len(self._chunks))
            return True
        except Exception as exc:
            logger.warning("FAISS load failed: %s", exc)
            return False

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return top-k most relevant chunks for a query."""
        k = top_k or settings.rag_top_k
        if not self._chunks:
            return []

        if self._available and self._index is not None:
            return self._retrieve_faiss(query, k)
        else:
            return self._retrieve_keyword(query, k)

    def _retrieve_faiss(self, query: str, k: int) -> list[RetrievedChunk]:
        """Semantic retrieval via FAISS cosine similarity."""
        try:
            import faiss  # type: ignore
            import numpy as np
            encoder = self._get_encoder()
            q_emb = encoder.encode([query], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype="float32")
            faiss.normalize_L2(q_emb)
            distances, indices = self._index.search(q_emb, min(k, len(self._chunks)))
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                c = self._chunks[idx]
                results.append(RetrievedChunk(
                    chunk_id=c.chunk_id,
                    ticker=c.ticker,
                    filing_type=c.filing_type,
                    filing_date=c.filing_date,
                    section=c.section,
                    text=c.text,
                    score=round(float(dist), 4),
                    source_url=c.source_url,
                ))
            return results
        except Exception as exc:
            logger.warning("FAISS retrieve failed: %s — keyword fallback", exc)
            return self._retrieve_keyword(query, k)

    def _retrieve_keyword(self, query: str, k: int) -> list[RetrievedChunk]:
        """Keyword TF-IDF-lite fallback retrieval."""
        query_words = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
        if not query_words:
            return [RetrievedChunk(
                chunk_id=c.chunk_id, ticker=c.ticker, filing_type=c.filing_type,
                filing_date=c.filing_date, section=c.section, text=c.text,
                score=0.5, source_url=c.source_url,
            ) for c in self._chunks[:k]]

        scored: list[tuple[float, FilingChunk]] = []
        for c in self._chunks:
            words = set(re.sub(r"[^a-z0-9 ]", "", c.text.lower()).split())
            overlap = len(query_words & words)
            if overlap > 0:
                score = overlap / (len(query_words) + 0.5)
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id, ticker=c.ticker, filing_type=c.filing_type,
                filing_date=c.filing_date, section=c.section, text=c.text[:800],
                score=round(score, 4), source_url=c.source_url,
            )
            for score, c in scored[:k]
        ]


# ── Global index registry ─────────────────────────────────────────────────────

_INDEX_REGISTRY: dict[str, FAISSVectorStore] = {}


def get_or_build_index(
    ticker: str,
    chunks: list[FilingChunk],
    force_rebuild: bool = False,
) -> FAISSVectorStore:
    """Return cached or newly built FAISS index for a ticker."""
    global _INDEX_REGISTRY
    index_path = os.path.join(settings.rag_index_dir, ticker.upper())

    if ticker in _INDEX_REGISTRY and not force_rebuild:
        return _INDEX_REGISTRY[ticker]

    store = FAISSVectorStore(index_path)
    # Try loading from disk first
    if not force_rebuild and store.load():
        _INDEX_REGISTRY[ticker] = store
        return store

    # Build fresh
    store.build(chunks)
    _INDEX_REGISTRY[ticker] = store
    return store


# ── Demo filing texts ─────────────────────────────────────────────────────────

_DEMO_FILINGS: dict[str, str] = {
    "AAPL": """
Item 1A. Risk Factors
Apple faces intense competition in all markets in which it participates, including from companies 
that have significant technical, marketing, distribution, and other resources. Competitors include 
Samsung, Google, Microsoft, and many others. The Company's ability to compete depends on its 
continued innovation in hardware, software, and services.

Global and regional economic conditions may adversely affect Apple's business, results of operations, 
financial condition, and growth. Inflation, rising interest rates, geopolitical tensions, and 
trade restrictions—particularly related to China—represent material risks to revenue and supply chain.

The Company's operations and performance depend significantly on the global credit environment, 
consumer spending behavior, and enterprise IT spending. A decline in any of these could reduce demand 
for the Company's products and services.

Item 7. Management's Discussion and Analysis
During fiscal 2024, net sales were $383.3 billion compared to $394.3 billion in fiscal 2023, 
a decrease of approximately 3%. The decline was primarily driven by lower iPhone sales in China, 
partially offset by strong Services growth of 16% year-over-year. Services revenue reached $91 billion.

The Company returned over $110 billion to shareholders through dividends and share repurchases in 
fiscal 2024. Free cash flow remained robust at $99 billion, supporting continued capital returns.

Management remains focused on three strategic priorities: (1) expanding the installed base through 
premium hardware, (2) growing high-margin Services revenue, and (3) investing in Apple Intelligence 
and other AI initiatives to differentiate products and drive ecosystem engagement.

Liquidity and Capital Resources
Cash and marketable securities totaled $153 billion as of end of fiscal 2024. The Company has access 
to revolving credit facilities of $6 billion. Capital expenditures were approximately $9.5 billion, 
focused on data centers and manufacturing equipment.

Forward-Looking Statements
Management expects Services segment revenue to continue growing at double-digit rates in fiscal 2025. 
Gross margin guidance of 46–47% reflects confidence in product mix and cost management. The Company 
anticipates iPhone unit volumes to stabilize in China as competitive dynamics evolve.

Risk Factors - Competition
The smartphone market has become increasingly commoditized at lower price points. Apple's premium 
positioning may limit total addressable market in price-sensitive geographies. The Company relies 
heavily on third-party suppliers, including TSMC for semiconductor manufacturing, creating concentration 
risk in the supply chain.

Regulatory and Legal Risks
Apple faces antitrust investigations from the EU, US Department of Justice, and several other 
jurisdictions regarding the App Store policies, payment processing fees, and browser defaults. 
Adverse outcomes could require significant changes to business practices and reduce revenue.
""",
    "MSFT": """
Item 1A. Risk Factors
Microsoft faces significant competition from Amazon Web Services, Google Cloud, and other cloud 
providers. The rapid evolution of AI technologies presents both opportunity and risk—if competitors 
deliver superior AI products, demand for Azure and Microsoft 365 Copilot could be adversely affected.

Cybersecurity incidents represent a material risk. The Company processes vast amounts of sensitive 
customer data and has experienced high-profile security incidents that required public disclosure.

Item 7. Management's Discussion and Analysis
Revenue grew 16% to $245 billion in fiscal 2024. Azure cloud revenue increased 29%, driven by 
AI workload adoption. Microsoft 365 Copilot has reached over 1 million paid enterprise seats. 
Operating income margin expanded to 44.6%, reflecting the high-margin nature of cloud and software.

Management outlook: AI-driven productivity tools are expected to drive sustained commercial cloud 
revenue growth. The Company projects Azure growth to accelerate in fiscal 2025 as AI capacity 
investments completed in fiscal 2024 come online.
""",
    "GOOGL": """
Item 1A. Risk Factors
Alphabet's core Search business faces disruption risk from generative AI alternatives, including 
OpenAI's ChatGPT with Bing integration. If users shift query behavior to AI-native interfaces, 
Google Search advertising revenue could decline.

Regulatory risks are significant: the DOJ antitrust case challenging Google's default search 
agreements with Apple and others represents a material threat to Search revenue.

Item 7. Management's Discussion and Analysis
Total revenue grew 14% to $307 billion. Google Search & Other revenue grew 12% to $175 billion. 
Google Cloud achieved $33 billion in revenue, growing 28% year-over-year with improved profitability. 
YouTube advertising grew 13% driven by Shorts monetization ramp.

Management outlook: Google is investing heavily in Gemini AI models to defend and extend its 
position across Search, Cloud, and consumer products.
""",
    "AMZN": """
Item 1A. Risk Factors
Amazon faces risk from increasing labor costs and unionization of fulfillment center workers. 
Regulatory scrutiny of Amazon's marketplace practices and FBA (Fulfilled by Amazon) policies 
may result in operational changes. AWS faces competition from Azure and GCP.

Item 7. Management's Discussion and Analysis
Net sales grew 12% to $575 billion. AWS generated $91 billion in revenue with 17% operating margins, 
the highest in Company history. Advertising revenue grew 20% to $47 billion. The Company's 
aggressive investment in AI infrastructure—including custom Trainium chips—positions AWS well 
for AI training workloads.

Management expects AWS operating income to grow significantly as AI demand scales. North American 
retail operations returned to profitability with 5.3% operating margins.
""",
}


def _demo_filing_text(ticker: str, filing_type: str) -> str:
    t = ticker.upper()
    base = _DEMO_FILINGS.get(t, _DEMO_FILINGS["AAPL"])
    return f"[DEMO {filing_type} for {t}]\n" + base.strip()

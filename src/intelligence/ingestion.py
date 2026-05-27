import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — Section 5.2
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHUNK_WORDS     = 512
CHUNK_OVERLAP   = 64
CHROMA_PATH     = "models/knowledge_base"
COLLECTION_NAME = "immuniwatch_kb"

# Browser headers — required to avoid 403 from WHO/NPHCDA servers
FETCH_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# Verified document sources — Section 5.1.1
# All URLs verified working in browser as of May 2026
# ---------------------------------------------------------------------------
SOURCES = [
    # ── WHO — General vaccine information ─────────────────────────
    {
        "name":     "WHO Vaccines and Immunization",
        "url":      "https://www.who.int/health-topics/vaccines-and-immunization",
        "language": "en",
    },
    {
        "name":     "WHO Vaccine Safety Q&A",
        "url":      "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety",
        "language": "en",
    },
    {
        "name":     "WHO Immunization Coverage",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/immunization-coverage",
        "language": "en",
    },

    # ── WHO — Specific vaccines ───────────────────────────────────
    {
        "name":     "WHO HPV and Cervical Cancer",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/human-papillomavirus-(hpv)-and-cervical-cancer",
        "language": "en",
    },
    {
        "name":     "WHO Measles",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/measles",
        "language": "en",
    },
    {
        "name":     "WHO Yellow Fever",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/yellow-fever",
        "language": "en",
    },
    {
        "name":     "WHO Meningitis",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/meningococcal-meningitis",
        "language": "en",
    },
    {
        "name":     "WHO Polio",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/poliomyelitis",
        "language": "en",
    },
    {
        "name":     "WHO Tetanus",
        "url":      "https://www.who.int/news-room/fact-sheets/detail/tetanus",
        "language": "en",
    },
    {
        "name":     "WHO COVID-19 Vaccines",
        "url":      "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/covid-19-vaccines",
        "language": "en",
    },

    # ── WHO Africa — Nigeria specific ────────────────────────────
    {
        "name":     "WHO Africa Nigeria",
        "url":      "https://www.afro.who.int/countries/nigeria",
        "language": "en",
    },
    {
        "name":     "WHO Africa Nigeria Vaccine Campaign",
        "url":      "https://www.afro.who.int/countries/nigeria/news/nigeria-intensifies-fight-against-vaccine-preventable-diseases-nationwide-measles-rubella-and-polio",
        "language": "en",
    },

    # ── NPHCDA Nigeria ────────────────────────────────────────────
    {
        "name":     "NPHCDA Nigeria Homepage",
        "url":      "https://nphcda.gov.ng/",
        "language": "en",
    },
    {
        "name":     "NPHCDA Integrated Vaccine Campaign",
        "url":      "https://nphcda.gov.ng/protecting-the-future-nigeria-launches-integrated-campaign-against-measles-rubella-human-papillomavirus-hpv-polio-and-neglected-tropical-diseases/",
        "language": "en",
    },

    # ── UNICEF Nigeria ────────────────────────────────────────────
  
    {
        "name":     "UNICEF Nigeria Health and HIV",
        "url":      "https://www.unicef.org/nigeria/health-hiv",
        "language": "en",
    },
]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def _fetch_text(url: str) -> Optional[str]:
    """Fetch and extract clean text from a URL."""
    try:
        resp = requests.get(
            url,
            headers=FETCH_HEADERS,
            timeout=15,
            allow_redirects=True,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 100 else None

    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Chunking — Section 5.1.2
# ---------------------------------------------------------------------------
def _chunk_text(text: str, source_name: str,
                source_url: str, language: str) -> List[dict]:
    """Split text into overlapping chunks."""
    words  = text.split()
    chunks = []
    step   = CHUNK_WORDS - CHUNK_OVERLAP
    idx    = 0

    while idx < len(words):
        chunk_text = " ".join(words[idx: idx + CHUNK_WORDS])
        if len(chunk_text.strip()) > 50:
            chunks.append({
                "text":        chunk_text,
                "source":      source_name,
                "url":         source_url,
                "language":    language,
                "chunk_index": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })
        idx += step

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------
def _get_collection():
    """Get or create ChromaDB collection with multilingual-e5-large."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------
def ingest_all() -> None:
    """Crawl all sources and store chunks in ChromaDB."""
    log.info("=" * 55)
    log.info("ImmuniWatch — Knowledge Base Ingestion")
    log.info("Sources:         %d", len(SOURCES))
    log.info("Embedding model: %s", EMBEDDING_MODEL)
    log.info("Storage:         %s", CHROMA_PATH)
    log.info("=" * 55)

    collection      = _get_collection()
    total_chunks    = 0
    total_documents = 0

    for source in SOURCES:
        log.info("Fetching: %s", source["name"])
        text = _fetch_text(source["url"])

        if not text:
            log.warning("  Skipped — could not fetch content")
            continue

        chunks = _chunk_text(
            text=        text,
            source_name= source["name"],
            source_url=  source["url"],
            language=    source["language"],
        )

        if not chunks:
            log.warning("  Skipped — no chunks produced")
            continue

        ids       = [f"{source['name']}_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{k: v for k, v in c.items() if k != "text"}
                     for c in chunks]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        log.info("  Stored %d chunks", len(chunks))
        total_chunks    += len(chunks)
        total_documents += 1

    log.info("=" * 55)
    log.info("Ingestion complete")
    log.info("  Documents: %d", total_documents)
    log.info("  Chunks:    %d", total_chunks)
    log.info("  Location:  %s", CHROMA_PATH)
    log.info("=" * 55)


if __name__ == "__main__":
    ingest_all()
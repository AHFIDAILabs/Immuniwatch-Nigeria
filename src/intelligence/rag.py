import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — Section 5.2 and 5.3
# ---------------------------------------------------------------------------
EMBEDDING_MODEL    = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHROMA_PATH        = "models/knowledge_base"
COLLECTION_NAME    = "immuniwatch_kb"
TOP_K              = 5       # top-5 per system design Section 5.3
SIMILARITY_THRESHOLD = 0.72  # Section 5.2 — cosine threshold


# ---------------------------------------------------------------------------
# Evidence record — Section 5.3.1
# ---------------------------------------------------------------------------
class EvidenceRecord:
    """
    Structured evidence record returned for a misinformation post.
    Matches system design Section 5.3.1 EvidenceRecord schema.
    """

    def __init__(
        self,
        source_title:    str,
        source_url:      str,
        snippet:         str,
        similarity:      float,
        language:        str = "en",
    ):
        self.source_title = source_title
        self.source_url   = source_url
        self.snippet      = snippet
        self.similarity   = round(similarity, 4)
        self.language     = language

    def to_dict(self) -> dict:
        return {
            "source_title": self.source_title,
            "source_url":   self.source_url,
            "snippet":      self.snippet[:300],  # truncate for API response
            "similarity":   self.similarity,
            "language":     self.language,
        }


# ---------------------------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------------------------
class RAGRetriever:
    """
    Retrieves verified health facts from ChromaDB knowledge base.
    Uses multilingual-e5-large for cross-language matching —
    a Hausa claim will match an English WHO fact correctly.

    Must call ingest_all() from ingestion.py first to populate
    the knowledge base before retrieval works.
    """

    def __init__(self):
        self._collection = None
        self._ready      = False
        self._init()

    def _init(self) -> None:
        """Connect to ChromaDB. Fails gracefully if not yet ingested."""
        kb_path = Path(CHROMA_PATH)
        if not kb_path.exists():
            log.warning(
                "Knowledge base not found at %s. "
                "Run: python -m src.intelligence.ingestion",
                CHROMA_PATH,
            )
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=CHROMA_PATH)
            ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                device="cpu",
            )
            self._collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
            count = self._collection.count()
            self._ready = count > 0
            log.info(
                "RAG ready — %d chunks in knowledge base", count
            )
        except Exception as e:
            log.error("RAG initialisation failed: %s", e)

    def is_ready(self) -> bool:
        return self._ready

    def retrieve(
        self,
        post_text: str,
        language:  Optional[str] = None,
    ) -> List[EvidenceRecord]:
        """
        Retrieve top-5 verified facts most relevant to the post.

        Args:
            post_text: The misinformation post content.
            language:  Post language code (en/pcm/ha/yo/ig).
                       multilingual-e5-large handles cross-language
                       matching automatically.

        Returns:
            List of EvidenceRecord objects above similarity threshold.
            Empty list if knowledge base is not ready or no match found.
        """
        if not self._ready:
            log.warning("RAG not ready — returning empty evidence")
            return []

        if not post_text or len(post_text.strip()) < 5:
            return []

        try:
            # multilingual-e5-large requires "query: " prefix for queries
            query = f"query: {post_text.strip()}"

            results = self._collection.query(
                query_texts=[query],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )

            evidence = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, distance in zip(documents, metadatas, distances):
                # ChromaDB returns cosine distance — convert to similarity
                # similarity = 1 - distance
                similarity = 1.0 - distance

                if similarity < SIMILARITY_THRESHOLD:
                    continue

                evidence.append(EvidenceRecord(
                    source_title= meta.get("source", "Unknown"),
                    source_url=   meta.get("url", ""),
                    snippet=      doc,
                    similarity=   similarity,
                    language=     meta.get("language", "en"),
                ))

            log.debug(
                "RAG retrieved %d evidence records for post (threshold=%.2f)",
                len(evidence), SIMILARITY_THRESHOLD,
            )
            return evidence

        except Exception as e:
            log.error("RAG retrieval failed: %s", e)
            return []

    def retrieve_as_dicts(
        self,
        post_text: str,
        language:  Optional[str] = None,
    ) -> List[dict]:
        """
        Same as retrieve() but returns plain dicts.
        Used by FastAPI routes to serialise response.
        """
        return [e.to_dict() for e in self.retrieve(post_text, language)]


# ---------------------------------------------------------------------------
# Direct embedding — used by POST /embed and POST /embed/batch
# Model: multilingual-e5-base (768-dimensional output).
# The ML service spec float[768] is authoritative. -large was a typo in the
# system design doc (Section 5.2). Backend expects exactly 768 dimensions.
# ---------------------------------------------------------------------------
_st_model = None
_st_lock  = threading.Lock()


def preload_embedder() -> None:
    """Called at startup to warm the SentenceTransformer before first request."""
    try:
        _get_st_model()
        log.info("Embedding model loaded and ready.")
    except Exception as exc:
        log.warning("Embedding model failed to preload: %s", exc)


def _get_st_model():
    """Lazy singleton — loads multilingual-e5-large once and reuses it."""
    global _st_model
    if _st_model is not None:
        return _st_model
    with _st_lock:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading SentenceTransformer: %s", EMBEDDING_MODEL)
            _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


def is_embedder_ready() -> bool:
    return _st_model is not None


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.
    Prefix 'query: ' is required by multilingual-e5 models for query inputs.
    Returns list of 768 floats (multilingual-e5-base).
    """
    model = _get_st_model()
    vec   = model.encode(f"query: {text}", normalize_embeddings=True)
    return vec.tolist()


def embed_batch(items: List[dict]) -> List[dict]:
    """
    Embed multiple items.
    items: list of {"doc_id": str, "text": str}
    Returns: list of {"doc_id": str, "embedding": list[float]}
    """
    model   = _get_st_model()
    texts   = [f"query: {it['text']}" for it in items]
    vecs    = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [
        {"doc_id": item["doc_id"], "embedding": vec.tolist()}
        for item, vec in zip(items, vecs)
    ]
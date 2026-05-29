"""
One-shot script: rebuild knowledge base with multilingual-e5-base (768-dim).

Reads all 56 documents from the existing 1024-dim collection,
deletes it, and recreates it with the correct 768-dim model.
No web scraping — preserves all existing content.

Usage:
    python scripts/rebuild_kb.py
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHROMA_PATH     = "models/knowledge_base"
COLLECTION_NAME = "immuniwatch_kb"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def rebuild() -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # ── Step 1: read everything out of the old collection ───────────
    try:
        old = client.get_collection(name=COLLECTION_NAME)
        total = old.count()
        log.info("Reading %d documents from existing collection ...", total)

        # fetch in batches of 100
        batch_size = 100
        all_ids, all_docs, all_metas = [], [], []
        offset = 0
        while offset < total:
            batch = old.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_ids   += batch["ids"]
            all_docs  += batch["documents"]
            all_metas += batch["metadatas"]
            offset    += len(batch["ids"])
            log.info("  read %d / %d ...", len(all_ids), total)

        log.info("Extracted %d documents.", len(all_ids))
    except Exception as e:
        log.error("Could not read existing collection: %s", e)
        sys.exit(1)

    # ── Step 2: delete old collection ───────────────────────────────
    try:
        client.delete_collection(name=COLLECTION_NAME)
        log.info("Old collection deleted.")
    except Exception as e:
        log.error("Could not delete old collection: %s", e)
        sys.exit(1)

    # ── Step 3: create new collection with -base (768-dim) ──────────
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )
    new_col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("New collection created with %s.", EMBEDDING_MODEL)

    # ── Step 4: re-insert (ChromaDB re-embeds automatically) ────────
    chunk = 50
    for i in range(0, len(all_ids), chunk):
        new_col.upsert(
            ids=       all_ids[i:i+chunk],
            documents= all_docs[i:i+chunk],
            metadatas= all_metas[i:i+chunk],
        )
        log.info("  inserted %d / %d ...", min(i + chunk, len(all_ids)), len(all_ids))

    final_count = new_col.count()
    log.info("=" * 50)
    log.info("Rebuild complete — %d documents in new collection.", final_count)
    log.info("Model: %s (768-dim)", EMBEDDING_MODEL)
    log.info("=" * 50)


if __name__ == "__main__":
    # make src importable from project root
    sys.path.insert(0, str(Path(__file__).parent.parent))
    rebuild()

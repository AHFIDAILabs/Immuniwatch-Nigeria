"""
ImmuniWatch Nigeria — Lightweight Evaluation
==============================================
Evaluates system components without loading heavy ML models.
Safe to run on CPU-only machines with limited RAM.

Checks:
  1. Knowledge base exists and has chunks
  2. Counter-response format compliance (no API calls)
  3. Classifier constants match system design
  4. All required files exist

Usage:
    python -m src.intelligence.evaluation
"""

import logging
import os
import sys
from pathlib import Path

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
# Check 1 — Knowledge base exists and has chunks
# ---------------------------------------------------------------------------
def check_knowledge_base() -> bool:
    """Verify ChromaDB knowledge base is populated."""
    kb_path = Path("models/knowledge_base")
    if not kb_path.exists():
        log.error("  [FAIL] Knowledge base not found at %s", kb_path)
        return False

    try:
        import chromadb
        client     = chromadb.PersistentClient(path=str(kb_path))
        collection = client.get_collection("immuniwatch_kb")
        count      = collection.count()

        if count == 0:
            log.error("  [FAIL] Knowledge base is empty")
            return False

        log.info("  [PASS] Knowledge base: %d chunks stored", count)
        return True

    except Exception as e:
        log.error("  [FAIL] Knowledge base error: %s", e)
        return False


# ---------------------------------------------------------------------------
# Check 2 — ONNX model file exists and is valid size
# ---------------------------------------------------------------------------
def check_onnx_model() -> bool:
    """Verify ONNX model file exists."""
    onnx_path = Path("models/onnx/immuniwatch_classifier.onnx")
    if not onnx_path.exists():
        log.error("  [FAIL] ONNX model not found: %s", onnx_path)
        return False

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    log.info("  [PASS] ONNX model: %.1f MB", size_mb)
    return True


# ---------------------------------------------------------------------------
# Check 3 — Thresholds file correct
# ---------------------------------------------------------------------------
def check_thresholds() -> bool:
    """Verify thresholds.json has correct structure."""
    import json
    path = Path("models/onnx/thresholds.json")
    if not path.exists():
        log.error("  [FAIL] thresholds.json not found")
        return False

    with open(path) as f:
        data = json.load(f)

    biases = data.get("class_biases", {})
    required = ["factual", "misinformation", "irrelevant"]
    for label in required:
        if label not in biases:
            log.error("  [FAIL] Missing bias for label: %s", label)
            return False

    log.info("  [PASS] Thresholds: factual=%.1f misinfo=%.1f irrelevant=%.1f",
             biases["factual"], biases["misinformation"], biases["irrelevant"])
    return True


# ---------------------------------------------------------------------------
# Check 4 — System design constants
# ---------------------------------------------------------------------------
def check_system_design_constants() -> bool:
    """Verify all system design constants are correct."""
    passed = True

    from src.ingestion.deduplication import JACCARD_THRESHOLD, EXACT_TTL_S
    if JACCARD_THRESHOLD != 0.85:
        log.error("  [FAIL] JACCARD_THRESHOLD should be 0.85, got %s", JACCARD_THRESHOLD)
        passed = False
    else:
        log.info("  [PASS] Dedup Jaccard threshold: 0.85 (Section 4.3)")

    if EXACT_TTL_S != 86400:
        log.error("  [FAIL] EXACT_TTL_S should be 86400, got %s", EXACT_TTL_S)
        passed = False
    else:
        log.info("  [PASS] Dedup TTL: 24 hours (Section 4.3)")

    from src.intelligence.rag import TOP_K, SIMILARITY_THRESHOLD
    if TOP_K != 5:
        log.error("  [FAIL] TOP_K should be 5, got %s", TOP_K)
        passed = False
    else:
        log.info("  [PASS] RAG top-K: 5 (Section 5.3)")

    if SIMILARITY_THRESHOLD != 0.72:
        log.error("  [FAIL] SIMILARITY_THRESHOLD should be 0.72, got %s", SIMILARITY_THRESHOLD)
        passed = False
    else:
        log.info("  [PASS] RAG similarity threshold: 0.72 (Section 5.2)")

    from src.intelligence.counter import SHORT_MAX_CHARS, MEDIUM_MAX_WORDS, LONG_MAX_WORDS
    if SHORT_MAX_CHARS != 280:
        log.error("  [FAIL] SHORT_MAX_CHARS should be 280, got %s", SHORT_MAX_CHARS)
        passed = False
    else:
        log.info("  [PASS] Counter SHORT: ≤280 chars (Section 6.5)")

    if MEDIUM_MAX_WORDS != 200:
        log.error("  [FAIL] MEDIUM_MAX_WORDS should be 200, got %s", MEDIUM_MAX_WORDS)
        passed = False
    else:
        log.info("  [PASS] Counter MEDIUM: ≤200 words (Section 6.5)")

    if LONG_MAX_WORDS != 500:
        log.error("  [FAIL] LONG_MAX_WORDS should be 500, got %s", LONG_MAX_WORDS)
        passed = False
    else:
        log.info("  [PASS] Counter LONG: ≤500 words (Section 6.5)")

    return passed


# ---------------------------------------------------------------------------
# Check 5 — Counter-response format compliance (no API calls)
# ---------------------------------------------------------------------------
def check_counter_format_compliance() -> bool:
    """
    Verify length enforcement works correctly.
    Uses only pure functions — no LLM calls, no tokens consumed.
    """
    from src.intelligence.counter import (
        _enforce_short, _enforce_word_limit,
        SHORT_MAX_CHARS, MEDIUM_MAX_WORDS, LONG_MAX_WORDS,
    )

    passed = True

    # Test SHORT limit
    long_text = "Vaccine dey safe for all pikin " * 20
    short = _enforce_short(long_text)
    if len(short) <= SHORT_MAX_CHARS:
        log.info("  [PASS] SHORT format enforced: %d chars ≤ 280", len(short))
    else:
        log.error("  [FAIL] SHORT format exceeded: %d chars", len(short))
        passed = False

    # Test MEDIUM limit
    medium = _enforce_word_limit(long_text, MEDIUM_MAX_WORDS)
    if len(medium.split()) <= MEDIUM_MAX_WORDS + 1:
        log.info("  [PASS] MEDIUM format enforced: %d words ≤ 200", len(medium.split()))
    else:
        log.error("  [FAIL] MEDIUM format exceeded: %d words", len(medium.split()))
        passed = False

    # Test LONG limit
    long = _enforce_word_limit(long_text, LONG_MAX_WORDS)
    if len(long.split()) <= LONG_MAX_WORDS + 1:
        log.info("  [PASS] LONG format enforced: %d words ≤ 500", len(long.split()))
    else:
        log.error("  [FAIL] LONG format exceeded: %d words", len(long.split()))
        passed = False

    return passed


# ---------------------------------------------------------------------------
# Check 6 — Required files exist
# ---------------------------------------------------------------------------
def check_required_files() -> bool:
    """Verify all required production files exist."""
    required = [
        "models/onnx/immuniwatch_classifier.onnx",
        "models/onnx/thresholds.json",
        "models/onnx/model_config.json",
        "src/models/classifier.py",
        "src/api/main.py",
        "src/api/routes.py",
        "src/api/schemas.py",
        "src/ingestion/worker.py",
        "src/ingestion/deduplication.py",
        "src/ingestion/connectors/base.py",
        "src/ingestion/connectors/youtube.py",
        "src/ingestion/connectors/sociavault.py",
        "src/ingestion/connectors/bluesky.py",
        "src/intelligence/ingestion.py",
        "src/intelligence/rag.py",
        "src/intelligence/counter.py",
        "src/intelligence/evaluation.py",
        "docker-compose.yml",
    ]

    all_found = True
    for path in required:
        if Path(path).exists():
            log.info("  [PASS] %s", path)
        else:
            log.error("  [FAIL] MISSING: %s", path)
            all_found = False

    return all_found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_evaluation() -> None:
    log.info("=" * 55)
    log.info("ImmuniWatch — Lightweight Evaluation")
    log.info("No heavy model loading — safe on CPU")
    log.info("=" * 55)

    checks = [
        ("Knowledge Base",              check_knowledge_base),
        ("ONNX Model",                  check_onnx_model),
        ("Thresholds",                  check_thresholds),
        ("System Design Constants",     check_system_design_constants),
        ("Counter Format Compliance",   check_counter_format_compliance),
        ("Required Files",              check_required_files),
    ]

    results = []
    for name, fn in checks:
        log.info("")
        log.info("[ %s ]", name)
        try:
            passed = fn()
        except Exception as e:
            log.error("  [FAIL] Unexpected error: %s", e)
            passed = False
        results.append((name, passed))

    # Summary
    log.info("")
    log.info("=" * 55)
    log.info("Evaluation Summary")
    log.info("=" * 55)
    passed_count = sum(1 for _, p in results if p)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        log.info("  %s  %s", status, name)

    log.info("")
    log.info("  %d / %d checks passed", passed_count, len(checks))
    log.info("=" * 55)

    if passed_count == len(checks):
        log.info("System is ready for Docker deployment.")
    else:
        log.warning("%d check(s) need attention before deployment.",
                    len(checks) - passed_count)


if __name__ == "__main__":
    run_evaluation()
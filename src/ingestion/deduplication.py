import hashlib
import logging
import re
import time
from threading import Lock
from typing import Optional, Set

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — from system design Section 4.3
# ---------------------------------------------------------------------------
JACCARD_THRESHOLD = 0.85    # near-duplicate similarity threshold
NUM_PERMUTATIONS  = 128     # MinHash accuracy vs speed balance
SHINGLE_SIZE      = 3       # character shingles for MinHash
EXACT_TTL_S       = 86400   # 24 hours — per system design spec


# ---------------------------------------------------------------------------
# Text normalisation
# System design: "lowercased, punctuation stripped, URLs removed"
# ---------------------------------------------------------------------------
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _shingles(text: str) -> Set[str]:
    """Character k-shingles for MinHash input."""
    return {text[i:i + SHINGLE_SIZE]
            for i in range(max(1, len(text) - SHINGLE_SIZE + 1))}


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------
class Deduplicator:
    """
    Two-layer deduplication per system design Section 4.3.
    Thread-safe — safe to call from multiple connector threads.
    """

    def __init__(self):
        self._lock = Lock()

        # Layer 1: {sha256_hex: inserted_timestamp}
        self._exact: dict = {}

        # Layer 2: MinHash LSH (optional — degrades gracefully)
        self._lsh             = None
        self._lsh_available   = False
        self._init_lsh()

    # ── Initialisation ───────────────────────────────────────────

    def _init_lsh(self) -> None:
        """
        Initialise MinHash LSH index.
        If datasketch is not installed, Layer 2 is skipped and
        only exact deduplication runs — no crash, just a warning.
        """
        try:
            from datasketch import MinHashLSH
            self._lsh           = MinHashLSH(
                threshold=JACCARD_THRESHOLD,
                num_perm=NUM_PERMUTATIONS,
            )
            self._lsh_available = True
            log.info(
                "Deduplicator ready — exact + MinHash LSH "
                "(threshold=%.2f)", JACCARD_THRESHOLD
            )
        except ImportError:
            log.warning(
                "datasketch not installed — running exact deduplication only. "
                "Install with: pip install datasketch"
            )

    # ── Public API ───────────────────────────────────────────────

    def is_duplicate(self, post_id: str, text: str) -> bool:
        """
        Check whether this post is a duplicate.

        Args:
            post_id: Unique post identifier from the platform.
            text:    Raw post text content.

        Returns:
            True  — post is a duplicate, discard it.
            False — post is new, safe to publish to Kafka.
        """
        if not text or len(text.strip()) < 5:
            return True

        normalised = _normalise(text)

        with self._lock:
            self._cleanup_expired()

            # Layer 1 — exact hash check
            text_hash = hashlib.sha256(normalised.encode()).hexdigest()
            if text_hash in self._exact:
                log.debug("Exact duplicate discarded: post_id=%s", post_id)
                return True
            self._exact[text_hash] = time.time()

            # Layer 2 — near-duplicate check
            if self._lsh_available:
                if self._check_near_duplicate(post_id, normalised):
                    log.debug(
                        "Near-duplicate discarded: post_id=%s", post_id
                    )
                    return True

        return False

    # ── Internal helpers ─────────────────────────────────────────

    def _check_near_duplicate(self, post_id: str, normalised: str) -> bool:
        """
        Run MinHash LSH query.
        Returns True if a near-duplicate exists.
        Inserts the post into the LSH index if it is new.
        """
        from datasketch import MinHash

        minhash = MinHash(num_perm=NUM_PERMUTATIONS)
        for shingle in _shingles(normalised):
            minhash.update(shingle.encode())

        results = self._lsh.query(minhash)
        if results:
            return True

        # Insert as new — use post_id as key
        try:
            self._lsh.insert(post_id, minhash)
        except ValueError:
            # post_id already in index — harmless
            pass

        return False

    def _cleanup_expired(self) -> None:
        """
        Remove exact hashes older than 24 hours.
        Called on every is_duplicate() call inside the lock.
        Keeps memory bounded over long running periods.
        """
        cutoff = time.time() - EXACT_TTL_S
        expired = [h for h, ts in self._exact.items() if ts < cutoff]
        for h in expired:
            del self._exact[h]
        if expired:
            log.debug("Cleaned up %d expired hashes", len(expired))
from src.ingestion.deduplication import (
    Deduplicator,
    _normalise,
    _shingles,
    JACCARD_THRESHOLD,
    EXACT_TTL_S,
    NUM_PERMUTATIONS,
    SHINGLE_SIZE,
)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

def test_jaccard_threshold_matches_system_design():
    """System design Section 4.3 specifies 0.85."""
    assert JACCARD_THRESHOLD == 0.85


def test_exact_ttl_is_24_hours():
    """System design Section 4.3 specifies 24-hour TTL."""
    assert EXACT_TTL_S == 86400


def test_num_permutations():
    assert NUM_PERMUTATIONS == 128


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

def test_normalise_lowercases():
    assert _normalise("VACCINE") == "vaccine"


def test_normalise_removes_urls():
    result = _normalise("Check this https://example.com for info")
    assert "https" not in result
    assert "example" not in result


def test_normalise_removes_punctuation():
    result = _normalise("vaccine! kills? people.")
    assert "!" not in result
    assert "?" not in result
    assert "." not in result


def test_normalise_collapses_whitespace():
    result = _normalise("vaccine   kills   people")
    assert "  " not in result


def test_normalise_strips_whitespace():
    result = _normalise("  vaccine Nigeria  ")
    assert result == result.strip()


# ---------------------------------------------------------------------------
# _shingles
# ---------------------------------------------------------------------------

def test_shingles_returns_set():
    result = _shingles("vaccine")
    assert isinstance(result, set)


def test_shingles_not_empty_for_valid_text():
    result = _shingles("vaccine Nigeria")
    assert len(result) > 0


def test_shingles_uses_correct_size():
    text = "abcdefg"
    result = _shingles(text)
    for shingle in result:
        assert len(shingle) == SHINGLE_SIZE


# ---------------------------------------------------------------------------
# Deduplicator — exact duplicates
# ---------------------------------------------------------------------------

def test_exact_duplicate_is_caught():
    dedup = Deduplicator()
    text = "Vaccine dey kill pikin for Lagos"
    assert dedup.is_duplicate("post-001", text) is False  # first time — new
    assert dedup.is_duplicate("post-002", text) is True   # second time — duplicate


def test_unique_posts_pass_through():
    dedup = Deduplicator()
    assert dedup.is_duplicate("post-001", "Vaccine causes infertility") is False
    assert dedup.is_duplicate("post-002", "Vaccines are safe for children") is False
    assert dedup.is_duplicate("post-003", "NPHCDA launches immunization campaign") is False


def test_empty_text_is_duplicate():
    """Empty posts should be discarded."""
    dedup = Deduplicator()
    assert dedup.is_duplicate("post-001", "") is True
    assert dedup.is_duplicate("post-002", "   ") is True


def test_very_short_text_is_duplicate():
    """Posts shorter than 5 characters should be discarded."""
    dedup = Deduplicator()
    assert dedup.is_duplicate("post-001", "ok") is True


def test_same_text_different_case_is_duplicate():
    """Case differences should not bypass deduplication."""
    dedup = Deduplicator()
    assert dedup.is_duplicate("post-001", "Vaccine kills children") is False
    assert dedup.is_duplicate("post-002", "VACCINE KILLS CHILDREN") is True


def test_same_text_different_urls_is_duplicate():
    """URLs are stripped before hashing."""
    dedup = Deduplicator()
    text1 = "Vaccine kills children https://link1.com"
    text2 = "Vaccine kills children https://link2.com"
    assert dedup.is_duplicate("post-001", text1) is False
    assert dedup.is_duplicate("post-002", text2) is True


# ---------------------------------------------------------------------------
# Deduplicator — thread safety (basic check)
# ---------------------------------------------------------------------------

def test_deduplicator_is_instantiable():
    dedup = Deduplicator()
    assert dedup is not None


def test_deduplicator_has_lock():
    dedup = Deduplicator()
    assert hasattr(dedup, "_lock")


def test_deduplicator_exact_store_starts_empty():
    dedup = Deduplicator()
    assert len(dedup._exact) == 0
import pytest
from src.intelligence.rag import (
    EvidenceRecord,
    RAGRetriever,
    TOP_K,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

def test_top_k_matches_system_design():
    """System design Section 5.3 specifies top-5."""
    assert TOP_K == 5


def test_similarity_threshold_matches_system_design():
    """System design Section 5.2 specifies 0.72."""
    assert SIMILARITY_THRESHOLD == 0.72


def test_embedding_model_is_multilingual_e5():
    """ML spec float[768] is authoritative — e5-base, not e5-large."""
    assert EMBEDDING_MODEL == "intfloat/multilingual-e5-base"


# ---------------------------------------------------------------------------
# EvidenceRecord
# ---------------------------------------------------------------------------

def test_evidence_record_creation():
    record = EvidenceRecord(
        source_title="WHO Vaccine Safety",
        source_url="https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety",
        snippet="Vaccines are safe and effective. Side effects are usually mild.",
        similarity=0.89,
        language="en",
    )
    assert record.source_title == "WHO Vaccine Safety"
    assert record.similarity == 0.89
    assert record.language == "en"


def test_evidence_record_similarity_rounded():
    record = EvidenceRecord(
        source_title="WHO",
        source_url="https://who.int",
        snippet="Vaccines are safe",
        similarity=0.876543,
    )
    assert record.similarity == 0.8765


def test_evidence_record_to_dict():
    record = EvidenceRecord(
        source_title="WHO Measles",
        source_url="https://www.who.int/news-room/fact-sheets/detail/measles",
        snippet="The measles vaccine is safe and highly effective.",
        similarity=0.91,
    )
    d = record.to_dict()
    assert "source_title" in d
    assert "source_url" in d
    assert "snippet" in d
    assert "similarity" in d
    assert "language" in d


def test_evidence_record_snippet_truncated_to_300():
    long_snippet = "x" * 500
    record = EvidenceRecord(
        source_title="WHO",
        source_url="https://who.int",
        snippet=long_snippet,
        similarity=0.80,
    )
    d = record.to_dict()
    assert len(d["snippet"]) <= 300


def test_evidence_record_default_language_is_en():
    record = EvidenceRecord(
        source_title="WHO",
        source_url="https://who.int",
        snippet="Vaccines are safe",
        similarity=0.85,
    )
    assert record.language == "en"


# ---------------------------------------------------------------------------
# RAGRetriever — graceful degradation
# ---------------------------------------------------------------------------

def test_rag_retriever_instantiates():
    """RAGRetriever should instantiate even if KB not populated."""
    retriever = RAGRetriever()
    assert retriever is not None


def test_rag_retriever_is_ready_reflects_kb_state():
    """is_ready() should return bool — True or False depending on KB."""
    retriever = RAGRetriever()
    assert isinstance(retriever.is_ready(), bool)


def test_rag_retriever_returns_empty_list_for_empty_text():
    """Empty text should return empty evidence."""
    retriever = RAGRetriever()
    result = retriever.retrieve("")
    assert result == []


def test_rag_retriever_returns_empty_list_for_short_text():
    retriever = RAGRetriever()
    result = retriever.retrieve("ok")
    assert result == []


def test_rag_retrieve_as_dicts_returns_list():
    retriever = RAGRetriever()
    result = retriever.retrieve_as_dicts("vaccine causes infertility")
    assert isinstance(result, list)


def test_rag_retrieve_as_dicts_items_are_dicts():
    retriever = RAGRetriever()
    results = retriever.retrieve_as_dicts("vaccine causes infertility")
    for item in results:
        assert isinstance(item, dict)
        if item:
            assert "source_title" in item
            assert "snippet" in item
            assert "similarity" in item
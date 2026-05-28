import pytest
from pydantic import ValidationError
from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    Alternative,
    BatchClassifyRequest,
    FeedbackRequest,
    RetrainRequest,
    EmbedRequest,
)


# ---------------------------------------------------------------------------
# ClassifyRequest
# ---------------------------------------------------------------------------

def test_classify_request_valid():
    req = ClassifyRequest(
        post_id="test-001",
        content="Vaccine causes infertility in women",
        platform="twitter",
    )
    assert req.post_id == "test-001"
    assert req.platform == "twitter"
    assert req.language is None
    assert req.kb_snippets == []


def test_classify_request_with_all_fields():
    req = ClassifyRequest(
        post_id="test-002",
        content="Rigakafi yana haddasa matsala",
        language="ha",
        platform="facebook",
        context="Some context",
        kb_snippets=["WHO says vaccines are safe"],
    )
    assert req.language == "ha"
    assert len(req.kb_snippets) == 1


def test_classify_request_missing_post_id_fails():
    with pytest.raises(ValidationError):
        ClassifyRequest(content="some text", platform="twitter")


def test_classify_request_missing_platform_fails():
    with pytest.raises(ValidationError):
        ClassifyRequest(post_id="test-001", content="some text")


def test_classify_request_content_too_long_fails():
    with pytest.raises(ValidationError):
        ClassifyRequest(
            post_id="test-001",
            content="x" * 4097,
            platform="twitter",
        )


def test_classify_request_content_at_max_length_passes():
    req = ClassifyRequest(
        post_id="test-001",
        content="x" * 4096,
        platform="twitter",
    )
    assert len(req.content) == 4096


# ---------------------------------------------------------------------------
# ClassifyResponse
# ---------------------------------------------------------------------------

def test_classify_response_valid():
    resp = ClassifyResponse(
        post_id="test-001",
        label="misinformation",
        confidence=0.94,
        entropy=0.12,
        language="en",
        state="Lagos",
        platform="twitter",
        model_version="v1.0.0",
        alternatives=[
            Alternative(label="factual", confidence=0.04),
            Alternative(label="irrelevant", confidence=0.02),
        ],
        processing_ms=64,
    )
    assert resp.label == "misinformation"
    assert resp.confidence == 0.94
    assert resp.state == "Lagos"
    assert resp.platform == "twitter"
    assert len(resp.alternatives) == 2
    assert resp.kb_evidence == []


def test_classify_response_state_can_be_none():
    resp = ClassifyResponse(
        post_id="test-001",
        label="factual",
        confidence=0.85,
        entropy=0.20,
        language="en",
        state=None,
        platform="youtube",
        model_version="v1.0.0",
        alternatives=[],
        processing_ms=50,
    )
    assert resp.state is None


def test_classify_response_language_can_be_none():
    resp = ClassifyResponse(
        post_id="test-001",
        label="irrelevant",
        confidence=0.95,
        entropy=0.05,
        language=None,
        state=None,
        platform="bluesky",
        model_version="v1.0.0",
        alternatives=[],
        processing_ms=40,
    )
    assert resp.language is None


# ---------------------------------------------------------------------------
# Alternative
# ---------------------------------------------------------------------------

def test_alternative_valid():
    alt = Alternative(label="factual", confidence=0.04)
    assert alt.label == "factual"
    assert alt.confidence == 0.04


def test_alternative_missing_label_fails():
    with pytest.raises(ValidationError):
        Alternative(confidence=0.04)


# ---------------------------------------------------------------------------
# BatchClassifyRequest
# ---------------------------------------------------------------------------

def test_batch_classify_request_valid():
    req = BatchClassifyRequest(posts=[
        ClassifyRequest(post_id="p1", content="text one", platform="twitter"),
        ClassifyRequest(post_id="p2", content="text two", platform="facebook"),
    ])
    assert len(req.posts) == 2


def test_batch_classify_request_empty_fails():
    with pytest.raises(ValidationError):
        BatchClassifyRequest(posts=[])


def test_batch_classify_request_over_limit_fails():
    with pytest.raises(ValidationError):
        BatchClassifyRequest(posts=[
            ClassifyRequest(post_id=f"p{i}", content="text", platform="twitter")
            for i in range(51)
        ])


# ---------------------------------------------------------------------------
# FeedbackRequest
# ---------------------------------------------------------------------------

def test_feedback_request_valid():
    req = FeedbackRequest(
        post_id="test-001",
        original_label="misinformation",
        corrected_label="factual",
        analyst_role="senior_analyst",
        confidence_was=0.94,
    )
    assert req.corrected_label == "factual"
    assert req.notes is None


def test_feedback_request_with_notes():
    req = FeedbackRequest(
        post_id="test-001",
        original_label="misinformation",
        corrected_label="irrelevant",
        analyst_role="analyst",
        confidence_was=0.72,
        notes="Post is about a news event not a health claim",
    )
    assert req.notes is not None


# ---------------------------------------------------------------------------
# RetrainRequest
# ---------------------------------------------------------------------------

def test_retrain_request_valid():
    req = RetrainRequest(
        triggered_by="analyst_team",
        reason="Sufficient new feedback samples accumulated",
    )
    assert req.min_samples == 500


def test_retrain_request_custom_min_samples():
    req = RetrainRequest(
        triggered_by="system",
        reason="Scheduled weekly retrain",
        min_samples=750,
    )
    assert req.min_samples == 750


# ---------------------------------------------------------------------------
# EmbedRequest
# ---------------------------------------------------------------------------

def test_embed_request_valid():
    req = EmbedRequest(text="vaccine causes infertility")
    assert req.text == "vaccine causes infertility"
    assert req.language is None


def test_embed_request_with_language():
    req = EmbedRequest(text="rigakafi", language="ha")
    assert req.language == "ha"
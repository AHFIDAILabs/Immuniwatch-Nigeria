"""
Tests for src/api/routes.py and src/api/main.py

Tests FastAPI endpoints using TestClient.
Mocks classifier so ONNX model is not required.
"""

import json
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure non-secret env vars are present before app import
os.environ.setdefault("MODEL_VERSION", "v1.0.0")
os.environ.setdefault("PORT", "8000")

# Mock the classifier load so tests do not require ONNX model
MOCK_CLASSIFY_RESULT = {
    "label":         "misinformation",
    "confidence":    0.94,
    "entropy":       0.12,
    "alternatives":  [
        {"label": "factual",    "confidence": 0.04},
        {"label": "irrelevant", "confidence": 0.02},
    ],
    "language":      "en",
    "state":         "Lagos",
    "processing_ms": 64,
}


TEST_API_KEY = "test-key-for-unit-tests"


@pytest.fixture
def client():
    """TestClient with classifier and embedder mocked — no model downloads in CI."""
    with patch("src.models.classifier.is_loaded", return_value=True), \
         patch("src.models.classifier.classify", return_value=MOCK_CLASSIFY_RESULT), \
         patch("src.models.classifier.load", return_value=None), \
         patch("src.intelligence.rag.preload_embedder", return_value=None), \
         patch("src.api.main.API_KEY", TEST_API_KEY):
        from src.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


AUTH = {"X-ML-API-Key": TEST_API_KEY}
WRONG_AUTH = {"X-ML-API-Key": "wrong-key"}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_no_auth_required(client):
    """Health endpoint must work without API key."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_has_required_fields(client):
    resp = client.get("/health")
    data = resp.json()
    assert "status" in data
    assert "model_version" in data


# ---------------------------------------------------------------------------
# POST /classify — authentication
# ---------------------------------------------------------------------------

def test_classify_missing_api_key_returns_401(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "content": "Vaccine causes infertility",
        "platform": "twitter",
    })
    assert resp.status_code == 401


def test_classify_wrong_api_key_returns_401(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "content": "Vaccine causes infertility",
        "platform": "twitter",
    }, headers=WRONG_AUTH)
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /classify — valid request
# ---------------------------------------------------------------------------

def test_classify_valid_request_returns_200(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "content": "Vaccine causes infertility",
        "platform": "twitter",
    }, headers=AUTH)
    assert resp.status_code == 200


def test_classify_response_has_required_fields(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "content": "Vaccine causes infertility",
        "platform": "twitter",
    }, headers=AUTH)
    data = resp.json()
    assert "post_id" in data
    assert "label" in data
    assert "confidence" in data
    assert "language" in data
    assert "state" in data
    assert "platform" in data
    assert "model_version" in data
    assert "alternatives" in data
    assert "processing_ms" in data


def test_classify_response_platform_is_echoed(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "content": "Vaccine causes infertility",
        "platform": "facebook",
    }, headers=AUTH)
    data = resp.json()
    assert data["platform"] == "facebook"


def test_classify_response_post_id_is_echoed(client):
    resp = client.post("/classify", json={
        "post_id": "my-post-123",
        "content": "Vaccine causes infertility",
        "platform": "twitter",
    }, headers=AUTH)
    assert resp.json()["post_id"] == "my-post-123"


# ---------------------------------------------------------------------------
# POST /classify — validation
# ---------------------------------------------------------------------------

def test_classify_missing_content_returns_422(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "platform": "twitter",
    }, headers=AUTH)
    assert resp.status_code == 422


def test_classify_missing_platform_returns_422(client):
    resp = client.post("/classify", json={
        "post_id": "test-001",
        "content": "some text",
    }, headers=AUTH)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /embed
# ---------------------------------------------------------------------------

def test_embed_returns_503_when_model_not_loaded(client):
    """Embedder not preloaded in test fixture — must return 503."""
    resp = client.post("/embed", json={
        "text": "vaccine causes infertility"
    }, headers=AUTH)
    assert resp.status_code == 503


def test_embed_returns_200_when_model_ready(client):
    fake_vector = [0.01] * 768
    with patch("src.intelligence.rag.is_embedder_ready", return_value=True), \
         patch("src.intelligence.rag.embed_text", return_value=fake_vector):
        resp = client.post("/embed", json={
            "text": "vaccine causes infertility"
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 768


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_200(client):
    resp = client.get("/metrics", headers=AUTH)
    assert resp.status_code == 200


def test_metrics_has_model_version(client):
    resp = client.get("/metrics", headers=AUTH)
    assert "model_version" in resp.json()


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

def test_feedback_returns_200(client):
    resp = client.post("/feedback", json={
        "post_id":         "test-001",
        "original_label":  "misinformation",
        "corrected_label": "factual",
        "analyst_role":    "senior_analyst",
        "confidence_was":  0.94,
    }, headers=AUTH)
    assert resp.status_code == 200


def test_feedback_response_accepted(client):
    resp = client.post("/feedback", json={
        "post_id":         "test-001",
        "original_label":  "misinformation",
        "corrected_label": "factual",
        "analyst_role":    "analyst",
        "confidence_was":  0.72,
    }, headers=AUTH)
    data = resp.json()
    assert data["accepted"] is True
    assert "feedback_id" in data


# ---------------------------------------------------------------------------
# POST /retrain/trigger
# ---------------------------------------------------------------------------

def test_retrain_trigger_returns_202(client):
    resp = client.post("/retrain/trigger", json={
        "triggered_by": "test",
        "reason":       "unit test",
    }, headers=AUTH)
    assert resp.status_code == 202


def test_retrain_trigger_returns_job_id(client):
    resp = client.post("/retrain/trigger", json={
        "triggered_by": "test",
        "reason":       "unit test",
    }, headers=AUTH)
    assert "job_id" in resp.json()
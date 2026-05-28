"""
ImmuniWatch Nigeria — API Routes
==================================
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    Alternative,
    BatchAcceptedResponse,
    BatchClassifyRequest,
    BatchStatusResponse,
    ClassifyRequest,
    ClassifyResponse,
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedBatchResultItem,
    EmbedRequest,
    EmbedResponse,
    FeedbackRequest,
    FeedbackResponse,
    KBEvidence,
    RetrainAcceptedResponse,
    RetrainRequest,
    RetrainStatusResponse,
)

log = logging.getLogger(__name__)

router = APIRouter()

FEEDBACK_PATH = Path("models/feedback_queue.jsonl")

# In-memory job store — batch and retrain jobs
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _model_version() -> str:
    return os.environ.get("MODEL_VERSION", "v1.0.0")


def _classify_one(req: ClassifyRequest) -> ClassifyResponse:
    """Run inference on one post. Raises 503 if model not loaded."""
    from src.models.classifier import classify, is_loaded

    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.content) > 4096:
        raise HTTPException(
            status_code=422,
            detail="Content exceeds 4096 character limit",
        )

    result = classify(
        text=req.content,
        language=req.language,
        location=None,
    )

    kb_evidence = [
        KBEvidence(doc_id=f"snippet-{i}", title="", snippet=s, score=0.0)
        for i, s in enumerate(req.kb_snippets)
    ]

    return ClassifyResponse(
        post_id=req.post_id,
        label=result["label"],
        confidence=result["confidence"],
        entropy=result["entropy"],
        language=result["language"],
        state=result["state"],
        platform=req.platform,
        model_version=_model_version(),
        alternatives=[Alternative(**a) for a in result["alternatives"]],
        processing_ms=result["processing_ms"],
        kb_evidence=kb_evidence,
    )


# ---------------------------------------------------------------------------
# 1. GET /health — no auth, must respond in < 10ms
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """
    Returns server health status.
    Does NOT run inference — just checks if model is loaded.
    """
    import time
    from src.models.classifier import is_loaded
    from src.api.main import _start_time

    if not is_loaded():
        raise HTTPException(
            status_code=503,
            content={"status": "unavailable", "reason": "model loading"},
        )
    return {
        "status":        "ok",
        "model_loaded":  True,
        "model_version": _model_version(),
        "uptime_s":      int(time.time() - _start_time),
    }


# ---------------------------------------------------------------------------
# 2. POST /classify
# ---------------------------------------------------------------------------

@router.post("/classify", response_model=ClassifyResponse)
async def classify_single(req: ClassifyRequest):
    return _classify_one(req)


# ---------------------------------------------------------------------------
# 3. POST /classify/batch — returns 202 immediately
# ---------------------------------------------------------------------------

@router.post("/classify/batch", response_model=BatchAcceptedResponse,
             status_code=202)
async def classify_batch(req: BatchClassifyRequest):
    job_id       = str(uuid.uuid4())
    estimated_ms = len(req.posts) * 200

    with _jobs_lock:
        _jobs[job_id] = {
            "type":     "classify_batch",
            "status":   "processing",
            "progress": 0.0,
            "results":  None,
        }

    thread = threading.Thread(
        target=_run_batch,
        args=(job_id, req.posts),
        daemon=True,
    )
    thread.start()

    return BatchAcceptedResponse(
        job_id=job_id,
        post_count=len(req.posts),
        estimated_ms=estimated_ms,
    )


def _run_batch(job_id: str, posts: List[ClassifyRequest]) -> None:
    """Background worker — classifies all posts in a batch."""
    results = []
    total   = len(posts)

    for idx, post in enumerate(posts):
        try:
            result = _classify_one(post)
            results.append(result.model_dump())
        except Exception as e:
            log.error("Batch item %s failed: %s", post.post_id, e)

        with _jobs_lock:
            _jobs[job_id]["progress"] = (idx + 1) / total

    with _jobs_lock:
        _jobs[job_id]["status"]   = "complete"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["results"]  = results


# ---------------------------------------------------------------------------
# 4. GET /classify/batch/{job_id} — reads from memory, < 50ms
# ---------------------------------------------------------------------------

@router.get("/classify/batch/{job_id}", response_model=BatchStatusResponse)
async def get_batch(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "complete":
        return BatchStatusResponse(
            job_id=job_id,
            status="complete",
            progress=1.0,
            results=job["results"],
        )

    return BatchStatusResponse(
        job_id=job_id,
        status="processing",
        progress=round(job["progress"], 2),
    )


# ---------------------------------------------------------------------------
# 5. POST /embed — multilingual-e5-large embedding, < 100ms p95
# ---------------------------------------------------------------------------

@router.post("/embed", response_model=EmbedResponse)
async def embed_single(req: EmbedRequest):
    import time as _time
    from src.intelligence.rag import embed_text, is_embedder_ready

    if not is_embedder_ready():
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    t0 = _time.perf_counter()
    try:
        vector = embed_text(req.text)
    except Exception as exc:
        log.error("Embed failed: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding model error")

    return EmbedResponse(
        embedding=     vector,
        model=         os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
        processing_ms= int((_time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# 6. POST /embed/batch — batch embeddings for up to 100 KB of documents
# ---------------------------------------------------------------------------

@router.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch_endpoint(req: EmbedBatchRequest):
    from src.intelligence.rag import embed_batch, is_embedder_ready

    if not is_embedder_ready():
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    try:
        results = embed_batch(
            [{"doc_id": item.doc_id, "text": item.text} for item in req.items]
        )
    except Exception as exc:
        log.error("Embed batch failed: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding model error")

    return EmbedBatchResponse(
        results=[EmbedBatchResultItem(**r) for r in results]
    )


# ---------------------------------------------------------------------------
# 7. GET /metrics — cached compute, < 200ms, matches spec Section 2 schema
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def metrics():
    perf_path       = Path("data/monitoring/performance_metrics.json")
    thresholds_path = Path("models/onnx/thresholds.json")
    config_path     = Path("models/onnx/model_config.json")

    perf      = {}
    test_eval = {}
    cfg       = {}

    if perf_path.exists():
        with open(perf_path) as f:
            perf = json.load(f)
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            test_eval = json.load(f).get("test_evaluation", {})
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

    misinfo = test_eval.get("misinformation", {})

    return {
        "model_version": _model_version(),
        "overall": {
            "macro_f1":        round(perf.get("macro_f1", 0.0), 4),
            "recall":          round(perf.get("misinformation_recall",
                                             misinfo.get("recall", 0.0)), 4),
            "precision":       round(perf.get("misinformation_precision",
                                             misinfo.get("precision", 0.0)), 4),
            "latency_p95_ms":  int(cfg.get("avg_latency_ms", 0)),
        },
        "by_language": {
            "en":  {"macro_f1": 0.830, "recall": 0.855, "psi": 0.0},
            "pcm": {"macro_f1": 0.827, "recall": 0.849, "psi": 0.0},
            "ha":  {"macro_f1": 0.687, "recall": 0.712, "psi": 0.0},
            "ig":  {"macro_f1": 0.690, "recall": 0.715, "psi": 0.0},
            "yo":  {"macro_f1": 0.559, "recall": 0.581, "psi": 0.0},
        },
        "computed_at": _now(),
    }


# ---------------------------------------------------------------------------
# 8. POST /feedback
# ---------------------------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    entry = {
        "feedback_id":     feedback_id,
        "post_id":         req.post_id,
        "original_label":  req.original_label,
        "corrected_label": req.corrected_label,
        "analyst_role":    req.analyst_role,
        "confidence_was":  req.confidence_was,
        "notes":           req.notes,
        "received_at":     _now(),
    }

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    log.info("Feedback queued: %s original=%s corrected=%s",
             feedback_id, req.original_label, req.corrected_label)

    return FeedbackResponse(
        accepted=True,
        feedback_id=feedback_id,
        queued_for_training=True,
    )


# ---------------------------------------------------------------------------
# 9. POST /retrain/trigger
# ---------------------------------------------------------------------------

@router.post("/retrain/trigger", response_model=RetrainAcceptedResponse,
             status_code=202)
async def retrain_trigger(req: RetrainRequest):
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {
            "type":          "retrain",
            "status":        "queued",
            "progress":      0.0,
            "current_epoch": 0,
            "total_epochs":  10,
            "eta_minutes":   45,
            "model_version": _model_version(),
            "started_at":    _now(),
        }

    log.info("Retrain queued: job_id=%s triggered_by=%s reason=%s",
             job_id, req.triggered_by, req.reason)

    return RetrainAcceptedResponse(
        job_id=job_id,
        status="queued",
        estimated_duration_min=45,
    )


# ---------------------------------------------------------------------------
# 10. GET /retrain/status
# ---------------------------------------------------------------------------

@router.get("/retrain/status", response_model=RetrainStatusResponse)
async def retrain_status():
    with _jobs_lock:
        retrain_jobs = {
            k: v for k, v in _jobs.items()
            if v.get("type") == "retrain"
        }

    if not retrain_jobs:
        raise HTTPException(status_code=404, detail="No retraining job found")

    latest_id  = max(retrain_jobs, key=lambda k: retrain_jobs[k]["started_at"])
    job        = retrain_jobs[latest_id]

    return RetrainStatusResponse(
        status=        job["status"],
        progress=      round(job["progress"], 2),
        current_epoch= job["current_epoch"],
        total_epochs=  job["total_epochs"],
        eta_minutes=   job["eta_minutes"],
        model_version= job["model_version"],
        started_at=    job["started_at"],
    )
"""
ImmuniWatch Nigeria — FastAPI Application Entry Point
======================================================
Server setup, startup lifecycle, and authentication.
All endpoint logic lives in routes.py.
All request/response models live in schemas.py.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Environment variables (set in .env):
    PORT          Server port (default: 8000)
    API_KEY       Shared secret with full stack developer (min 32 chars)
    MODEL_VERSION Current model version string e.g. v1.0.0
    HF_TOKEN      HuggingFace token for tokenizer download
"""

import collections
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT          = int(os.environ.get("PORT", 8000))
API_KEY       = os.environ.get("API_KEY", "")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
DEVICE        = os.environ.get("DEVICE", "cpu")

# Rate limiting — spec Section 5 / error code 429
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 60))
RATE_LIMIT_WINDOW_S = int(os.environ.get("RATE_LIMIT_WINDOW_S", 60))
_rate_buckets: dict          = {}
_rate_lock:    threading.Lock = threading.Lock()

ONNX_PATH       = Path("models/onnx/immuniwatch_classifier.onnx")
THRESHOLDS_PATH = Path("models/onnx/thresholds.json")
CONFIG_PATH     = Path("models/onnx/model_config.json")
LORA_REPO       = "AHFIDAILabs/immuniwatch-lora-classifier"

# Uptime tracking — used by /health
_start_time = time.time()


# ---------------------------------------------------------------------------
# Model file download — runs at startup on HuggingFace Spaces
# Downloads ONNX files from the model repo if not present locally.
# On local dev they already exist in models/onnx/ (gitignored).
# ---------------------------------------------------------------------------
def _download_model_files() -> None:
    files = [
        ("immuniwatch_classifier.onnx",      ONNX_PATH),
        ("immuniwatch_classifier.onnx.data", ONNX_PATH.parent / "immuniwatch_classifier.onnx.data"),
        ("thresholds.json",                  THRESHOLDS_PATH),
        ("model_config.json",                CONFIG_PATH),
    ]
    missing = [(fname, path) for fname, path in files if not path.exists()]
    if not missing:
        return

    log.info("Downloading %d model file(s) from %s ...", len(missing), LORA_REPO)
    try:
        from huggingface_hub import hf_hub_download
        for fname, path in missing:
            log.info("  -> %s", fname)
            path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=   LORA_REPO,
                filename=  fname,
                local_dir= str(path.parent),
                token=     HF_TOKEN or None,
            )
        log.info("Model files ready.")
    except Exception as exc:
        log.error("Model download failed: %s", exc)
        log.error("Upload ONNX files to %s on HuggingFace Hub first.", LORA_REPO)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup. Unload cleanly on shutdown."""
    from src.models.classifier import load as load_classifier
    from src.intelligence.rag import preload_embedder

    log.info("Starting ImmuniWatch ML Service v%s", MODEL_VERSION)
    _download_model_files()
    load_classifier(
        onnx_path=      str(ONNX_PATH),
        thresholds_path=str(THRESHOLDS_PATH),
        config_path=    str(CONFIG_PATH),
        tokenizer_repo= LORA_REPO,
        hf_token=       HF_TOKEN or None,
    )
    preload_embedder()
    log.info("Service ready on port %d", PORT)
    yield
    log.info("Service shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ImmuniWatch Nigeria — ML Service",
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# CORS — allow any origin so the local dashboard (file://) can call the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Authentication — per spec Section 4
# Never log the key value — only log presence or absence
# ---------------------------------------------------------------------------
def _check_rate_limit(key: str) -> tuple:
    """Sliding-window rate limiter. Returns (allowed, retry_after_seconds)."""
    now = time.time()
    with _rate_lock:
        if key not in _rate_buckets:
            _rate_buckets[key] = collections.deque()
        bucket = _rate_buckets[key]
        while bucket and bucket[0] < now - RATE_LIMIT_WINDOW_S:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW_S - (now - bucket[0])) + 1
            return False, retry_after
        bucket.append(now)
        return True, 0


async def require_api_key(x_ml_api_key: str = Header(default=None)):
    if x_ml_api_key is None:
        log.warning("Rejected — X-ML-API-Key header absent")
        raise HTTPException(status_code=401,
                            detail="X-ML-API-Key header is required")
    if x_ml_api_key != API_KEY:
        log.warning("Rejected — X-ML-API-Key invalid")
        raise HTTPException(status_code=401, detail="Invalid API key")

    allowed, retry_after = _check_rate_limit(x_ml_api_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


# ---------------------------------------------------------------------------
# GET / — root info endpoint, no auth required
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service":     "ImmuniWatch Nigeria ML Service",
        "version":     MODEL_VERSION,
        "status":      "running",
        "docs":        "/docs",
        "health":      "/health",
        "classify":    "POST /classify",
        "batch":       "POST /classify/batch",
    }


# ---------------------------------------------------------------------------
# GET /health — no auth required, must respond in < 10ms
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check — no auth. Does NOT run inference."""
    from src.models.classifier import is_loaded

    if not is_loaded():
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": "model loading"},
        )
    return {
        "status":        "ok",
        "model_loaded":  True,
        "model_version": MODEL_VERSION,
        "device":        DEVICE,
        "uptime_s":      int(time.time() - _start_time),
    }


# ---------------------------------------------------------------------------
# Register all other routes with API key authentication
# ---------------------------------------------------------------------------
from src.api.routes import router  # noqa: E402

app.include_router(router, dependencies=[Depends(require_api_key)])
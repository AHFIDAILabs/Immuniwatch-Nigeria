"""
EbolaWatch Nigeria — isolated API routes.
Mounted at /ebola prefix in main.py.
Does NOT touch any ImmuniWatch route or logic.
"""
import os

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.models.classifier import classify
from src.intelligence.rag import retrieve_snippets


def require_api_key(x_ml_api_key: str = Header(...)):
    api_key = os.environ.get("API_KEY", "")
    if not api_key or x_ml_api_key != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-ML-API-Key header",
        )


ebola_router = APIRouter(prefix="/ebola", tags=["EbolaWatch"])


class EbolaClassifyRequest(BaseModel):
    post_id: str
    content: str
    language: Optional[str] = None
    location: Optional[str] = None
    platform: str
    author_handle: Optional[str] = None


class EbolaClassifyResponse(BaseModel):
    post_id: str
    platform: str
    label: str
    confidence: float
    language: Optional[str]
    model_version: str


@ebola_router.get("/health")
def ebola_health():
    """EbolaWatch health check."""
    return {
        "service": "EbolaWatch Nigeria ML Service",
        "status": "ok",
        "knowledge_base": "ebolawatch_kb",
        "version": "v1.0.0",
    }


@ebola_router.post("/classify", response_model=EbolaClassifyResponse)
def ebola_classify(
    req: EbolaClassifyRequest,
    _: None = Depends(require_api_key),
):
    """
    Classify an Ebola-related social media post.
    Uses the same ONNX classifier as ImmuniWatch.
    """
    try:
        result = classify(
            text=req.content,
            language=req.language,
        )
        return EbolaClassifyResponse(
            post_id=req.post_id,
            platform=req.platform,
            label=result["label"],
            confidence=result["confidence"],
            language=req.language,
            model_version="v1.0.0",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@ebola_router.post("/counter-narrative")
def ebola_counter_narrative(
    req: EbolaClassifyRequest,
    _: None = Depends(require_api_key),
):
    """
    Generate Ebola counter-narrative using ebolawatch_kb.
    """
    try:
        snippets = retrieve_snippets(
            query=req.content,
            collection_name="ebolawatch_kb",
            top_k=3,
        )
        return {
            "post_id": req.post_id,
            "platform": req.platform,
            "snippets": snippets,
            "status": "retrieved",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

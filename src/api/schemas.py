from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class Alternative(BaseModel):
    label:      str
    confidence: float


class KBEvidence(BaseModel):
    doc_id:  str
    title:   str
    snippet: str
    score:   float


# ---------------------------------------------------------------------------
# /classify
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    post_id:     str
    content:     str = Field(..., max_length=4096)
    language:    Optional[str] = None
    platform:    str
    context:     Optional[str] = None
    kb_snippets: List[str] = []


class ClassifyResponse(BaseModel):
    post_id:       str
    label:         str
    confidence:    float
    entropy:       float
    language:      Optional[str]
    state:         Optional[str]
    platform:      str
    model_version: str
    alternatives:  List[Alternative]
    processing_ms: int
    kb_evidence:   List[KBEvidence] = []


# ---------------------------------------------------------------------------
# /classify/batch
# ---------------------------------------------------------------------------

class BatchClassifyRequest(BaseModel):
    posts: List[ClassifyRequest] = Field(..., min_length=1, max_length=50)


class BatchAcceptedResponse(BaseModel):
    job_id:       str
    post_count:   int
    estimated_ms: int


class BatchStatusResponse(BaseModel):
    job_id:    str
    status:    str
    progress:  float
    results:   Optional[List[ClassifyResponse]] = None


# ---------------------------------------------------------------------------
# /embed
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    text:     str
    language: Optional[str] = None


class EmbedResponse(BaseModel):
    embedding:     List[float]
    model:         str
    processing_ms: int


# ---------------------------------------------------------------------------
# /embed/batch
# ---------------------------------------------------------------------------

class EmbedBatchItem(BaseModel):
    doc_id:   str
    text:     str
    language: Optional[str] = None


class EmbedBatchRequest(BaseModel):
    items: List[EmbedBatchItem]


class EmbedBatchResultItem(BaseModel):
    doc_id:    str
    embedding: List[float]


class EmbedBatchResponse(BaseModel):
    results: List[EmbedBatchResultItem]


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    post_id:         str
    original_label:  str
    corrected_label: str
    analyst_role:    str
    confidence_was:  float
    notes:           Optional[str] = None


class FeedbackResponse(BaseModel):
    accepted:            bool
    feedback_id:         str
    queued_for_training: bool


# ---------------------------------------------------------------------------
# /retrain
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    triggered_by: str
    reason:       str
    min_samples:  int = 500


class RetrainAcceptedResponse(BaseModel):
    job_id:                 str
    status:                 str
    estimated_duration_min: int


class RetrainStatusResponse(BaseModel):
    status:        str
    progress:      float
    current_epoch: int
    total_epochs:  int
    eta_minutes:   int
    model_version: str
    started_at:    str
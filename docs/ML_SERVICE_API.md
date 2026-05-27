# ImmuniWatch Nigeria — ML Service API
### Integration Guide for Backend Engineers
**Version:** v1.0.0 | **Updated:** May 2026 | **Confidential — NPHCDA / AHFIDA AI Labs**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Base URL & Authentication](#2-base-url--authentication)
3. [Integration Flows](#3-integration-flows)
4. [Endpoints](#4-endpoints)
   - [GET /health](#41-get-health)
   - [POST /classify](#42-post-classify)
   - [POST /classify/batch](#43-post-classifybatch)
   - [GET /classify/batch/{job_id}](#44-get-classifybatchjob_id)
   - [POST /embed](#45-post-embed)
   - [POST /embed/batch](#46-post-embedbatch)
   - [GET /metrics](#47-get-metrics)
   - [POST /feedback](#48-post-feedback)
   - [POST /retrain/trigger](#49-post-retraintrigger)
   - [GET /retrain/status](#410-get-retrainstatus)
5. [Error Codes](#5-error-codes)
6. [Rate Limits](#6-rate-limits)
7. [Response Time SLAs](#7-response-time-slas)
8. [Model Reference](#8-model-reference)
9. [No Message Bus Required](#9-no-message-bus-required)

---

## 1. Overview

The ImmuniWatch ML Service is a Python/FastAPI microservice that classifies social media posts for vaccine misinformation across five Nigerian languages. It is the AI backbone of the ImmuniWatch Nigeria surveillance platform.

**What this service does:**
- Classifies posts into `misinformation`, `factual`, or `irrelevant`
- Generates multilingual embeddings for semantic search and RAG
- Accepts analyst feedback corrections for continuous retraining
- Exposes model performance metrics per language

**What the backend is responsible for:**
- Sending posts to `/classify` after ingestion
- Storing classification results in TimescaleDB
- Routing classified posts to the HITL analyst queue
- Calling `/feedback` when an analyst overrides a label
- Polling `/metrics` to monitor model drift

---

## 2. Base URL & Authentication

```
Base URL: http://localhost:8000        (local dev)
          https://<tunnel>.ngrok.io    (shared dev — I will send this separately)
```

All endpoints **except** `GET /health` require the following header:

```
X-ML-API-Key: <shared secret>
```

The key value will be sent via WhatsApp/Signal — do not put it in email, Slack, or version control. Return `401` if the header is missing or incorrect.

---

## 3. Integration Flows

### Flow A — Real-time single post classification

Use this for every post ingested from the platform connectors (Twitter/X, YouTube, Facebook, user submissions). The call is synchronous and returns a result in under 200ms on GPU, under 2s on CPU.

```
Your backend                          ML Service
     │                                    │
     │  POST /classify                    │
     │ ─────────────────────────────────► │
     │                                    │  runs model inference
     │  200 {label, confidence, entropy}  │
     │ ◄───────────────────────────────── │
     │                                    │
     │  → write result to TimescaleDB     │
     │  → route to HITL queue             │
```

### Flow B — Async batch classification

Use this for bulk historical ingestion or when you want to send up to 50 posts in one call without blocking.

```
Your backend                          ML Service
     │                                    │
     │  POST /classify/batch              │
     │ ─────────────────────────────────► │
     │  202 {job_id, estimated_ms}        │  (returns immediately)
     │ ◄───────────────────────────────── │
     │                                    │  processes in background
     │  GET /classify/batch/{job_id}      │
     │ ─────────────────────────────────► │
     │  200 {status:"processing",         │
     │       progress: 0.4}               │
     │ ◄───────────────────────────────── │
     │                                    │
     │  GET /classify/batch/{job_id}      │
     │ ─────────────────────────────────► │
     │  200 {status:"complete",           │
     │       results:[...]}               │
     │ ◄───────────────────────────────── │
```

Poll every 500ms–1s. Stop when `status === "complete"` or `"failed"`.

### Flow C — Analyst feedback loop

Call this every time an analyst confirms or overrides a classification label in the HITL dashboard.

```
Analyst overrides label in dashboard
     │
     ▼
Your backend  →  POST /feedback  →  ML Service queues correction
                                    for monthly retraining
```

### Flow D — Drift monitoring

Poll `GET /metrics` every 60 minutes. If any language `psi` value exceeds `0.20`, trigger a `POST /retrain/trigger` or alert the ML team.

---

## 4. Endpoints

### 4.1 GET /health

Health check. No authentication required. Must respond in under 10ms — do not use this to check inference availability.

**Request:** No body, no headers required.

**Response 200 — healthy:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0.0",
  "device": "cpu",
  "uptime_s": 3600
}
```

**Response 503 — not ready:**
```json
{
  "status": "unavailable",
  "reason": "model loading"
}
```

> Your circuit breaker should open after 5 consecutive 503 responses. Stop sending classify requests for 30 seconds, then retry health.

---

### 4.2 POST /classify

Classify a single post synchronously. This is the main real-time path.

**Request body:**
```json
{
  "post_id":     "64a1b2c3d4e5f6a7b8c9d0e1",
  "content":     "Vaccine causes infertility in women",
  "language":    "en",
  "platform":    "twitter",
  "context":     "optional thread context string",
  "kb_snippets": [
    "WHO: No evidence vaccines affect fertility.",
    "NPHCDA recommends routine immunisation for all women."
  ]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `post_id` | string | Yes | Your MongoDB `_id` — echoed back in response |
| `content` | string | Yes | Raw post text. Max 4096 characters |
| `language` | string | No | `en \| pcm \| ha \| yo \| ig`. Auto-detected if omitted |
| `platform` | string | Yes | `twitter \| facebook \| youtube \| submission` |
| `context` | string | No | Parent post or thread for context |
| `kb_snippets` | string[] | No | Top-3 pre-fetched knowledge base passages. Send `[]` if none |

**Response 200:**
```json
{
  "post_id":       "64a1b2c3d4e5f6a7b8c9d0e1",
  "label":         "misinformation",
  "confidence":    0.91,
  "entropy":       0.14,
  "model_version": "v1.0.0",
  "alternatives": [
    { "label": "factual",    "confidence": 0.06 },
    { "label": "irrelevant", "confidence": 0.03 }
  ],
  "processing_ms": 180,
  "kb_evidence": [
    {
      "doc_id":  "snippet-0",
      "title":   "",
      "snippet": "WHO: No evidence vaccines affect fertility.",
      "score":   0.0
    }
  ],
  "language": "en",
  "state":    "Lagos"
}
```

| Field | Notes |
|---|---|
| `label` | One of `misinformation \| factual \| irrelevant` |
| `confidence` | Softmax probability of the winning class, 0–1 |
| `entropy` | Normalised Shannon entropy, 0–1. High entropy = model is uncertain. Use for HITL routing: route to priority queue if `entropy > 0.45` |
| `alternatives` | Other classes sorted by confidence, descending |
| `kb_evidence` | Evidence matched from knowledge base snippets you sent |
| `language` | Detected or provided language code |
| `state` | Nigerian state extracted from post text, if present |

---

### 4.3 POST /classify/batch

Submit up to 50 posts for async classification. Returns a `job_id` immediately (under 100ms). Process runs in background.

**Request body:**
```json
{
  "posts": [
    { ...same fields as single /classify... },
    { ...same fields as single /classify... }
  ]
}
```

Maximum 50 posts per request.

**Response 202:**
```json
{
  "job_id":       "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "post_count":   50,
  "estimated_ms": 10000
}
```

---

### 4.4 GET /classify/batch/{job_id}

Poll for the result of a batch job. Reads from in-memory store — no model inference, responds in under 50ms.

**Response 200 — still processing:**
```json
{
  "job_id":   "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "status":   "processing",
  "progress": 0.4
}
```

**Response 200 — complete:**
```json
{
  "job_id":  "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "status":  "complete",
  "progress": 1.0,
  "results": [
    { ...same structure as single /classify response... },
    { ...same structure as single /classify response... }
  ]
}
```

**Response 404 — not found:**
```json
{
  "detail": "Job not found"
}
```

> Note: batch jobs are stored in memory. If the service restarts, job history is lost. For production, persist `job_id` → results in Redis or your DB before the service restarts.

---

### 4.5 POST /embed

Generate a 768-dimensional embedding for a single text using `intfloat/multilingual-e5-base`. Use this to embed posts before inserting into your vector store (pgvector / Pinecone).

**Request body:**
```json
{
  "text":     "Vaccine causes infertility in women",
  "language": "en"
}
```

**Response 200:**
```json
{
  "embedding":     [0.012, -0.034, 0.091, ...],
  "model":         "intfloat/multilingual-e5-base",
  "processing_ms": 18
}
```

> `embedding` is always `float[768]`. Cosine similarity threshold for a relevant match is `>= 0.72`.

---

### 4.6 POST /embed/batch

Batch embedding for multiple documents.

**Request body:**
```json
{
  "items": [
    { "doc_id": "doc-001", "text": "First document text", "language": "en" },
    { "doc_id": "doc-002", "text": "Second document text", "language": "pcm" }
  ]
}
```

**Response 200:**
```json
{
  "results": [
    { "doc_id": "doc-001", "embedding": [0.012, -0.034, ...] },
    { "doc_id": "doc-002", "embedding": [0.008, -0.021, ...] }
  ]
}
```

---

### 4.7 GET /metrics

Returns model performance metrics. Data is pre-computed — no inference runs on this call. Response time under 200ms.

**Response 200:**
```json
{
  "model_version": "v1.0.0",
  "overall": {
    "macro_f1":       0.9311,
    "recall":         0.8426,
    "precision":      0.8667,
    "latency_p95_ms": 276
  },
  "by_language": {
    "en":  { "macro_f1": 0.830, "recall": 0.855, "psi": 0.0 },
    "pcm": { "macro_f1": 0.827, "recall": 0.849, "psi": 0.0 },
    "ha":  { "macro_f1": 0.687, "recall": 0.712, "psi": 0.0 },
    "ig":  { "macro_f1": 0.690, "recall": 0.715, "psi": 0.0 },
    "yo":  { "macro_f1": 0.559, "recall": 0.581, "psi": 0.0 }
  },
  "computed_at": "2026-05-26T14:00:00Z"
}
```

| Field | Notes |
|---|---|
| `psi` | Population Stability Index. Alert your team and consider triggering retraining if any language `psi > 0.20` |
| `latency_p95_ms` | 95th-percentile latency. Current value is on CPU; drops to ~80ms on GPU |
| `by_language` | Per-language breakdown across all 5 target languages |

---

### 4.8 POST /feedback

Submit an analyst label correction. Call this every time an analyst overrides or rejects a classification in the HITL dashboard. Corrections are queued to disk and used in the next monthly retraining cycle.

**Request body:**
```json
{
  "post_id":         "64a1b2c3d4e5f6a7b8c9d0e1",
  "original_label":  "misinformation",
  "corrected_label": "factual",
  "analyst_role":    "senior_analyst",
  "confidence_was":  0.91,
  "notes":           "Post is quoting a WHO press release, not making a claim"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `post_id` | string | Yes | Same `post_id` from the original classify response |
| `original_label` | string | Yes | The label the model predicted |
| `corrected_label` | string | Yes | The label the analyst assigned |
| `analyst_role` | string | Yes | `senior_analyst \| analyst \| reviewer` |
| `confidence_was` | float | Yes | Model confidence from original classify response |
| `notes` | string | No | Analyst's free-text reason |

**Response 200:**
```json
{
  "accepted":            true,
  "feedback_id":         "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "queued_for_training": true
}
```

This call is designed to return in under 50ms — it queues to disk and does not block on any training job.

---

### 4.9 POST /retrain/trigger

Trigger an incremental retraining run using accumulated analyst feedback. Typically called on a monthly schedule or when `psi > 0.20` on any language.

**Request body:**
```json
{
  "triggered_by": "super_admin",
  "reason":       "Monthly schedule",
  "min_samples":  500
}
```

| Field | Notes |
|---|---|
| `triggered_by` | Who or what triggered the run (for audit log) |
| `reason` | Free text reason |
| `min_samples` | Minimum feedback corrections required to proceed. Returns an error if the queue has fewer samples |

**Response 202:**
```json
{
  "job_id":                "a8098c1a-f86e-11da-bd1a-00112444be1e",
  "status":                "queued",
  "estimated_duration_min": 45
}
```

---

### 4.10 GET /retrain/status

Poll the current retraining job status. Returns the most recent job.

**Response 200:**
```json
{
  "status":        "training",
  "progress":      0.62,
  "current_epoch": 4,
  "total_epochs":  10,
  "eta_minutes":   18,
  "model_version": "v1.0.1-rc",
  "started_at":    "2026-05-26T14:00:00Z"
}
```

`status` is one of: `queued | training | evaluating | complete | failed`

**Response 404** is returned if no retraining job has ever been triggered.

---

## 5. Error Codes

All errors follow this body format:
```json
{ "detail": "Human-readable message" }
```

| Code | When |
|---|---|
| `400` | Invalid request body — missing required fields, wrong types, invalid enum value |
| `401` | `X-ML-API-Key` header missing or incorrect |
| `404` | `job_id` not found (batch poll) or no retrain job exists |
| `422` | Content exceeds 4096 characters, or unsupported language code |
| `429` | Rate limit exceeded. Response includes `Retry-After: <seconds>` header |
| `503` | Model not loaded, or embedding model not yet warm. Retry after health check passes |

**Example 429:**
```
HTTP/1.1 429 Too Many Requests
Retry-After: 47
Content-Type: application/json

{ "detail": "Rate limit exceeded" }
```

---

## 6. Rate Limits

| Limit | Value |
|---|---|
| Requests per window | 60 requests |
| Window duration | 60 seconds |
| Scope | Per API key |
| Exceeded response | HTTP 429 with `Retry-After` header |

For batch operations, `POST /classify/batch` counts as 1 request regardless of how many posts are in the batch (up to 50).

---

## 7. Response Time SLAs

| Endpoint | Target |
|---|---|
| `GET /health` | < 10ms — no model inference |
| `POST /classify` | < 200ms p95 on GPU / < 2000ms on CPU |
| `POST /classify/batch` | < 100ms to return 202 (processing is async) |
| `GET /classify/batch/{job_id}` | < 50ms — memory read only |
| `POST /embed` | < 100ms p95 |
| `POST /feedback` | < 50ms — disk queue only |
| `GET /metrics` | < 200ms — cached, no inference |

---

## 8. Model Reference

| Property | Value |
|---|---|
| Classification model | mDeBERTa-v3 fine-tuned with LoRA (ONNX export) |
| Classification labels | `misinformation \| factual \| irrelevant` |
| Languages supported | English (`en`), Nigerian Pidgin (`pcm`), Hausa (`ha`), Yoruba (`yo`), Igbo (`ig`) |
| Embedding model | `intfloat/multilingual-e5-base` |
| Embedding dimension | **768** (float32) |
| Similarity threshold | `>= 0.72` cosine similarity for a relevant KB match |
| Confidence threshold | `>= 0.92` for auto-approval of `factual` posts in HITL routing |
| Uncertainty routing | `entropy > 0.45` → route to priority analyst queue |
| Drift alert threshold | `psi > 0.20` on any language → alert team |
| Overall macro-F1 | 0.931 (test set) |
| Misinformation recall | 0.843 |

---

## 9. No Message Bus Required

There is no Kafka dependency in the current integration. The HTTP REST API covers all data flows:

| Data flow | How to handle it |
|---|---|
| New post to classify | `POST /classify` (sync) or `POST /classify/batch` (async + poll) |
| Get classification result | Response body from `/classify`, or poll `/classify/batch/{job_id}` |
| Analyst correction | `POST /feedback` |
| Trigger retraining | `POST /retrain/trigger` |
| Monitor model drift | Poll `GET /metrics` every 60 minutes, check `psi` per language |

A message bus (Redis Streams or Kafka) will be introduced in Phase 2 when ingestion volume exceeds what synchronous HTTP can handle. The topic names and message schemas are already defined. When that time comes, the existing endpoint contracts will not change — the bus will sit in front of them.

---

## Quick Reference — curl Examples

```bash
# Health check (no auth)
curl http://localhost:8000/health

# Classify a post
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -H "X-ML-API-Key: <your key>" \
  -d '{
    "post_id": "test-001",
    "content": "Vaccine causes infertility in women",
    "language": "en",
    "platform": "twitter",
    "kb_snippets": []
  }'

# Submit analyst feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -H "X-ML-API-Key: <your key>" \
  -d '{
    "post_id": "test-001",
    "original_label": "misinformation",
    "corrected_label": "factual",
    "analyst_role": "senior_analyst",
    "confidence_was": 0.91
  }'

# Get model metrics
curl http://localhost:8000/metrics \
  -H "X-ML-API-Key: <your key>"

# Trigger retraining
curl -X POST http://localhost:8000/retrain/trigger \
  -H "Content-Type: application/json" \
  -H "X-ML-API-Key: <your key>" \
  -d '{
    "triggered_by": "super_admin",
    "reason": "Monthly schedule",
    "min_samples": 500
  }'
```

---

*ImmuniWatch Nigeria — ML Service API v1.0.0 | NPHCDA / AHFIDA AI Labs | Confidential*

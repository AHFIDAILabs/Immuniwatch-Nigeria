# ImmuniWatch Nigeria — Technical Documentation

**Organisation:** AHFID AI & Social Informatics Team (AHFIDAILabs)  
**System:** ImmuniWatch Nigeria — Vaccine Misinformation Detection Platform  
**Environment:** Production (HuggingFace Spaces) / Development (local)  
**Last Updated:** June 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites and Local Setup](#4-prerequisites-and-local-setup)
5. [Environment Variables](#5-environment-variables)
6. [Ingestion Pipeline](#6-ingestion-pipeline)
7. [API Reference](#7-api-reference)
8. [Quota and Credit Management](#8-quota-and-credit-management)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Counter-Narrative System](#10-counter-narrative-system)
11. [Knowledge Base](#11-knowledge-base)
12. [Operational Runbook](#12-operational-runbook)
13. [Security and Privacy](#13-security-and-privacy)
14. [Upgrading to Paid Tiers](#14-upgrading-to-paid-tiers)

---

## 1. Project Overview

ImmuniWatch Nigeria is a real-time vaccine misinformation detection and counter-narrative platform built for the National Primary Health Care Development Agency (NPHCDA). The system continuously monitors social media platforms for vaccine-related misinformation targeting Nigerian audiences, classifies content using a fine-tuned machine learning model, and supports a Human-in-the-Loop (HITL) workflow for dispatching evidence-based counter-narratives.

**Supported languages:** English, Nigerian Pidgin (PCM), Hausa (HA), Yoruba (YO), Igbo (IG)

**Monitored platforms:** YouTube (comments), Bluesky, Twitter/X (via SociaVault)

**Core capabilities:**
- Automated misinformation detection with confidence scoring
- Multilingual content classification (5 Nigerian languages)
- Real-time ingestion from three social media platforms
- RAG-powered counter-narrative generation (Groq LLaMA-3.3-70B)
- Human-in-the-Loop review and dispatch workflow
- Knowledge base management for trusted health sources

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 HuggingFace Space (Docker)               │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              FastAPI ML Service (port 7860)       │   │
│  │                                                  │   │
│  │  ┌─────────────┐   ┌──────────────────────────┐  │   │
│  │  │ ONNX        │   │  Ingestion Worker        │  │   │
│  │  │ Classifier  │   │  (background thread)     │  │   │
│  │  └─────────────┘   │                          │  │   │
│  │                    │  ┌──────────────────┐    │  │   │
│  │  ┌─────────────┐   │  │ BlueskyConnector │    │  │   │
│  │  │ RAG Engine  │   │  │ YouTubeConnector │    │  │   │
│  │  │ (FAISS)     │   │  │ SociaVault       │    │  │   │
│  │  └─────────────┘   │  │ Connector        │    │  │   │
│  │                    │  └──────────────────┘    │  │   │
│  │  ┌─────────────┐   └──────────────────────────┘  │   │
│  │  │ SQLite DB   │                                  │   │
│  │  │ (HITL queue)│                                  │   │
│  │  └─────────────┘                                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ▲                          ▲
         │ REST API                 │ Social Media APIs
         │                          │
┌────────────────┐        ┌─────────────────────┐
│ Frontend       │        │ YouTube Data API v3  │
│ (Render.com)   │        │ Bluesky AT Protocol  │
└────────────────┘        │ SociaVault API       │
                          └─────────────────────┘
```

**Technology stack:**
- Runtime: Python 3.11, FastAPI, Uvicorn
- ML model: ONNX (fine-tuned transformer, hosted on HuggingFace Hub)
- Vector store: FAISS (in-memory, rebuilt on startup)
- Embeddings: sentence-transformers
- LLM (counter-narrative): Groq — llama-3.3-70b-versatile
- Deduplication: exact SHA-256 hash (24h TTL) + optional MinHash LSH
- Database: SQLite (HITL counter-narrative queue)
- Deployment: HuggingFace Spaces (Docker)
- CI/CD: GitHub Actions

---

## 3. Repository Structure

```
immuniwatch_project/
├── src/
│   ├── api/
│   │   ├── main.py                    # FastAPI app, lifespan, middleware
│   │   ├── routes.py                  # /classify, /classify/batch, /feed, /feedback
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   ├── counter_narrative_routes.py # /counter-narrative/* endpoints
│   │   ├── counter_narrative_store.py # SQLite HITL queue
│   │   └── kb_routes.py               # /knowledge-base/* endpoints
│   ├── ingestion/
│   │   ├── connectors/
│   │   │   ├── base.py                # BaseConnector, RawPost dataclass
│   │   │   ├── bluesky.py             # AT Protocol connector
│   │   │   ├── youtube.py             # YouTube Data API v3 connector
│   │   │   └── sociavault.py          # SociaVault Twitter/X connector
│   │   ├── deduplication.py           # Hash + TTL deduplicator
│   │   └── direct_runner.py           # Ingestion worker entry point
│   ├── models/
│   │   └── classifier.py              # ONNX inference, language detection
│   └── intelligence/
│       └── rag.py                     # FAISS vector store, Groq generation
├── models/
│   └── onnx/                          # ONNX model files (gitignored, downloaded at startup)
│       ├── immuniwatch_classifier.onnx
│       ├── immuniwatch_classifier.onnx.data
│       ├── thresholds.json
│       └── model_config.json
├── tests/                             # pytest test suite
├── .github/
│   └── workflows/
│       ├── ci.yml                     # ruff lint + pytest
│       └── deploy.yml                 # Upload to HuggingFace Spaces
├── dashboard.html                     # Served at GET /dashboard
├── Dockerfile                         # HuggingFace Spaces container
├── requirements.txt
└── DOCUMENTATION.md                   # This file
```

---

## 4. Prerequisites and Local Setup

### Requirements
- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/AHFIDAILabs/Immuniwatch-Nigeria.git
cd Immuniwatch-Nigeria
pip install -r requirements.txt
```

### Environment file

Create a `.env` file in the project root (never commit this file):

```
API_KEY=your_ml_service_api_key
YOUTUBE_API_KEY=your_youtube_data_api_key
BLUESKY_HANDLE=your_bluesky_handle
BLUESKY_APP_PASSWORD=your_bluesky_app_password
SOCIAVAULT_API_KEY=your_sociavault_api_key
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
ML_SERVICE_URL=http://localhost:7860
MODEL_VERSION=v1.0.0
```

### Running locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 7860 --reload
```

### Running tests

```bash
pytest tests/ -q
```

---

## 5. Environment Variables

All secrets are stored as HuggingFace Space secrets and GitHub Actions secrets. Never hardcode values in source code or commit them to version control.

### Required secrets (HuggingFace Space)

| Variable | Description | Where to obtain |
|----------|-------------|-----------------|
| `API_KEY` | Authentication key for all ML service endpoints | Set by team — any strong random string |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key | Google Cloud Console |
| `BLUESKY_HANDLE` | Bluesky account handle (e.g. `user.bsky.social`) | Bluesky account settings |
| `BLUESKY_APP_PASSWORD` | Bluesky app password (not account password) | Bluesky → Settings → App Passwords |
| `SOCIAVAULT_API_KEY` | SociaVault API key for Twitter/X search | SociaVault dashboard |
| `GROQ_API_KEY` | Groq API key for counter-narrative generation | console.groq.com |
| `HF_TOKEN` | HuggingFace token (Write) for model file download | huggingface.co → Settings → Access Tokens |

### Required secrets (GitHub Actions)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Same HuggingFace Write token — used by deploy workflow to upload code to Space |

### Optional tuning variables (HuggingFace Space)

| Variable | Default | Description |
|----------|---------|-------------|
| `YOUTUBE_POLL_INTERVAL` | `8100` | Seconds between YouTube polls (8100s = 11 polls/day = 90.6% quota) |
| `BLUESKY_POLL_INTERVAL` | `30` | Seconds between Bluesky polls |
| `SOCIAVAULT_POLL_INTERVAL` | `21600` | Seconds between SociaVault polls (21600s = 4 polls/day) |
| `MODEL_VERSION` | `v1.0.0` | Reported in /health response |
| `DEVICE` | `cpu` | Inference device |
| `RATE_LIMIT_REQUESTS` | `60` | Max API requests per window |
| `RATE_LIMIT_WINDOW_S` | `60` | Rate limit window in seconds |

---

## 6. Ingestion Pipeline

### Overview

The ingestion worker runs as a background daemon thread inside the FastAPI process. On startup, it initialises all three connectors. Each connector runs in its own thread, polling its platform on a configurable interval and passing new posts to the classifier via the `on_post` callback.

```
Connector thread
    └── _poll_loop()
         └── _poll_once()
              └── fetch posts from platform API
                   └── deduplicate
                        └── on_post(raw_post)
                             └── POST /classify (HTTP, with retry)
                                  └── result stored in live feed
```

### Deduplication

Each post is deduplicated before classification using a two-level check:
1. **Exact hash** — SHA-256 of the post content, with a 24-hour TTL
2. Posts that match an existing hash within the TTL window are silently dropped

This prevents the same post appearing multiple times if the same content is returned across multiple poll cycles or search queries.

### BlueskyConnector

| Property | Value |
|----------|-------|
| Protocol | AT Protocol (app.bsky.feed.searchPosts) |
| Auth | JWT session (accessJwt), refreshed automatically on 401 |
| Poll interval | 30 seconds (configurable via `BLUESKY_POLL_INTERVAL`) |
| Search terms | 8 vaccine-related terms across all 5 Nigerian languages |
| Rate limit | None (free, ~30,000 API calls/day within AT Protocol limits) |
| Startup behaviour | Auth failure is non-fatal — connector starts and retries auth on next poll cycle |

**Search terms monitored:**
- vaccine Nigeria, vaccination Nigeria, NPHCDA vaccine
- rigakafi, ajesara, vakin Nigeria
- polio vaccine Nigeria, COVID vaccine Nigeria

### YouTubeConnector

| Property | Value |
|----------|-------|
| Protocol | YouTube Data API v3 |
| Auth | API key (no OAuth required) |
| Poll interval | 8100 seconds / ~2h 15m (configurable via `YOUTUBE_POLL_INTERVAL`) |
| Search queries | 8 queries (English + Hausa + Yoruba terms) |
| Comments per video | Up to 50 (top 3 videos per query) |
| Quota cost per poll | 824 units (800 search + 24 comments) |
| Polls per day | 11 |
| Daily quota usage | 9,064 units / 90.6% of 10,000 free tier limit |
| Quota reset | Midnight Pacific Time, every day |

**Search queries monitored:**
- vaccine Nigeria, vaccination Nigeria, rigakafi Nigeria
- NPHCDA vaccine, COVID vaccine Nigeria, polio vaccine Nigeria
- HPV vaccine Nigeria, ajesara Nigeria

**Quota protection:** If a 403 `quotaExceeded` response is received mid-poll, the connector logs a warning and skips all remaining searches in that cycle. It resumes normally at the next poll interval after the quota resets.

### SociaVaultConnector

| Property | Value |
|----------|-------|
| Protocol | SociaVault REST API (Twitter/X Search) |
| Endpoint | `GET /v1/scrape/twitter/search` |
| Auth | `X-API-Key` header |
| Poll interval | 21600 seconds / 6 hours (configurable via `SOCIAVAULT_POLL_INTERVAL`) |
| Keywords | 5 vaccine misinformation keywords |
| Search type | `Latest` (most recent tweets, avoids People carousel) |
| Cost | 1 credit per API call |
| Daily usage | 20 credits/day (5 keywords × 4 polls) |
| Free tier credits | 50 total (~2.5 days) |

**Keywords monitored:**
- vaccine infertility, vaccine kills, vaccine microchip
- COVID vaccine dangerous, polio vaccine Nigeria

**Response parsing:** The SociaVault response follows the native Twitter timeline structure. The connector navigates `data.result.timeline.instructions[].entries[]` and extracts only `TimelineTweet` type entries, skipping People carousels and promoted content.

---

## 7. API Reference

All endpoints except `/`, `/health`, and `/dashboard` require the `X-ML-API-Key` header.

### Authentication

```
X-ML-API-Key: <API_KEY>
```

Requests without a valid key receive `401 Unauthorized`. Exceeding 60 requests per 60-second window receives `429 Too Many Requests` with a `Retry-After` header.

### Core endpoints

#### `GET /`
Returns service info. No authentication required.

#### `GET /health`
Returns model status and uptime. No authentication required.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0.0",
  "device": "cpu",
  "uptime_s": 3600
}
```

#### `GET /dashboard`
Serves the HTML monitoring dashboard. No authentication required.

#### `POST /classify`
Classifies a single post.

**Request:**
```json
{
  "post_id": "abc123",
  "content": "Vaccines cause infertility",
  "language": "en",
  "platform": "twitter",
  "location": "Lagos",
  "author_handle": "@user",
  "kb_snippets": []
}
```

**Response:**
```json
{
  "post_id": "abc123",
  "label": "misinformation",
  "confidence": 0.94,
  "model_version": "v1.0.0",
  "alternatives": [...],
  "kb_evidence": [...]
}
```

#### `POST /classify/batch`
Submits a batch of posts for asynchronous classification. Returns a `job_id`.

#### `GET /classify/batch/{job_id}`
Returns the status and results of a batch classification job.

#### `GET /feed`
Returns the last 100 classified posts (newest first) with cumulative count.

#### `POST /feedback`
Submits a label correction for a post (used for model retraining pipeline).

### Counter-narrative endpoints

#### `GET /counter-narrative/pending`
Returns all counter-narratives awaiting HITL review (max 100).

#### `GET /counter-narrative/{post_id}`
Returns the counter-narrative record for a specific post.

#### `POST /counter-narrative/generate`
Generates a counter-narrative for a given misinformation post using RAG + Groq.

#### `POST /counter-narrative/{id}/deploy`
Approves and marks a counter-narrative as dispatched (HITL action).

#### `DELETE /counter-narrative/{id}`
Rejects and removes a counter-narrative from the queue (HITL action).

### Knowledge base endpoints

#### `GET /knowledge-base/status`
Returns knowledge base readiness, document count, and index status.

#### `GET /knowledge-base/documents`
Returns all indexed documents with metadata.

#### `POST /knowledge-base/upload`
Adds a new document to the knowledge base and indexes it for RAG retrieval.

**Request:**
```json
{
  "title": "WHO Vaccine Safety Statement",
  "content": "...",
  "source": "WHO",
  "url": "https://who.int/...",
  "language": "en"
}
```

#### `DELETE /knowledge-base/{doc_id}`
Removes a document from the knowledge base and rebuilds the index.

---

## 8. Quota and Credit Management

### YouTube Data API v3

- **Free tier:** 10,000 units per day, resets at midnight Pacific Time
- **Reset cadence:** Daily — the quota is not a finite pool; it restores every 24 hours
- **Current configuration:** 8,100s interval → 11 polls/day → 9,064 units/day (90.6%)
- **Safety buffer:** 936 units covers one unexpected Space restart (824 units per poll)
- **No expiry:** The free tier is available indefinitely at the same daily limit

| Interval | Polls/day | Units/day | Utilisation |
|----------|-----------|-----------|-------------|
| 10,800s (3h) | 8 | 6,592 | 65.9% |
| **8,100s (2h 15m)** | **11** | **9,064** | **90.6% ← current** |
| 7,200s (2h) | 12 | 9,888 | 98.9% (too close) |

### Bluesky AT Protocol

- **Free tier:** No meaningful quota limitation for our use case
- **Current configuration:** 30s interval → ~2,880 polls/day → ~23,040 API calls/day
- **AT Protocol global limit:** ~30,000 authenticated requests/day — current usage is within limit
- **No expiry:** Free and unlimited for our search volume

### SociaVault (Twitter/X)

- **Free tier:** 50 credits total (one-time allocation, does not reset)
- **Cost:** 1 credit per API call
- **Current configuration:** 5 keywords × 4 polls/day = 20 credits/day → exhausted in ~2.5 days
- **Paid tier:** $199 for 75,000 credits (see Section 14)

---

## 9. CI/CD Pipeline

### Overview

Every push to the `main` branch triggers an automated two-stage pipeline:

```
git push → CI workflow (ruff + pytest) → Deploy workflow (upload to HuggingFace)
```

The deploy workflow only runs if CI passes. HuggingFace automatically rebuilds the Docker container after each upload.

### CI workflow (`ci.yml`)

1. Checks out code
2. Sets up Python 3.11
3. Installs dependencies
4. Runs `ruff check` (lint)
5. Runs `pytest tests/`

### Deploy workflow (`deploy.yml`)

1. Triggered on CI success via `workflow_run`
2. Installs `huggingface_hub`
3. Calls `api.upload_folder()` to push code to the HuggingFace Space
4. HuggingFace rebuilds the Space container automatically

### Required GitHub secret

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | HuggingFace Write token — must be set in repository Settings → Secrets and variables → Actions |

### Model files

ONNX model files are **not** stored in the repository (gitignored). They are hosted on the private HuggingFace model repository `AHFIDAILabs/immuniwatch-lora-classifier` and downloaded automatically at Space startup using `hf_hub_download`. The `HF_TOKEN` Space secret is used for this download.

### Deployment checklist

If a deploy fails, verify:
1. `HF_TOKEN` secret is set correctly in GitHub → Settings → Secrets and variables → Actions
2. `HF_TOKEN` Space secret is set correctly in HuggingFace Space settings
3. CI passed (deploy only runs on CI success)
4. The HuggingFace Space `AHFIDAILabs/immuniwatch-ml-service` exists and is accessible with the token

---

## 10. Counter-Narrative System

The counter-narrative system uses Retrieval-Augmented Generation (RAG) to produce evidence-based responses to detected misinformation.

### Workflow

```
Misinformation detected
    └── POST /counter-narrative/generate
         └── RAG: retrieve relevant KB documents (FAISS similarity search)
              └── Groq API: generate counter-narrative (llama-3.3-70b-versatile)
                   └── Store in SQLite queue (status: pending)
                        └── HITL reviewer approves or rejects
                             └── If approved: mark as dispatched
```

### Groq rate limits

Groq's free tier allows approximately 30 requests per minute. During a large startup burst (many posts classified simultaneously after a Space restart), some counter-narrative generation requests may receive a `429 Too Many Requests` response. The system logs these and the affected posts are not retried automatically. This is a startup-only condition and does not affect steady-state operation.

### HITL review

The HITL (Human-in-the-Loop) queue is accessible via the dashboard under the "HITL Review" section. Reviewers see pending counter-narratives and can:
- **Approve** — marks the narrative as reviewed and dispatched
- **Reject** — removes the narrative from the queue

All approved counter-narratives are tracked in the SQLite database for audit purposes.

---

## 11. Knowledge Base

The knowledge base stores trusted health information documents used for RAG retrieval during counter-narrative generation.

- Documents are indexed using sentence-transformer embeddings stored in a FAISS in-memory vector index
- The index is rebuilt in memory on each Space startup from the stored documents
- Documents can be added or removed via the `/knowledge-base` API endpoints
- Supported languages: en, ha, yo, ig, pcm

**Recommended sources for documents:**
- WHO vaccine safety statements
- NPHCDA official communications
- NAFDAC approved drug and vaccine information
- Peer-reviewed vaccine safety studies

---

## 12. Operational Runbook

### Checking system status

1. Open the dashboard: `https://<space-url>/dashboard`
2. Check `GET /health` — if `model_loaded: false`, the ONNX model is still downloading
3. Check `GET /feed` — if empty, connectors may not have ingested yet (wait for first poll cycle)

### Ingestion not increasing (posts stuck)

**Likely causes and resolutions:**

| Symptom | Cause | Resolution |
|---------|-------|------------|
| YouTube count stuck, no increase for 3+ hours | Quota exceeded | Wait for midnight PT quota reset |
| YouTube count stuck for ~2h 15m then jumps | Normal — expected between poll cycles | No action needed |
| Bluesky count not increasing | No new content matching search terms on Bluesky | Normal — content is sparse |
| Twitter count at zero | SociaVault credits exhausted | Subscribe to paid plan ($199) |
| All counts stuck after Space restart | Space restarted during deploy; running old code | Wait for deploy to complete before restarting |

### Connector died permanently

If a connector stops producing posts entirely and does not recover:
1. Check HuggingFace Space logs for error messages
2. Verify API credentials are still valid (keys can expire or be revoked)
3. Restart the Space — connectors reinitialise automatically on startup
4. If Bluesky auth fails at startup, the connector recovers automatically on the next 30-second poll cycle

### Deploy failed on GitHub Actions

1. Open GitHub → Actions → "Deploy to HuggingFace Spaces" — click the failed run
2. Check the error message
3. Most common cause: `HF_TOKEN` secret missing or expired in GitHub repository settings
4. Fix: Go to GitHub → Settings → Secrets → update `HF_TOKEN` with a fresh HuggingFace Write token
5. Also update `HF_TOKEN` in HuggingFace Space secrets (used for model file downloads)
6. Re-run failed jobs from the Actions page

### HuggingFace token refresh procedure

1. Go to huggingface.co → Settings → Access Tokens
2. Find `immuniwatch-deploy` → click ⋮ → "Invalidate and refresh"
3. Copy the new token immediately (shown only once)
4. Update `HF_TOKEN` in GitHub → Settings → Secrets and variables → Actions
5. Update `HF_TOKEN` in HuggingFace Space → Settings → Variables and secrets
6. Re-run any failed deploy workflows

---

## 13. Security and Privacy

### API key security

- The `X-ML-API-Key` value is **never logged** — only its presence or absence is recorded
- API keys are stored exclusively as HuggingFace Space secrets or GitHub Actions secrets
- The `.env` file must never be committed to the repository (it is gitignored)
- API key values must never be shared via email, Slack, or any chat platform — use Signal or WhatsApp for out-of-band sharing

### Author privacy

- No author personally identifiable information (PII) is stored
- Author IDs are hashed using SHA-256 before storage: `hash_author(author_id)`
- The original author ID is never persisted to disk or logged

### Data retention

- Deduplication hashes are held in memory with a 24-hour TTL and are lost on Space restart
- Counter-narrative records are persisted in SQLite until explicitly deleted via the API
- The live feed holds the last 100 classified posts in memory only; it resets on restart

### CORS

The API accepts requests from any origin (`*`) to allow the dashboard (hosted on Render.com) to call the ML service directly. This is intentional for the current architecture.

---

## 14. Upgrading to Paid Tiers

### SociaVault paid plan ($199 — 75,000 credits)

When upgrading from the free tier (50 credits) to the paid plan:

1. Subscribe via the SociaVault dashboard
2. Update `SOCIAVAULT_API_KEY` in HuggingFace Space secrets if a new key is issued
3. Optionally reduce `SOCIAVAULT_POLL_INTERVAL` for more frequent Twitter monitoring:

| Interval | Polls/day | Credits/day | Notes |
|----------|-----------|-------------|-------|
| 21,600s (6h) — current | 4 | 20 | Conservative |
| 3,600s (1h) | 24 | 120 | Good production coverage |
| 1,800s (30m) | 48 | 240 | High frequency |

4. Optionally expand `VACCINE_KEYWORDS` in `sociavault.py` beyond the current 5 to improve coverage (each additional keyword = 4 extra credits/day at 6h interval)

### YouTube production upgrade

The YouTube free tier (10,000 units/day) is sufficient for the current search volume. If the number of search queries is significantly expanded in the future, request a quota increase via the Google Cloud Console.

### HuggingFace Spaces upgrade

The current Space runs on a free CPU instance. For production scale and guaranteed uptime, upgrade to a persistent CPU or GPU instance via the HuggingFace Spaces billing settings.

---

*This document covers the system as deployed in June 2026. Update the SociaVault section when upgrading to the paid plan, and update quota tables if poll intervals are reconfigured.*

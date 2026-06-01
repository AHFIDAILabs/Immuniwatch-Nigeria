import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

cn_router = APIRouter(prefix="/counter-narrative", tags=["counter-narrative"])


class DeployRequest(BaseModel):
    approved_text: str


class GenerateRequest(BaseModel):
    post_id:           str
    content:           str
    platform:          str
    language:          Optional[str] = None
    author_handle:     str = ""
    original_post_cid: str = ""


@cn_router.get("/pending")
async def get_pending(limit: int = 50):
    from src.api.counter_narrative_store import get_pending
    items = get_pending(min(limit, 100))
    return {"items": items, "count": len(items)}


@cn_router.get("/{post_id}")
async def get_by_post_id(post_id: str):
    from src.api.counter_narrative_store import get_by_post_id as _get
    item = _get(post_id)
    if item is None:
        raise HTTPException(status_code=404, detail="No counter-narrative found for this post_id")
    return item


@cn_router.post("/generate")
async def generate_on_demand(body: GenerateRequest):
    from src.intelligence.counter import generate_counter_response
    from src.intelligence.rag import RAGRetriever
    from src.api.counter_narrative_store import queue_post, get_by_post_id

    existing = get_by_post_id(body.post_id)
    if existing and existing.get("generated_short"):
        return existing

    retriever = RAGRetriever()
    evidence  = retriever.retrieve(body.content, language=body.language)
    snippets  = [e.snippet for e in evidence]
    sources   = [e.source_url for e in evidence if e.source_url]

    counter = generate_counter_response(
        post_id=           body.post_id,
        claim=             body.content,
        language=          body.language or "en",
        evidence_snippets= snippets,
        source_urls=       sources,
    )
    if counter is None:
        raise HTTPException(status_code=503, detail="Counter-narrative generation failed")

    queue_post(
        post_id=           body.post_id,
        platform=          body.platform,
        author_handle=     body.author_handle,
        original_post_uri= body.post_id,
        original_post_cid= body.original_post_cid,
        content_snippet=   body.content[:280],
        label=             "misinformation",
        confidence=        1.0,
        language=          body.language or "en",
        generated_short=   counter.short,
        generated_medium=  counter.medium,
        generated_long=    counter.long,
        sources=           counter.sources,
    )

    return {
        "post_id":          body.post_id,
        "generated_short":  counter.short,
        "generated_medium": counter.medium,
        "generated_long":   counter.long,
        "sources":          counter.sources,
        "status":           "pending",
    }


@cn_router.get("/history")
async def get_history(limit: int = 50):
    from src.api.counter_narrative_store import get_history
    items = get_history(min(limit, 200))
    return {"items": items, "count": len(items)}


@cn_router.post("/{post_id}/deploy")
async def deploy(post_id: str, body: DeployRequest):
    from src.api.counter_narrative_store import (
        get_pending, mark_deployed, mark_failed,
    )
    from src.ingestion.replier.registry import get_replier

    pending = get_pending(limit=500)
    record  = next((p for p in pending if p["post_id"] == post_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Post not found in pending queue")

    if not body.approved_text or len(body.approved_text.strip()) < 5:
        raise HTTPException(status_code=422, detail="approved_text is too short")

    platform = record["platform"]
    replier  = get_replier(platform)

    if replier is None:
        raise HTTPException(status_code=422, detail=f"No replier available for platform: {platform}")

    result = replier.reply(
        original_post_id=  record["original_post_uri"] or record["post_id"],
        original_post_cid= record["original_post_cid"],
        author_handle=     record["author_handle"],
        text=              body.approved_text.strip(),
    )

    if result.success and not result.error:
        mark_deployed(post_id, reply_uri=result.reply_uri, manual_url=result.manual_url)
        return {
            "status":     "deployed",
            "post_id":    post_id,
            "platform":   platform,
            "reply_uri":  result.reply_uri,
            "manual_url": result.manual_url,
        }
    elif result.manual_url:
        mark_deployed(post_id, manual_url=result.manual_url)
        return {
            "status":     "manual_required",
            "post_id":    post_id,
            "platform":   platform,
            "manual_url": result.manual_url,
            "message":    result.error,
        }
    else:
        mark_failed(post_id, error=result.error)
        raise HTTPException(status_code=502, detail=f"Reply failed: {result.error}")


@cn_router.post("/{post_id}/skip")
async def skip(post_id: str):
    from src.api.counter_narrative_store import get_pending, mark_skipped

    pending = get_pending(limit=500)
    record  = next((p for p in pending if p["post_id"] == post_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Post not found in pending queue")

    mark_skipped(post_id)
    return {"status": "skipped", "post_id": post_id}

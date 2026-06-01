import logging
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

cn_router = APIRouter(prefix="/counter-narrative", tags=["counter-narrative"])


class DeployRequest(BaseModel):
    approved_text: str


@cn_router.get("/pending")
async def get_pending(limit: int = 50):
    from src.api.counter_narrative_store import get_pending
    items = get_pending(min(limit, 100))
    return {"items": items, "count": len(items)}


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

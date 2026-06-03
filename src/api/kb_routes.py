import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

kb_router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])


class UploadRequest(BaseModel):
    title:    str
    content:  str
    source:   str
    url:      str = ""
    language: str = "en"
    doc_id:   Optional[str] = None


@kb_router.get("/status")
async def kb_status():
    from src.intelligence.rag import get_kb_status
    return get_kb_status()


@kb_router.get("/documents")
async def kb_documents():
    from src.intelligence.rag import get_kb_status
    data = get_kb_status()
    return {
        "documents":      data["documents"],
        "document_count": data["document_count"],
        "ready":          data["ready"],
    }


@kb_router.post("/upload")
async def kb_upload(body: UploadRequest):
    from src.intelligence.rag import add_document_to_kb

    if not body.title or not body.content:
        raise HTTPException(status_code=422, detail="title and content are required")

    if len(body.content.strip()) < 10:
        raise HTTPException(status_code=422, detail="content is too short")

    doc_id = body.doc_id or str(uuid.uuid4())
    result = add_document_to_kb(
        doc_id=   doc_id,
        title=    body.title,
        content=  body.content,
        source=   body.source,
        url=      body.url,
        language= body.language,
    )

    if not result["success"]:
        raise HTTPException(status_code=503, detail=result.get("error", "Indexing failed"))

    return {
        "doc_id":         doc_id,
        "title":          body.title,
        "chunks_indexed": result["chunks_indexed"],
        "status":         "indexed",
        "language":       body.language,
        "source":         body.source,
    }


@kb_router.delete("/{doc_id}")
async def kb_delete(doc_id: str):
    from src.intelligence.rag import remove_document_from_kb
    result = remove_document_from_kb(doc_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return {"doc_id": doc_id, "status": "removed", "chunks_removed": result["chunks_removed"]}

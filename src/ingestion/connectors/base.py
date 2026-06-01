import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
@dataclass
class RawPost:
    post_id:      str
    platform:     str      # twitter|facebook|youtube|submission|bluesky
    content_text: str
    content_type: str      # TEXT|IMAGE_WITH_CAPTION|VIDEO|AUDIO
    author_hash:  str      # SHA-256 of author ID — no PII stored
    language:     Optional[str]   # en|pcm|ha|yo|ig — None means auto-detect
    timestamp:    datetime
    ingestion_ts: datetime
    raw_url:           Optional[str] = None
    location_raw:      Optional[str] = None
    likes:             Optional[int] = None
    shares:            Optional[int] = None
    author_handle:     str = ""
    original_post_cid: str = ""

    def to_kafka_message(self) -> dict:
        return {
            "schema_version":    "1.0",
            "post_id":           self.post_id,
            "platform":          self.platform,
            "content":           self.content_text,
            "content_type":      self.content_type,
            "author_hash":       self.author_hash,
            "language":          self.language,
            "raw_url":           self.raw_url,
            "location_raw":      self.location_raw,
            "likes":             self.likes,
            "shares":            self.shares,
            "author_handle":     self.author_handle,
            "original_post_cid": self.original_post_cid,
            "timestamp":         self.timestamp.isoformat(),
            "ingestion_ts":      self.ingestion_ts.isoformat(),
        }


# ---------------------------------------------------------------------------
# Helper — hash author ID so no PII is stored (Section 7.2)
# ---------------------------------------------------------------------------
def hash_author(author_id: str) -> str:
    return hashlib.sha256(str(author_id).encode()).hexdigest()


# ---------------------------------------------------------------------------
# BaseConnector — all connectors extend this
# ---------------------------------------------------------------------------
class BaseConnector(ABC):

    def __init__(self, on_post: Callable[[RawPost], None]):
        self.on_post  = on_post
        self._running = False

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return self._running

    def _safe_on_post(self, post: RawPost) -> None:
        try:
            self.on_post(post)
        except Exception as e:
            log.error(
                "%s: on_post callback failed for post_id=%s: %s",
                self.__class__.__name__,
                post.post_id,
                e,
            )
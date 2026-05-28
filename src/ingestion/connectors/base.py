import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RawPost — standard post format across all platforms
# Matches Avro schema in system design Section 4.4
# ---------------------------------------------------------------------------
@dataclass
class RawPost:
    """
    Standard post format regardless of source platform.
    Every connector produces RawPost objects.
    Worker consumes RawPost objects.
    Neither cares which platform produced it.
    """
    post_id:      str
    platform:     str      # twitter|facebook|youtube|submission|bluesky
    content_text: str
    content_type: str      # TEXT|IMAGE_WITH_CAPTION|VIDEO|AUDIO
    author_hash:  str      # SHA-256 of author ID — no PII stored
    language:     str      # en|pcm|ha|yo|ig
    timestamp:    datetime
    ingestion_ts: datetime
    raw_url:      Optional[str] = None
    location_raw: Optional[str] = None
    likes:        Optional[int] = None
    shares:       Optional[int] = None

    def to_kafka_message(self) -> dict:
        """
        Serialise to dict for Kafka publishing.
        Datetimes converted to ISO strings.
        """
        return {
            "schema_version": "1.0",
            "post_id":        self.post_id,
            "platform":       self.platform,
            "content":        self.content_text,
            "content_type":   self.content_type,
            "author_hash":    self.author_hash,
            "language":       self.language,
            "raw_url":        self.raw_url,
            "location_raw":   self.location_raw,
            "likes":          self.likes,
            "shares":         self.shares,
            "timestamp":      self.timestamp.isoformat(),
            "ingestion_ts":   self.ingestion_ts.isoformat(),
        }


# ---------------------------------------------------------------------------
# Helper — hash author ID so no PII is stored (Section 7.2)
# ---------------------------------------------------------------------------
def hash_author(author_id: str) -> str:
    """SHA-256 hash of author identifier. Never store raw author IDs."""
    return hashlib.sha256(str(author_id).encode()).hexdigest()


# ---------------------------------------------------------------------------
# BaseConnector — all connectors extend this
# ---------------------------------------------------------------------------
class BaseConnector(ABC):
    """
    Abstract base class for all platform connectors.

    Each connector:
      - Runs in its own background thread
      - Calls on_post() for every new post it finds
      - Handles its own errors without crashing others
      - Reads credentials from environment variables only

    Swapping connectors:
      Create a new class extending BaseConnector,
      implement start() and stop(),
      add the API key to .env.
      Nothing else in the pipeline changes.
    """

    def __init__(self, on_post: Callable[[RawPost], None]):
        """
        Args:
            on_post: Callback called for every new post.
                     The worker passes its publish_to_kafka
                     function here.
        """
        self.on_post  = on_post
        self._running = False

    @abstractmethod
    def start(self) -> None:
        """Start streaming or polling for posts."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop cleanly — no abrupt thread kills."""
        pass

    @property
    def is_running(self) -> bool:
        return self._running

    def _safe_on_post(self, post: RawPost) -> None:
        """
        Call on_post with error isolation.
        One bad post never stops the connector.
        """
        try:
            self.on_post(post)
        except Exception as e:
            log.error(
                "%s: on_post callback failed for post_id=%s: %s",
                self.__class__.__name__,
                post.post_id,
                e,
            )
import hashlib
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

import requests
from dotenv import load_dotenv

from src.ingestion.connectors.base import BaseConnector, RawPost, hash_author
from src.ingestion.deduplication import Deduplicator

load_dotenv()

log = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Vaccine search queries — all 5 languages per system design Section 4.6
SEARCH_QUERIES = [
    "vaccine Nigeria",
    "vaccination Nigeria",
    "rigakafi Nigeria",
    "NPHCDA vaccine",
    "COVID vaccine Nigeria",
    "polio vaccine Nigeria",
    "HPV vaccine Nigeria",
    "ajesara Nigeria",
]


class YouTubeConnector(BaseConnector):
    """
    Polls YouTube Data API v3 for vaccine-related video comments.
    Publishes new comments to Kafka via the on_post callback.
    """

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)
        self.api_key       = os.environ.get("YOUTUBE_API_KEY", "")
        self.poll_interval = int(os.environ.get("YOUTUBE_POLL_INTERVAL", 300))
        self._thread: Optional[threading.Thread] = None
        self._dedup        = Deduplicator()

        if not self.api_key:
            log.warning("YOUTUBE_API_KEY not set — connector will not start")

    def start(self) -> None:
        if not self.api_key:
            log.error("Cannot start YouTubeConnector — YOUTUBE_API_KEY missing")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="youtube-connector",
        )
        self._thread.start()
        log.info("YouTubeConnector started — polling every %ds", self.poll_interval)

    def stop(self) -> None:
        self._running = False
        log.info("YouTubeConnector stopped.")

    # ── Internal ─────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                log.error("YouTubeConnector poll error: %s", e)
            time.sleep(self.poll_interval)

    def _poll_once(self) -> None:
        for query in SEARCH_QUERIES:
            video_ids = self._search_videos(query)
            for video_id in video_ids[:3]:  # top 3 per query — saves quota
                comments = self._get_comments(video_id)
                for item in comments:
                    post = self._to_raw_post(item, video_id)
                    if post and not self._dedup.is_duplicate(
                        post.post_id, post.content_text
                    ):
                        self._safe_on_post(post)

    def _search_videos(self, query: str) -> List[str]:
        """Search YouTube. Returns list of video IDs."""
        try:
            resp = requests.get(
                f"{YOUTUBE_API_BASE}/search",
                params={
                    "key":        self.api_key,
                    "q":          query,
                    "type":       "video",
                    "part":       "id",
                    "maxResults": 5,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return [
                item["id"]["videoId"]
                for item in resp.json().get("items", [])
            ]
        except Exception as e:
            log.warning("YouTube search failed for '%s': %s", query, e)
            return []

    def _get_comments(self, video_id: str) -> list:
        """Fetch top-level comments for a video."""
        try:
            resp = requests.get(
                f"{YOUTUBE_API_BASE}/commentThreads",
                params={
                    "key":        self.api_key,
                    "videoId":    video_id,
                    "part":       "snippet",
                    "maxResults": 50,
                    "order":      "time",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("items", [])
        except Exception as e:
            log.warning("YouTube comments failed for video %s: %s", video_id, e)
            return []

    def _to_raw_post(self, item: dict, video_id: str) -> Optional[RawPost]:
        """Convert YouTube comment item to RawPost."""
        try:
            snippet    = item["snippet"]["topLevelComment"]["snippet"]
            comment_id = item["id"]
            content    = snippet.get("textOriginal", "").strip()

            if not content or len(content) < 5:
                return None

            author_id   = snippet.get(
                "authorChannelId", {}
            ).get("value", comment_id)

            ts_str = snippet.get("publishedAt", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

            return RawPost(
                post_id=      comment_id,
                platform=     "youtube",
                content_text= content,
                content_type= "TEXT",
                author_hash=  hash_author(author_id),
                language=     "en",   # YouTube does not reliably provide language
                timestamp=    ts,
                ingestion_ts= datetime.now(timezone.utc),
                raw_url=      f"https://www.youtube.com/watch?v={video_id}",
                location_raw= None,
                likes=        snippet.get("likeCount"),
                shares=       None,
            )
        except Exception as e:
            log.error("Failed to parse YouTube comment: %s", e)
            return None
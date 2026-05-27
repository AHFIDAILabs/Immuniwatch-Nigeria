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

BSKY_API_BASE = "https://bsky.social/xrpc"

# Vaccine search terms — covers all 5 languages
SEARCH_TERMS = [
    "vaccine Nigeria",
    "vaccination Nigeria",
    "NPHCDA vaccine",
    "rigakafi",
    "ajesara",
    "vakin Nigeria",
    "polio vaccine Nigeria",
    "COVID vaccine Nigeria",
]


class BlueskyConnector(BaseConnector):
    """
    Polls Bluesky for vaccine-related posts using the
    AT Protocol public API. Completely free and unlimited.
    """

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)
        self.handle        = os.environ.get("BLUESKY_HANDLE", "")
        self.app_password  = os.environ.get("BLUESKY_APP_PASSWORD", "")
        self.poll_interval = int(os.environ.get("BLUESKY_POLL_INTERVAL", 60))
        self._thread: Optional[threading.Thread] = None
        self._dedup        = Deduplicator()
        self._access_token: Optional[str] = None

        if not self.handle or not self.app_password:
            log.warning(
                "BLUESKY_HANDLE or BLUESKY_APP_PASSWORD not set "
                "— connector will not start"
            )

    def start(self) -> None:
        if not self.handle or not self.app_password:
            log.error(
                "Cannot start BlueskyConnector "
                "— BLUESKY_HANDLE or BLUESKY_APP_PASSWORD missing"
            )
            return

        if not self._authenticate():
            log.error("BlueskyConnector authentication failed — not starting")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="bluesky-connector",
        )
        self._thread.start()
        log.info(
            "BlueskyConnector started — polling every %ds", self.poll_interval
        )

    def stop(self) -> None:
        self._running = False
        log.info("BlueskyConnector stopped.")

    # ── Internal ─────────────────────────────────────────────────

    def _authenticate(self) -> bool:
        """
        Authenticate with Bluesky and store access token.
        Returns True on success, False on failure.
        """
        try:
            resp = requests.post(
                f"{BSKY_API_BASE}/com.atproto.server.createSession",
                json={
                    "identifier": self.handle,
                    "password":   self.app_password,
                },
                timeout=10,
            )
            resp.raise_for_status()
            self._access_token = resp.json().get("accessJwt")
            log.info("BlueskyConnector authenticated as %s", self.handle)
            return True
        except Exception as e:
            log.error("Bluesky authentication failed: %s", e)
            return False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                log.error("BlueskyConnector poll error: %s", e)
            time.sleep(self.poll_interval)

    def _poll_once(self) -> None:
        """Search all vaccine terms and publish new posts."""
        for term in SEARCH_TERMS:
            posts = self._search_posts(term)
            for item in posts:
                post = self._to_raw_post(item)
                if post and not self._dedup.is_duplicate(
                    post.post_id, post.content_text
                ):
                    self._safe_on_post(post)

    def _search_posts(self, term: str) -> List[dict]:
        """Search Bluesky posts for a term."""
        if not self._access_token:
            return []

        try:
            resp = requests.get(
                f"{BSKY_API_BASE}/app.bsky.feed.searchPosts",
                headers={"Authorization": f"Bearer {self._access_token}"},
                params={"q": term, "limit": 25},
                timeout=10,
            )

            # Re-authenticate if token expired
            if resp.status_code == 401:
                log.info("Bluesky token expired — re-authenticating")
                if self._authenticate():
                    resp = requests.get(
                        f"{BSKY_API_BASE}/app.bsky.feed.searchPosts",
                        headers={
                            "Authorization": f"Bearer {self._access_token}"
                        },
                        params={"q": term, "limit": 25},
                        timeout=10,
                    )
                else:
                    return []

            resp.raise_for_status()
            return resp.json().get("posts", [])

        except Exception as e:
            log.warning("Bluesky search failed for '%s': %s", term, e)
            return []

    def _to_raw_post(self, item: dict) -> Optional[RawPost]:
        """Convert Bluesky post item to RawPost."""
        try:
            record  = item.get("record", {})
            content = record.get("text", "").strip()

            if not content or len(content) < 5:
                return None

            post_id    = item.get("uri", "")
            author_did = item.get("author", {}).get("did", post_id)

            ts_str = record.get("createdAt", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

            return RawPost(
                post_id=      post_id,
                platform=     "bluesky",
                content_text= content,
                content_type= "TEXT",
                author_hash=  hash_author(author_did),
                language=     "en",
                timestamp=    ts,
                ingestion_ts= datetime.now(timezone.utc),
                raw_url=      None,
                location_raw= None,
                likes=        item.get("likeCount"),
                shares=       item.get("repostCount"),
            )
        except Exception as e:
            log.error("Failed to parse Bluesky post: %s", e)
            return None
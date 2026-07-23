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

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)
        self.handle        = os.environ.get("BLUESKY_HANDLE", "")
        self.app_password  = os.environ.get("BLUESKY_APP_PASSWORD", "")
        self.poll_interval = int(os.environ.get("BLUESKY_POLL_INTERVAL", 30))
        self._thread: Optional[threading.Thread] = None
        self._dedup        = Deduplicator()
        self._access_token: Optional[str] = None
        # Cache author DID → raw location string (None = already checked, no location).
        # Capped at 5,000 entries — when full, the oldest half is evicted so the
        # connector can run indefinitely without growing memory over months/years.
        self._profile_location_cache: dict = {}
        self._PROFILE_CACHE_MAX = 5000

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

        # Attempt initial auth but do NOT abort if it fails — the poll loop
        # retries authentication on every cycle, so a transient failure at
        # Space startup (network blip, Bluesky API briefly down) is recovered
        # automatically on the next poll rather than killing the connector forever.
        self._authenticate()

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
            data = resp.json()
            self._access_token = data.get("accessJwt")
            if self._access_token:
                log.info("BlueskyConnector authenticated as %s", self.handle)
                return True
            # Token absent — log available keys to aid diagnosis (no secret values)
            log.warning(
                "Bluesky auth succeeded but no accessJwt in response. "
                "Keys present: %s", list(data.keys())
            )
            return False
        except Exception as e:
            log.error("Bluesky authentication failed: %s", e)
            return False

    def _poll_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            try:
                poll_ok = self._poll_once()
            except Exception as e:
                log.error("BlueskyConnector poll error: %s", e)
                poll_ok = False

            if poll_ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    log.warning(
                        "Bluesky: %d consecutive failed polls — "
                        "clearing session and backing off 5 minutes",
                        consecutive_failures,
                    )
                    self._access_token = None
                    consecutive_failures = 0
                    time.sleep(300)
                    continue

            time.sleep(self.poll_interval)

    def _poll_once(self) -> bool:
        # If token is absent (startup auth failed or session was lost),
        # attempt re-authentication before searching.
        if not self._access_token:
            log.info("Bluesky token absent — attempting re-authentication")
            if not self._authenticate():
                log.warning("Bluesky re-auth failed — skipping this poll cycle")
                return False

        any_ok = False
        for term in SEARCH_TERMS:
            posts = self._search_posts(term)
            if posts is not None:
                any_ok = True
                for item in posts:
                    post = self._to_raw_post(item)
                    if post and not self._dedup.is_duplicate(
                        post.post_id, post.content_text
                    ):
                        self._safe_on_post(post)
        return any_ok

    def _search_posts(self, term: str) -> Optional[List[dict]]:
        if not self._access_token:
            return None

        try:
            resp = requests.get(
                f"{BSKY_API_BASE}/app.bsky.feed.searchPosts",
                headers={"Authorization": f"Bearer {self._access_token}"},
                params={"q": term, "limit": 25},
                timeout=10,
            )

            if resp.status_code == 401:
                # Token expired — re-auth and retry once
                log.info("Bluesky token expired — re-authenticating")
                self._access_token = None
                if self._authenticate():
                    resp = requests.get(
                        f"{BSKY_API_BASE}/app.bsky.feed.searchPosts",
                        headers={"Authorization": f"Bearer {self._access_token}"},
                        params={"q": term, "limit": 25},
                        timeout=10,
                    )
                else:
                    return None

            if resp.status_code == 400:
                # Auth token rejected — log the exact Bluesky error, then retry
                # without auth (Bluesky search is publicly accessible).
                log.warning(
                    "Bluesky 400 with auth for '%s' — Bluesky says: %s  "
                    "Retrying without auth token.",
                    term, resp.text[:400],
                )
                self._access_token = None
                resp = requests.get(
                    f"{BSKY_API_BASE}/app.bsky.feed.searchPosts",
                    params={"q": term, "limit": 25},
                    timeout=10,
                )

            if not resp.ok:
                log.warning(
                    "Bluesky HTTP %d for '%s': %s",
                    resp.status_code, term, resp.text[:300],
                )
                return None

            return resp.json().get("posts", [])

        except Exception as e:
            log.warning("Bluesky search failed for '%s': %s", term, e)
            return None

    def _get_author_location(self, author_did: str) -> Optional[str]:
        if author_did in self._profile_location_cache:
            return self._profile_location_cache[author_did]

        # Evict oldest half when cache is full to prevent unbounded memory growth
        if len(self._profile_location_cache) >= self._PROFILE_CACHE_MAX:
            evict_keys = list(self._profile_location_cache.keys())[:self._PROFILE_CACHE_MAX // 2]
            for k in evict_keys:
                del self._profile_location_cache[k]
            log.debug("Profile cache evicted %d entries", len(evict_keys))

        location = None
        if self._access_token:
            try:
                resp = requests.get(
                    f"{BSKY_API_BASE}/app.bsky.actor.getProfile",
                    headers={"Authorization": f"Bearer {self._access_token}"},
                    params={"actor": author_did},
                    timeout=5,
                )
                if resp.ok:
                    profile = resp.json()
                    # `location` is an explicit profile field on some clients;
                    # fall back to `description` (bio) which often contains city/state
                    location = (
                        profile.get("location")
                        or profile.get("description")
                        or None
                    )
            except Exception:
                pass  # network error — cache None so we don't retry immediately

        self._profile_location_cache[author_did] = location
        return location

    def _to_raw_post(self, item: dict) -> Optional[RawPost]:
        try:
            record  = item.get("record", {})
            content = record.get("text", "").strip()

            if not content or len(content) < 5:
                return None

            post_id    = item.get("uri", "")
            author     = item.get("author", {})
            author_did = author.get("did", post_id)

            ts_str = record.get("createdAt", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

            # Location: profile field / bio first, post text is the fallback
            # handled by classifier._resolve_state()
            location_raw   = self._get_author_location(author_did)
            author_handle  = author.get("handle", "")
            post_cid       = item.get("cid", "")

            return RawPost(
                post_id=           post_id,
                platform=          "bluesky",
                content_text=      content,
                content_type=      "TEXT",
                author_hash=       hash_author(author_did),
                language=          None,
                timestamp=         ts,
                ingestion_ts=      datetime.now(timezone.utc),
                raw_url=           None,
                location_raw=      location_raw,
                likes=             item.get("likeCount"),
                shares=            item.get("repostCount"),
                author_handle=     author_handle,
                original_post_cid= post_cid,
            )
        except Exception as e:
            log.error("Failed to parse Bluesky post: %s", e)
            return None
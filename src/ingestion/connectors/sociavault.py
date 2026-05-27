import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import requests
from dotenv import load_dotenv

from src.ingestion.connectors.base import BaseConnector, RawPost, hash_author
from src.ingestion.deduplication import Deduplicator

load_dotenv()

log = logging.getLogger(__name__)

SOCIAVAULT_API_BASE = "https://api.sociavault.com/v1"

# Map SociaVault source names to your platform values
# Dashboard shows original platform, not SociaVault
PLATFORM_MAP = {
    "twitter":   "twitter",
    "facebook":  "facebook",
    "instagram": "facebook",
    "tiktok":    "submission",
    "reddit":    "submission",
    "pinterest": "submission",
    "news":      "submission",
    "blog":      "submission",
}

# Vaccine keywords — all 5 languages per system design Section 4.6

VACCINE_KEYWORDS = [
    "vaccine Nigeria",
]


class SociaVaultConnector(BaseConnector):
    """
    Polls SociaVault API for vaccine-related posts across
    Twitter, Facebook, Instagram, TikTok, and Reddit.

    Platform shown on dashboard is always the original source
    platform — not SociaVault.
    """

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)
        self.api_key       = os.environ.get("SOCIAVAULT_API_KEY", "")
        self.poll_interval = int(os.environ.get("SOCIAVAULT_POLL_INTERVAL", 60))
        self._thread: Optional[threading.Thread] = None
        self._dedup        = Deduplicator()
        self._last_seen_id: Optional[str] = None

        if not self.api_key:
            log.warning("SOCIAVAULT_API_KEY not set — connector will not start")

    def start(self) -> None:
        if not self.api_key:
            log.error("Cannot start SociaVaultConnector — SOCIAVAULT_API_KEY missing")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="sociavault-connector",
        )
        self._thread.start()
        log.info(
            "SociaVaultConnector started — polling every %ds", self.poll_interval
        )

    def stop(self) -> None:
        self._running = False
        log.info("SociaVaultConnector stopped.")

    # ── Internal ─────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                log.error("SociaVaultConnector poll error: %s", e)
            time.sleep(self.poll_interval)

    def _poll_once(self) -> None:
        """Fetch latest mentions and publish new ones to Kafka."""
        for keyword in VACCINE_KEYWORDS:
            posts = self._fetch_posts(keyword)
            for raw in posts:
                post = self._to_raw_post(raw)
                if post and not self._dedup.is_duplicate(
                    post.post_id, post.content_text
                ):
                    self._safe_on_post(post)

    def _fetch_posts(self, keyword: str) -> list:
        """
        Fetch posts from SociaVault for a keyword.
        Returns empty list on any error — connector keeps running.
        """
        params = {
            "q":     keyword,
            "limit": 20,
        }

        #  FIXED: X-API-Key header per SociaVault docs
        headers = {
            "X-API-Key": self.api_key,
        }

        try:
            #  FIXED: correct endpoint per SociaVault docs
            resp = requests.get(
                f"{SOCIAVAULT_API_BASE}/scrape/twitter/search",
                headers=headers,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data  = resp.json()
            posts = data.get("posts", data.get("results", data.get("data", [])))

            if posts and isinstance(posts, list) and len(posts) > 0:
                first_id = posts[0].get("id") or posts[0].get("tweet_id")
                if first_id:
                    self._last_seen_id = str(first_id)

            return posts if isinstance(posts, list) else []

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                log.warning("SociaVault rate limit hit — waiting 60s")
                time.sleep(60)
            else:
                log.warning("SociaVault HTTP error for '%s': %s", keyword, e)
            return []
        except Exception as e:
            log.warning("SociaVault fetch failed for '%s': %s", keyword, e)
            return []

    def _to_raw_post(self, item: dict) -> Optional[RawPost]:
        """
        Convert SociaVault post dict to RawPost.
        Platform is taken from SociaVault source field so
        dashboard shows original platform (Twitter/Facebook etc).
        """
        try:
            content = (
                item.get("text") or
                item.get("content") or
                item.get("full_text") or
                item.get("body") or ""
            ).strip()

            if not content or len(content) < 5:
                return None

            post_id      = str(item.get("id") or item.get("tweet_id") or "")
            platform_raw = item.get("source", item.get("platform", "twitter")).lower()
            platform     = PLATFORM_MAP.get(platform_raw, "twitter")

            author_raw = (
                item.get("author_id") or
                item.get("user_id") or
                item.get("user", {}).get("id") or
                item.get("author") or
                post_id
            )

            ts_str = (
                item.get("published_at") or
                item.get("created_at") or
                item.get("date") or ""
            )
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

            return RawPost(
                post_id=      post_id,
                platform=     platform,
                content_text= content,
                content_type= "TEXT",
                author_hash=  hash_author(author_raw),
                language=     item.get("language", "en"),
                timestamp=    ts,
                ingestion_ts= datetime.now(timezone.utc),
                raw_url=      item.get("url"),
                location_raw= item.get("location") or item.get("region"),
                likes=        item.get("likes_count") or item.get("likes") or item.get("favorite_count"),
                shares=       item.get("shares_count") or item.get("shares") or item.get("retweet_count"),
            )
        except Exception as e:
            log.error("Failed to parse SociaVault post: %s", e)
            return None
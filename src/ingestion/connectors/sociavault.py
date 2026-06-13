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

# Vaccine keywords — Nigerian context, 5 terms chosen to stretch free-tier credits.
# Free tier: 5 keywords × 4 polls/day (every 6 h) = 20 credits/day → 50 credits lasts ~2.5 days.
# Paid tier: add more keywords and set SOCIAVAULT_POLL_INTERVAL=1800 (30 min) in Space secrets.
VACCINE_KEYWORDS = [
    "vaccine infertility",
    "vaccine kills",
    "vaccine microchip",
    "COVID vaccine dangerous",
    "polio vaccine Nigeria",
]

# Twitter timestamp format returned by SociaVault
_TWITTER_TS_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"


class SociaVaultConnector(BaseConnector):

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)
        self.api_key       = os.environ.get("SOCIAVAULT_API_KEY", "")
        self.poll_interval = int(os.environ.get("SOCIAVAULT_POLL_INTERVAL", 21600))
        self._thread: Optional[threading.Thread] = None
        self._dedup        = Deduplicator()

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
        Fetch tweets via SociaVault Twitter Search endpoint.

        Docs: GET /v1/scrape/twitter/search
        Required param: query (NOT q — q returns 400)
        Auth: X-API-Key header
        Cost: 1 credit per call
        """
        try:
            resp = requests.get(
                f"{SOCIAVAULT_API_BASE}/scrape/twitter/search",
                headers={"X-API-Key": self.api_key},
                params={
                    "query": keyword,   # required; "q" causes 400 Bad Request
                    "type":  "Latest",  # Latest avoids People carousel in Top results
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("success") or data.get("data", {}).get("error"):
                log.warning(
                    "SociaVault API returned error for '%s': %s",
                    keyword, data.get("data", {}).get("error"),
                )
                return []

            # Navigate the Twitter timeline response structure:
            # data.result.timeline.instructions[].entries[].content.itemContent
            instructions = (
                data
                .get("data", {})
                .get("result", {})
                .get("timeline", {})
                .get("instructions", [])
            )

            normalized = []
            for instruction in instructions:
                for entry in instruction.get("entries", []):
                    content = entry.get("content", {})
                    item_content = content.get("itemContent", {})

                    # Skip non-tweet entries (People carousel, promoted pins, etc.)
                    if item_content.get("__typename") != "TimelineTweet":
                        continue

                    tweet_result = (
                        item_content.get("tweet_results", {}).get("result", {})
                    )
                    if not tweet_result:
                        continue

                    legacy = tweet_result.get("legacy", {})
                    user_legacy = (
                        tweet_result
                        .get("core", {})
                        .get("user_results", {})
                        .get("result", {})
                        .get("legacy", {})
                    )

                    tweet_id = tweet_result.get("rest_id", "")
                    normalized.append({
                        "id":             tweet_id,
                        "text":           legacy.get("full_text", ""),
                        "created_at":     legacy.get("created_at", ""),
                        "favorite_count": legacy.get("favorite_count"),
                        "retweet_count":  legacy.get("retweet_count"),
                        "user_id":        legacy.get("user_id_str", ""),
                        "screen_name":    user_legacy.get("screen_name", ""),
                        "location":       user_legacy.get("location") or "",
                        "url":            f"https://twitter.com/i/web/status/{tweet_id}",
                    })

            log.info(
                "SociaVault: fetched %d tweets for '%s'", len(normalized), keyword
            )
            return normalized

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                log.warning("SociaVault rate limit hit — waiting 60s")
                time.sleep(60)
            else:
                status = e.response.status_code if e.response is not None else "?"
                log.warning(
                    "SociaVault HTTP %s for '%s': %s", status, keyword, e
                )
            return []
        except Exception as e:
            log.warning("SociaVault fetch failed for '%s': %s", keyword, e)
            return []

    def _to_raw_post(self, item: dict) -> Optional[RawPost]:
        try:
            content = item.get("text", "").strip()
            if not content or len(content) < 5:
                return None

            post_id    = str(item.get("id", ""))
            author_raw = item.get("user_id") or post_id

            ts_str = item.get("created_at", "")
            try:
                # SociaVault returns Twitter's native format: "Thu Jun 20 11:31:12 +0000 2019"
                ts = datetime.strptime(ts_str, _TWITTER_TS_FORMAT).replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                ts = datetime.now(timezone.utc)

            location = item.get("location") or None
            if location == "":
                location = None

            return RawPost(
                post_id=      post_id,
                platform=     "twitter",
                content_text= content,
                content_type= "TEXT",
                author_hash=  hash_author(author_raw),
                language=     None,
                timestamp=    ts,
                ingestion_ts= datetime.now(timezone.utc),
                raw_url=      item.get("url"),
                location_raw= location,
                likes=        item.get("favorite_count"),
                shares=       item.get("retweet_count"),
                author_handle=item.get("screen_name", ""),
            )
        except Exception as e:
            log.error("Failed to parse SociaVault post: %s", e)
            return None

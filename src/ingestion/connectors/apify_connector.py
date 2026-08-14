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

# Migration flag — set to False and implement official API methods
# when organisation upgrades to paid APIs.
APIFY_BACKEND = True

APIFY_BASE_URL = "https://api.apify.com/v2/acts"

# Daily hard caps — enforced to spread $5 across 30 days
DAILY_LIMIT_TWITTER   = 160   # ~$1.92/month
DAILY_LIMIT_INSTAGRAM = 30    # ~$1.35/month
DAILY_LIMIT_FACEBOOK  = 20    # ~$0.60/month

# Posts requested per poll cycle
BATCH_TWITTER   = 40
BATCH_INSTAGRAM = 10
BATCH_FACEBOOK  = 10

# Poll intervals in seconds
INTERVAL_TWITTER   = 21600   # 6 hours
INTERVAL_INSTAGRAM = 28800   # 8 hours
INTERVAL_FACEBOOK  = 172800  # 48 hours — 15 runs/month within $5 budget

# Search terms — rotate one per poll, do not fire all terms in a
# single poll (saves budget).
TWITTER_TERMS = [
    "vaccine Nigeria",
    "vaccination Nigeria",
    "NPHCDA",
    "polio vaccine Nigeria",
    "immunization Nigeria",
]

INSTAGRAM_HASHTAGS = [
    "Nigeria",
    "NigeriaHealth",
    "vaccination",
]

FACEBOOK_QUERY = "vaccine Nigeria"

# Timestamp formats returned by Apify actors
_TS_FORMATS = (
    "%a %b %d %H:%M:%S +0000 %Y",   # Twitter native
    "%Y-%m-%dT%H:%M:%S.%fZ",         # ISO with microseconds
    "%Y-%m-%dT%H:%M:%SZ",            # ISO without microseconds
)


class ApifyConnector(BaseConnector):

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)

        api_key = os.environ.get("APIFY_API_KEY", "")
        if not api_key:
            self._enabled = False
            self._api_key = ""
            log.warning(
                "ApifyConnector: APIFY_API_KEY absent — connector disabled. "
                "Set this secret to enable Facebook, Instagram and Twitter/X ingestion."
            )
        else:
            self._enabled = True
            self._api_key = api_key
            log.info("ApifyConnector: APIFY_API_KEY present — connector enabled.")

        self._twitter_interval   = int(os.environ.get("APIFY_TWITTER_INTERVAL",   INTERVAL_TWITTER))
        self._instagram_interval = int(os.environ.get("APIFY_INSTAGRAM_INTERVAL", INTERVAL_INSTAGRAM))
        self._facebook_interval  = int(os.environ.get("APIFY_FACEBOOK_INTERVAL",  INTERVAL_FACEBOOK))

        self._dedup   = Deduplicator()
        self._running = False
        self._lock    = threading.Lock()

        # Cycles through search terms — one term per poll, not all at once
        self._term_index: dict = {"twitter": 0, "instagram": 0}

        # Daily usage counters — reset at UTC midnight
        self._daily: dict = {
            "date":      "",
            "twitter":   0,
            "instagram": 0,
            "facebook":  0,
        }

        # Daemon threads — defined here, started only in start()
        self._thread_twitter = threading.Thread(
            target=self._poll_loop_twitter,
            daemon=True,
            name="apify-twitter",
        )
        self._thread_instagram = threading.Thread(
            target=self._poll_loop_instagram,
            daemon=True,
            name="apify-instagram",
        )
        self._thread_facebook = threading.Thread(
            target=self._poll_loop_facebook,
            daemon=True,
            name="apify-facebook",
        )

    def start(self) -> None:
        if not self._enabled:
            return

        self._running = True
        self._thread_twitter.start()
        self._thread_instagram.start()
        self._thread_facebook.start()
        log.info(
            "ApifyConnector started — Twitter every %ds, "
            "Instagram every %ds, Facebook every %ds",
            self._twitter_interval,
            self._instagram_interval,
            self._facebook_interval,
        )

    def stop(self) -> None:
        self._running = False

    # ── Internal helpers ─────────────────────────────────────────

    def _within_daily_limit(self, platform: str, batch: int) -> bool:
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily["date"] != today:
                self._daily = {
                    "date":      today,
                    "twitter":   0,
                    "instagram": 0,
                    "facebook":  0,
                }
            limits = {
                "twitter":   DAILY_LIMIT_TWITTER,
                "instagram": DAILY_LIMIT_INSTAGRAM,
                "facebook":  DAILY_LIMIT_FACEBOOK,
            }
            current = self._daily[platform]
            limit   = limits[platform]
            if current + batch > limit:
                log.warning(
                    "ApifyConnector: %s daily limit (%d) reached (%d used). "
                    "Skipping until tomorrow UTC.",
                    platform, limit, current,
                )
                return False
            self._daily[platform] += batch
            return True

    def _next_term(self, platform: str, terms: list) -> str:
        with self._lock:
            idx  = self._term_index.get(platform, 0)
            term = terms[idx % len(terms)]
            self._term_index[platform] = idx + 1
        return term

    def _call_apify(self, actor_slug: str, payload: dict) -> list:
        """Start an Apify actor run async, poll until done, return items."""
        if not APIFY_BACKEND:
            raise NotImplementedError("APIFY_BACKEND is False")

        run_url = (
            f"{APIFY_BASE_URL}/{actor_slug}/runs"
            f"?token={self._api_key}"
        )
        try:
            # Start the run (returns immediately with run ID)
            resp = requests.post(
                run_url,
                json=payload,
                timeout=30,
            )
            if resp.status_code not in (200, 201):
                log.warning(
                    "ApifyConnector: %s start failed HTTP %d",
                    actor_slug, resp.status_code,
                )
                return []

            run_data = resp.json()
            run_id = (
                run_data.get("data", {}).get("id")
                or run_data.get("id")
            )
            if not run_id:
                log.warning(
                    "ApifyConnector: no run ID from %s",
                    actor_slug,
                )
                return []

            log.info(
                "ApifyConnector: %s run started id=%s",
                actor_slug, run_id,
            )

            # Poll until SUCCEEDED or FAILED (max 5 minutes)
            status_url = (
                f"https://api.apify.com/v2/actor-runs/{run_id}"
                f"?token={self._api_key}"
            )
            for _ in range(30):  # 30 x 10s = 5 minutes max
                time.sleep(10)
                try:
                    sr = requests.get(status_url, timeout=15)
                    status = (
                        sr.json()
                        .get("data", {})
                        .get("status", "")
                    )
                except Exception:
                    continue
                if status == "SUCCEEDED":
                    break
                if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    log.warning(
                        "ApifyConnector: %s run %s status=%s",
                        actor_slug, run_id, status,
                    )
                    return []

            # Fetch dataset items
            items_url = (
                f"https://api.apify.com/v2/actor-runs/{run_id}"
                f"/dataset/items"
                f"?token={self._api_key}&format=json"
            )
            ir = requests.get(items_url, timeout=30)
            if ir.status_code != 200:
                log.warning(
                    "ApifyConnector: dataset fetch failed HTTP %d",
                    ir.status_code,
                )
                return []
            data = ir.json()
            if not isinstance(data, list):
                log.warning(
                    "ApifyConnector: unexpected dataset format "
                    "from %s", actor_slug,
                )
                return []
            log.info(
                "ApifyConnector: %s returned %d items",
                actor_slug, len(data),
            )
            return data

        except Exception as exc:
            log.error(
                "ApifyConnector: error calling %s: %s",
                actor_slug, exc,
            )
            return []

    # ── Poll loops ───────────────────────────────────────────────

    def _poll_loop_twitter(self) -> None:
        """Twitter via Apify disabled — needs cookies.
        Re-enable when official API or cookies available."""
        log.info(
            "ApifyConnector: Twitter polling disabled "
            "(requires Twitter cookies or paid API). "
            "Skipping."
        )

    def _poll_loop_instagram(self) -> None:
        while self._running:
            try:
                self._fetch_instagram()
            except Exception as exc:
                log.error("ApifyConnector instagram poll error: %s", exc)
            time.sleep(self._instagram_interval)

    def _poll_loop_facebook(self) -> None:
        while self._running:
            try:
                self._fetch_facebook()
            except Exception as exc:
                log.error("ApifyConnector facebook poll error: %s", exc)
            time.sleep(self._facebook_interval)

    # ── Fetch methods ────────────────────────────────────────────

    def _fetch_twitter(self) -> None:
        if not APIFY_BACKEND:
            raise NotImplementedError(
                "APIFY_BACKEND is False — implement Twitter official API here"
            )

        if not self._within_daily_limit("twitter", BATCH_TWITTER):
            return

        term = self._next_term("twitter", TWITTER_TERMS)
        log.debug("ApifyConnector: fetching Twitter term='%s'", term)

        payload = {
            "searchTerms": [term],
            "maxItems":    BATCH_TWITTER,
            "sort":        "Latest",
        }
        items = self._call_apify("apidojo~tweet-scraper", payload)

        ingested = 0
        for item in items:
            if not item or "noResults" in item or "error" in item:
                log.debug(
                    "ApifyConnector: skipping Twitter non-post item: %s",
                    list(item.keys()) if item else "empty",
                )
                continue
            post = self._to_raw_post_twitter(item)
            if post and not self._dedup.is_duplicate(post.post_id, post.content_text):
                self._safe_on_post(post)
                ingested += 1

        if ingested:
            log.info(
                "ApifyConnector: ingested %d new Twitter posts (term='%s')",
                ingested, term,
            )

    def _fetch_instagram(self) -> None:
        if not APIFY_BACKEND:
            raise NotImplementedError(
                "APIFY_BACKEND is False — implement Instagram official API here"
            )

        if not self._within_daily_limit("instagram", BATCH_INSTAGRAM):
            return

        tag = self._next_term("instagram", INSTAGRAM_HASHTAGS)
        log.debug("ApifyConnector: fetching Instagram hashtag='%s'", tag)

        payload = {
            "hashtags":            [tag],
            "maxPostsPerHashtag":  BATCH_INSTAGRAM,
        }
        items = self._call_apify(
            "scraping_solutions~instagram-hashtag-scraper-pro-no-cookies",
            payload,
        )

        ingested = 0
        for item in items:
            if not item or "error" in item or "errorDescription" in item:
                log.debug(
                    "ApifyConnector: skipping Instagram non-post item: %s",
                    list(item.keys()) if item else "empty",
                )
                continue
            post = self._to_raw_post_instagram(item)
            if post is None:
                log.warning(
                    "ApifyConnector: Instagram parse returned None"
                )
                continue
            is_dup = self._dedup.is_duplicate(
                post.post_id, post.content_text)
            log.info(
                "ApifyConnector: Instagram post %s | "
                "dup=%s | content=%s",
                post.post_id, is_dup,
                post.content_text[:50]
            )
            if not is_dup:
                self._safe_on_post(post)
                ingested += 1

        if ingested:
            log.info(
                "ApifyConnector: ingested %d new Instagram posts (tag='%s')",
                ingested, tag,
            )

    def _fetch_facebook(self) -> None:
        if not APIFY_BACKEND:
            raise NotImplementedError(
                "APIFY_BACKEND is False — implement Facebook official API here"
            )

        if not self._within_daily_limit("facebook", BATCH_FACEBOOK):
            return

        log.debug("ApifyConnector: fetching Facebook query='%s'", FACEBOOK_QUERY)

        payload = {
            "query":      FACEBOOK_QUERY,
            "maxResults": BATCH_FACEBOOK,
        }
        items = self._call_apify(
            "powerai~facebook-post-search-scraper", payload
        )

        ingested = 0
        for item in items:
            if not item or "error" in item:
                log.debug(
                    "ApifyConnector: skipping Facebook non-post item: %s",
                    list(item.keys()) if item else "empty",
                )
                continue
            post = self._to_raw_post_facebook(item)
            if post and not self._dedup.is_duplicate(post.post_id, post.content_text):
                self._safe_on_post(post)
                ingested += 1

        if ingested:
            log.info(
                "ApifyConnector: ingested %d new Facebook posts",
                ingested,
            )

    # ── RawPost parsers ──────────────────────────────────────────

    @staticmethod
    def _parse_ts(raw: str) -> datetime:
        """Try each known timestamp format; fall back to now() on failure."""
        for fmt in _TS_FORMATS:
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass
        return datetime.now(timezone.utc)

    def _to_raw_post_twitter(self, item: dict) -> Optional[RawPost]:
        try:
            log.info(
                "ApifyConnector: Twitter item keys: %s",
                list(item.keys())[:15]
            )
            author  = item.get("author", {})
            content = (item.get("full_text") or item.get("text", "")).strip()

            if not content or len(content) < 5:
                return None

            return RawPost(
                post_id           = str(item.get("id", "")),
                platform          = "twitter",
                content_text      = content,
                content_type      = "TEXT",
                author_hash       = hash_author(author.get("userName", "")),
                language          = item.get("lang", None),
                timestamp         = self._parse_ts(item.get("created_at", "")),
                ingestion_ts      = datetime.now(timezone.utc),
                raw_url           = item.get("url", ""),
                location_raw      = author.get("location", None) or None,
                likes             = item.get("likeCount", None),
                shares            = item.get("retweetCount", None),
                author_handle     = author.get("userName", ""),
                original_post_cid = "",
            )
        except Exception as exc:
            log.warning(
                "ApifyConnector: failed to parse Twitter item: %s", exc
            )
            return None

    def _to_raw_post_instagram(
        self, item: dict
    ) -> Optional[RawPost]:
        try:
            log.info(
                "ApifyConnector: IG raw item sample: %s",
                {k: v for k, v in item.items()
                 if k in ["raw_post_id", "caption.text",
                           "taken_at_date", "user.full_name",
                           "user.username", "link_post"]},
            )
            post_id = str(
                item.get("raw_post_id")
                or item.get("code")
                or item.get("link_post", "")
                or item.get("link_user", "")
            )
            content = (
                item.get("caption.text")
                or item.get("caption.hashtags")
                or item.get("caption", "")
                or item.get("hashtag", "")
                or "Instagram post"
            )
            author = (
                item.get("user.username")
                or item.get("user.full_name")
                or ""
            )
            ts_raw = item.get("taken_at_date", "")
            try:
                if isinstance(ts_raw, (int, float)):
                    ts = datetime.fromtimestamp(
                        ts_raw, tz=timezone.utc
                    ).replace(tzinfo=None)
                else:
                    ts = datetime.fromisoformat(
                        str(ts_raw).replace("Z", "")
                    )
            except Exception:
                ts = datetime.utcnow()

            return RawPost(
                post_id=post_id,
                platform="instagram",
                content_text=content,
                content_type="TEXT",
                author_hash=hash_author(author),
                language=None,
                timestamp=ts,
                ingestion_ts=datetime.utcnow(),
                raw_url=item.get("link_post", ""),
                location_raw=None,
                likes=item.get("like_count"),
                shares=None,
                author_handle=author,
                original_post_cid="",
            )
        except Exception as exc:
            log.warning(
                "ApifyConnector: failed to parse "
                "Instagram item: %s", exc,
            )
            return None

    def _to_raw_post_facebook(
        self, item: dict
    ) -> Optional[RawPost]:
        try:
            log.info(
                "ApifyConnector: Facebook item keys: %s",
                list(item.keys())[:20],
            )
            post_id = str(
                item.get("postId")
                or item.get("id")
                or item.get("url", "")
            )
            content = (
                item.get("text")
                or item.get("message")
                or item.get("story")
                or ""
            )
            author = (
                item.get("profileName")
                or item.get("pageName")
                or item.get("authorName")
                or ""
            )
            ts_raw = (
                item.get("time")
                or item.get("date")
                or item.get("createdTime")
                or ""
            )
            try:
                if isinstance(ts_raw, (int, float)):
                    ts = datetime.fromtimestamp(
                        ts_raw, tz=timezone.utc
                    ).replace(tzinfo=None)
                else:
                    ts = datetime.fromisoformat(
                        str(ts_raw).replace("Z", "")
                    )
            except Exception:
                ts = datetime.utcnow()

            return RawPost(
                post_id=post_id,
                platform="facebook",
                content_text=content,
                content_type="TEXT",
                author_hash=hash_author(author),
                language=None,
                timestamp=ts,
                ingestion_ts=datetime.utcnow(),
                raw_url=item.get("url", ""),
                location_raw=None,
                likes=item.get("likes")
                or item.get("likesCount"),
                shares=item.get("shares")
                or item.get("sharesCount"),
                author_handle=author,
                original_post_cid="",
            )
        except Exception as exc:
            log.warning(
                "ApifyConnector: failed to parse "
                "Facebook item: %s",
                exc,
            )
            return None

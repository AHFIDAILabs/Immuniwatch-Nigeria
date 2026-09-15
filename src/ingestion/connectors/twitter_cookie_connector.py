import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from dotenv import load_dotenv

from src.ingestion.connectors.base import BaseConnector, RawPost, hash_author
from src.ingestion.deduplication import Deduplicator

load_dotenv()

log = logging.getLogger(__name__)

_ALLOWED_LANGUAGES = {"en", "ha", "yo", "ig", "pcm"}

# Rotate one search term per poll — covers all 36 Nigerian vaccine topics
SEARCH_TERMS = [
    "vaccine Nigeria",
    "vaccination Nigeria",
    "NPHCDA vaccine",
    "polio vaccine Nigeria",
    "immunization Nigeria",
    "rigakafi Najeriya",
    "ajesara Nigeria",
    "vaccine Lagos",
    "vaccine Kano",
    "vaccine Abuja",
    "vaccine Kaduna",
    "vaccine Rivers",
    "vaccine Ogun",
    "vaccine Anambra",
    "vaccine Enugu",
    "COVID vaccine Nigeria",
    "measles vaccine Nigeria",
    "HPV vaccine Nigeria",
    "meningitis Nigeria",
    "diphtheria Nigeria",
    "cholera Nigeria vaccine",
    "malaria vaccine Nigeria",
    "yellow fever vaccine Nigeria",
    "tuberculosis Nigeria",
    "NCDC Nigeria disease",
    "health Nigeria immunization",
    "child vaccine Nigeria",
    "routine immunization Nigeria",
    "rigakafi Hausa",
    "ajesara Yoruba health",
    "NPHCDA health Nigeria",
    "WHO Nigeria vaccine",
    "UNICEF Nigeria immunization",
    "NaijaHealth vaccine",
    "Nigeria public health",
    "anti vaccine Nigeria",
]

POLL_INTERVAL = 1800   # 30 minutes — free tier, no budget cap
BATCH_SIZE    = 20     # tweets per search term per poll


class TwitterCookieConnector(BaseConnector):
    """Free cookie-based Twitter ingestion via twitter-api-client.

    Rotates through SEARCH_TERMS one per poll.  No Apify cost.
    Requires TWITTER_COOKIE_AUTH_TOKEN and TWITTER_COOKIE_CT0
    environment variables (never logged).
    """

    def __init__(self, on_post: Callable[[RawPost], None]):
        super().__init__(on_post)

        auth_token = os.environ.get("TWITTER_COOKIE_AUTH_TOKEN", "")
        ct0        = os.environ.get("TWITTER_COOKIE_CT0", "")

        if not auth_token or not ct0:
            self._enabled = False
            log.warning(
                "TwitterCookieConnector: TWITTER_COOKIE_AUTH_TOKEN or "
                "TWITTER_COOKIE_CT0 absent — connector disabled. "
                "Set both secrets to enable free Twitter ingestion."
            )
        else:
            self._enabled = True
            # Store credentials — never log their values
            self._cookies = {"auth_token": auth_token, "ct0": ct0}
            log.info(
                "TwitterCookieConnector: credentials present — "
                "connector enabled."
            )

        self._dedup      = Deduplicator()
        self._term_index = 0
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="twitter-cookie",
        )
        self._thread.start()
        log.info(
            "TwitterCookieConnector started — polling every %ds",
            POLL_INTERVAL,
        )

    def stop(self) -> None:
        self._running = False
        log.info("TwitterCookieConnector stopped.")

    # ── Poll loop ────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as exc:
                log.error("TwitterCookieConnector poll error: %s", exc)
            time.sleep(POLL_INTERVAL)

    def _poll_once(self) -> None:
        term = SEARCH_TERMS[self._term_index % len(SEARCH_TERMS)]
        self._term_index += 1

        log.debug("TwitterCookieConnector: searching term='%s'", term)

        try:
            from twitter.search import Search  # type: ignore[import]
            search = Search(
                cookies=self._cookies,
                save=False,
                debug=0,
            )
            pages = search.run(
                limit=BATCH_SIZE,
                retries=2,
                queries=[{"category": "Latest", "query": term}],
            )
        except Exception as exc:
            log.error("TwitterCookieConnector: search failed: %s", exc)
            return

        if not pages:
            log.debug(
                "TwitterCookieConnector: no results for term='%s'", term
            )
            return

        ingested = 0
        for page in pages:
            entries = self._extract_entries(page)
            for tweet_result in entries:
                post = self._to_raw_post(tweet_result)
                if post is None:
                    continue

                lang = post.language or ""
                if lang and lang not in _ALLOWED_LANGUAGES:
                    log.debug(
                        "TwitterCookieConnector: skipping lang=%s", lang
                    )
                    continue

                if not self._dedup.is_duplicate(
                    post.post_id, post.content_text
                ):
                    log.info(
                        "TwitterCookieConnector: emitting post %s lang=%s "
                        "text=%s",
                        post.post_id,
                        post.language,
                        post.content_text[:50],
                    )
                    self._safe_on_post(post)
                    ingested += 1
                else:
                    log.debug(
                        "TwitterCookieConnector: duplicate skipped %s",
                        post.post_id,
                    )

        if ingested:
            log.info(
                "TwitterCookieConnector: ingested %d new posts "
                "(term='%s')",
                ingested, term,
            )

    # ── GraphQL response parser ──────────────────────────────────

    @staticmethod
    def _extract_entries(page: dict) -> list:
        """Walk the GraphQL timeline structure to yield tweet_result dicts."""
        entries_out = []
        try:
            instructions = (
                page.get("data", {})
                    .get("search_by_raw_query", {})
                    .get("search_timeline", {})
                    .get("timeline", {})
                    .get("instructions", [])
            )
            for instruction in instructions:
                for entry in instruction.get("entries", []):
                    tweet_result = (
                        entry.get("content", {})
                             .get("itemContent", {})
                             .get("tweet_results", {})
                             .get("result", {})
                    )
                    if tweet_result:
                        entries_out.append(tweet_result)
        except Exception as exc:
            log.debug(
                "TwitterCookieConnector: entry extraction error: %s", exc
            )
        return entries_out

    def _to_raw_post(self, tweet_result: dict) -> Optional[RawPost]:
        try:
            legacy      = tweet_result.get("legacy", {})
            core        = tweet_result.get("core", {})
            user_legacy = (
                core.get("user_results", {})
                    .get("result", {})
                    .get("legacy", {})
            )

            tweet_id = legacy.get("id_str", "")
            content  = legacy.get("full_text", "").strip()

            if not tweet_id or not content or len(content) < 5:
                return None

            author_handle = user_legacy.get("screen_name", "")
            lang          = legacy.get("lang") or None
            ts_raw        = legacy.get("created_at", "")

            try:
                ts = datetime.strptime(
                    ts_raw, "%a %b %d %H:%M:%S +0000 %Y"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)

            return RawPost(
                post_id           = tweet_id,
                platform          = "twitter",
                content_text      = content,
                content_type      = "TEXT",
                author_hash       = hash_author(author_handle),
                language          = lang,
                timestamp         = ts,
                ingestion_ts      = datetime.now(timezone.utc),
                raw_url           = (
                    f"https://twitter.com/{author_handle}"
                    f"/status/{tweet_id}"
                ),
                location_raw      = user_legacy.get("location") or None,
                likes             = legacy.get("favorite_count"),
                shares            = legacy.get("retweet_count"),
                author_handle     = author_handle,
                original_post_cid = "",
            )
        except Exception as exc:
            log.warning(
                "TwitterCookieConnector: failed to parse tweet: %s", exc
            )
            return None

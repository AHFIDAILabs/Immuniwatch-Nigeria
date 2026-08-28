"""
Facebook connector using facebook-scraper library
with cookie authentication. Monitors specific known
Nigerian pages for vaccine misinformation content.
No Apify cost — completely free.
"""
import logging
import os
import threading
import time
from datetime import datetime

from src.ingestion.connectors.base import (
    BaseConnector,
    RawPost,
    hash_author,
)
from src.ingestion.deduplication import Deduplicator

log = logging.getLogger(__name__)

# Poll interval — every 6 hours
INTERVAL_SECONDS = 21600

# Nigerian Facebook pages to monitor for
# vaccine misinformation
PAGES_TO_MONITOR = [
    "christembassy",
    "davidoyedepo",
    "LoveworldInc",
    "WinnersChapelInt",
    "NaijaHealthTalk",
    "healthtalknaija",
    "VaccineChoiceNG",
]

# Posts to fetch per page per poll
POSTS_PER_PAGE = 10

_ALLOWED_LANGUAGES = {"en", "ha", "yo", "ig", "pcm"}


class FacebookCookieConnector(BaseConnector):
    """
    Monitors specific Nigerian Facebook pages using
    facebook-scraper library with cookie auth.
    Completely free — no Apify cost.
    """

    def __init__(self, on_post):
        super().__init__(on_post)
        self._c_user = os.environ.get(
            "FACEBOOK_COOKIE_C_USER", ""
        )
        self._xs = os.environ.get(
            "FACEBOOK_COOKIE_XS", ""
        )
        self._enabled = bool(self._c_user and self._xs)
        if not self._enabled:
            log.warning(
                "FacebookCookieConnector: "
                "FACEBOOK_COOKIE_C_USER or "
                "FACEBOOK_COOKIE_XS not set — "
                "connector disabled."
            )
        self._dedup = Deduplicator()
        self._running = False
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="facebook-cookie",
        )

    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._thread.start()
        log.info(
            "FacebookCookieConnector started — "
            "monitoring %d pages every %ds",
            len(PAGES_TO_MONITOR),
            INTERVAL_SECONDS,
        )

    def stop(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch_all_pages()
            except Exception as exc:
                log.error(
                    "FacebookCookieConnector poll "
                    "error: %s", exc
                )
            time.sleep(INTERVAL_SECONDS)

    def _fetch_all_pages(self) -> None:
        try:
            from facebook_scraper import get_posts
        except ImportError:
            log.error(
                "FacebookCookieConnector: "
                "facebook-scraper not installed. "
                "Run: pip install facebook-scraper"
            )
            return

        cookies = {
            "c_user": self._c_user,
            "xs": self._xs,
        }

        total_ingested = 0
        for page in PAGES_TO_MONITOR:
            try:
                ingested = self._fetch_page(
                    page, cookies, get_posts
                )
                total_ingested += ingested
            except Exception as exc:
                log.warning(
                    "FacebookCookieConnector: "
                    "error fetching page %s: %s",
                    page, exc,
                )
                continue

        if total_ingested:
            log.info(
                "FacebookCookieConnector: "
                "ingested %d new posts across "
                "%d pages",
                total_ingested,
                len(PAGES_TO_MONITOR),
            )

    def _fetch_page(
        self, page: str, cookies: dict, get_posts
    ) -> int:
        ingested = 0
        try:
            posts = get_posts(
                page,
                pages=1,
                cookies=cookies,
                options={
                    "posts_per_page": POSTS_PER_PAGE,
                },
            )
            for item in posts:
                post = self._to_raw_post(item, page)
                if post is None:
                    continue
                # Language filter
                if (
                    post.language is not None
                    and post.language
                    not in _ALLOWED_LANGUAGES
                ):
                    continue
                if not self._dedup.is_duplicate(
                    post.post_id, post.content_text
                ):
                    self._safe_on_post(post)
                    ingested += 1
        except Exception as exc:
            log.warning(
                "FacebookCookieConnector: "
                "page %s fetch error: %s",
                page, exc,
            )
        return ingested

    def _to_raw_post(
        self, item: dict, page: str
    ) -> RawPost | None:
        try:
            post_id = str(
                item.get("post_id")
                or item.get("post_url", "")
            )
            content = (
                item.get("text")
                or item.get("post_text")
                or ""
            )
            if not content:
                return None
            author = (
                item.get("username")
                or item.get("user_id")
                or page
            )
            ts_raw = item.get("time")
            try:
                if isinstance(ts_raw, datetime):
                    ts = ts_raw
                elif isinstance(ts_raw, (int, float)):
                    ts = datetime.utcfromtimestamp(
                        ts_raw
                    )
                else:
                    ts = datetime.utcnow()
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
                raw_url=item.get("post_url", ""),
                location_raw=None,
                likes=item.get("likes"),
                shares=item.get("shares"),
                author_handle=str(author),
                original_post_cid="",
            )
        except Exception as exc:
            log.warning(
                "FacebookCookieConnector: "
                "parse error: %s", exc
            )
            return None

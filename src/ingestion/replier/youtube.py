import logging

from src.ingestion.replier.base import PlatformReplier, ReplyResult

log = logging.getLogger(__name__)


class YouTubeReplier(PlatformReplier):

    def reply(
        self,
        original_post_id:  str,
        original_post_cid: str,
        author_handle:     str,
        text:              str,
    ) -> ReplyResult:
        manual_url = f"https://www.youtube.com/watch?v={original_post_id}" if len(original_post_id) == 11 else ""
        return ReplyResult(
            success=    True,
            platform=   "youtube",
            post_id=    original_post_id,
            manual_url= manual_url,
            error=      "YouTube requires OAuth 2.0 for automated replies. Use the manual_url to post manually.",
        )

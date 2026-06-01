import logging
import os
from datetime import datetime, timezone

import requests

from src.ingestion.replier.base import PlatformReplier, ReplyResult

log = logging.getLogger(__name__)
BSKY_API_BASE = "https://bsky.social/xrpc"


class BlueskyReplier(PlatformReplier):

    def __init__(self):
        self.handle       = os.environ.get("BLUESKY_HANDLE", "")
        self.app_password = os.environ.get("BLUESKY_APP_PASSWORD", "")
        self._access_jwt: str = ""
        self._did:        str = ""

    def _authenticate(self) -> bool:
        try:
            resp = requests.post(
                f"{BSKY_API_BASE}/com.atproto.server.createSession",
                json={"identifier": self.handle, "password": self.app_password},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_jwt = data.get("accessJwt", "")
            self._did        = data.get("did", "")
            return bool(self._access_jwt and self._did)
        except Exception as e:
            log.error("Bluesky replier auth failed: %s", e)
            return False

    def _resolve_did(self, handle: str) -> str:
        try:
            resp = requests.get(
                f"{BSKY_API_BASE}/com.atproto.identity.resolveHandle",
                params={"handle": handle},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("did", "")
        except Exception:
            return ""

    def reply(
        self,
        original_post_id:  str,
        original_post_cid: str,
        author_handle:     str,
        text:              str,
    ) -> ReplyResult:
        if not self.handle or not self.app_password:
            return ReplyResult(success=False, platform="bluesky", post_id=original_post_id,
                               error="BLUESKY_HANDLE or BLUESKY_APP_PASSWORD not set")

        if not self._authenticate():
            return ReplyResult(success=False, platform="bluesky", post_id=original_post_id,
                               error="Authentication failed")

        mention   = f"@{author_handle}" if author_handle else ""
        full_text = f"{mention} {text}".strip() if mention else text

        if len(full_text) > 300:
            full_text = full_text[:297] + "..."

        facets = []
        if mention and author_handle:
            author_did = self._resolve_did(author_handle)
            if author_did:
                mention_bytes = mention.encode("utf-8")
                facets.append({
                    "index": {"byteStart": 0, "byteEnd": len(mention_bytes)},
                    "features": [{
                        "$type": "app.bsky.richtext.facet#mention",
                        "did":   author_did,
                    }],
                })

        record = {
            "$type":     "app.bsky.feed.post",
            "text":      full_text,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "reply": {
                "root":   {"uri": original_post_id, "cid": original_post_cid},
                "parent": {"uri": original_post_id, "cid": original_post_cid},
            },
        }
        if facets:
            record["facets"] = facets

        try:
            resp = requests.post(
                f"{BSKY_API_BASE}/com.atproto.repo.createRecord",
                headers={"Authorization": f"Bearer {self._access_jwt}"},
                json={"repo": self._did, "collection": "app.bsky.feed.post", "record": record},
                timeout=15,
            )
            resp.raise_for_status()
            reply_uri = resp.json().get("uri", "")
            log.info("Bluesky reply posted: %s -> %s", original_post_id, reply_uri)
            return ReplyResult(success=True, platform="bluesky", post_id=original_post_id, reply_uri=reply_uri)
        except Exception as e:
            log.error("Bluesky reply failed: %s", e)
            return ReplyResult(success=False, platform="bluesky", post_id=original_post_id, error=str(e))

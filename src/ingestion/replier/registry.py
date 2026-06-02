from typing import Optional

from src.ingestion.replier.base import PlatformReplier


def get_replier(platform: str) -> Optional[PlatformReplier]:
    if platform == "bluesky":
        from src.ingestion.replier.bluesky import BlueskyReplier
        return BlueskyReplier()
    if platform == "youtube":
        from src.ingestion.replier.youtube import YouTubeReplier
        return YouTubeReplier()
    return None

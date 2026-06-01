import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class ReplyResult:
    success:    bool
    platform:   str
    post_id:    str
    reply_uri:  str = ""
    manual_url: str = ""
    error:      str = ""

class PlatformReplier(ABC):
    @abstractmethod
    def reply(
        self,
        original_post_id:  str,
        original_post_cid: str,
        author_handle:     str,
        text:              str,
    ) -> ReplyResult:
        pass

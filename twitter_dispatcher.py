"""
twitter_dispatcher.py — Twitter reply dispatch for ImmuniWatch.
"""
import logging
import os

logger = logging.getLogger(__name__)


def post_twitter_reply(reply_text: str, tweet_id: str) -> dict:
    """Post a reply to an existing tweet."""
    auth_token = os.environ.get("TWITTER_COOKIE_AUTH_TOKEN")
    ct0 = os.environ.get("TWITTER_COOKIE_CT0")

    if not auth_token or not ct0:
        raise EnvironmentError(
            "Twitter credentials missing. Set "
            "TWITTER_COOKIE_AUTH_TOKEN and "
            "TWITTER_COOKIE_CT0 in HuggingFace Space secrets."
        )

    if not tweet_id or not str(tweet_id).strip().isdigit():
        raise ValueError(
            f"tweet_id must be a numeric string, got: {tweet_id!r}"
        )

    from twitter.account import Account

    account = Account(
        cookies={"ct0": ct0, "auth_token": auth_token}
    )
    account.reply(reply_text, tweet_id=int(tweet_id))
    logger.info(
        "[twitter_dispatcher] reply posted to tweet_id=%s",
        tweet_id,
    )
    return {"success": True, "tweet_id": tweet_id}

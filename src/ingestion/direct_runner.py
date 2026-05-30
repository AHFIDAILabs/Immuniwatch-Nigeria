import logging
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

MAX_RETRIES   = 3
RETRY_DELAY_S = 2


def _classify_direct(post) -> None:
    ml_url  = os.environ.get("ML_SERVICE_URL", "http://localhost:7860")
    api_key = os.environ.get("API_KEY", "")

    payload = {
        "post_id":     post.post_id,
        "content":     post.content,
        "language":    post.language,
        "location":    post.location_raw,
        "platform":    post.platform,
        "kb_snippets": [],
    }
    headers = {
        "Content-Type": "application/json",
        "X-ML-API-Key": api_key,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{ml_url}/classify",
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            log.info(
                "Classified: post_id=%s platform=%s label=%s confidence=%.2f",
                post.post_id, post.platform,
                result.get("label"), result.get("confidence", 0),
            )
            return
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    log.error("All attempts failed for post_id=%s", post.post_id)


def run() -> None:
    log.info("=" * 55)
    log.info("ImmuniWatch — Direct Runner (background thread)")
    log.info("ML Service: %s", os.environ.get("ML_SERVICE_URL", "http://localhost:7860"))
    log.info("=" * 55)

    from src.ingestion.connectors.bluesky import BlueskyConnector
    from src.ingestion.connectors.youtube import YouTubeConnector
    from src.ingestion.connectors.sociavault import SociaVaultConnector

    connectors = [
        BlueskyConnector(_classify_direct),
        YouTubeConnector(_classify_direct),
        SociaVaultConnector(_classify_direct),
    ]

    started = []
    for connector in connectors:
        try:
            connector.start()
            if connector.is_running:
                started.append(connector.__class__.__name__)
        except Exception as e:
            log.warning("Connector %s failed to start: %s",
                        connector.__class__.__name__, e)

    if not started:
        log.warning("No connectors started — ingestion worker exiting.")
        return

    log.info("Running connectors: %s", ", ".join(started))

    # Keep thread alive — connectors run in their own threads
    while True:
        time.sleep(30)


if __name__ == "__main__":
    import sys
    import signal

    def _shutdown(signum, frame):
        log.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    run()

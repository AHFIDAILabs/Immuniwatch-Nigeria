import logging
import os
import signal
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000")
API_KEY        = os.environ.get("API_KEY", "")

MAX_RETRIES   = 3
RETRY_DELAY_S = 2


def _classify_direct(post) -> None:
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
        "X-ML-API-Key": API_KEY,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{ML_SERVICE_URL}/classify",
                json=payload,
                headers=headers,
                timeout=10,
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
    log.info("ImmuniWatch — Direct Runner (no Kafka)")
    log.info("ML Service: %s", ML_SERVICE_URL)
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
        connector.start()
        if connector.is_running:
            started.append(connector.__class__.__name__)

    if not started:
        log.error("No connectors started. Check your API keys in .env.")
        sys.exit(1)

    log.info("Running: %s", ", ".join(started))
    log.info("Posting directly to: %s/classify", ML_SERVICE_URL)

    def _shutdown(signum, frame):
        log.info("Shutting down...")
        for connector in connectors:
            connector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()

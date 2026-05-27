import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:9092")
TOPIC_RAW     = "iw.raw-posts"


# ---------------------------------------------------------------------------
# Kafka publisher — shared by all connectors
# ---------------------------------------------------------------------------
def _make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


def _publish_to_kafka(producer: KafkaProducer, post) -> None:
    """Publish a RawPost to iw.raw-posts topic."""
    try:
        message = post.to_kafka_message()
        producer.send(TOPIC_RAW, value=message)
        log.debug(
            "Published: post_id=%s platform=%s",
            post.post_id,
            post.platform,
        )
    except KafkaError as e:
        log.error("Kafka publish failed for post_id=%s: %s", post.post_id, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    """Start all connectors. Runs until Ctrl+C."""
    log.info("=" * 55)
    log.info("ImmuniWatch — Ingestion Runner")
    log.info("Kafka broker: %s", KAFKA_BROKERS)
    log.info("=" * 55)

    # Kafka producer shared across all connectors
    try:
        producer = _make_producer()
        log.info("Kafka producer connected.")
    except Exception as e:
        log.error("Failed to connect to Kafka: %s", e)
        log.error("Make sure Kafka is running: docker-compose up -d")
        sys.exit(1)

    def on_post(post):
        _publish_to_kafka(producer, post)

    # Import connectors
    from src.ingestion.connectors.youtube import YouTubeConnector
    from src.ingestion.connectors.sociavault import SociaVaultConnector
    from src.ingestion.connectors.bluesky import BlueskyConnector

    connectors = [
        YouTubeConnector(on_post),
        SociaVaultConnector(on_post),
        BlueskyConnector(on_post),
    ]

    # Start all connectors
    started = []
    for connector in connectors:
        connector.start()
        if connector.is_running:
            started.append(connector.__class__.__name__)

    if not started:
        log.error(
            "No connectors started. "
            "Check your API keys in .env and try again."
        )
        sys.exit(1)

    log.info("Running connectors: %s", ", ".join(started))
    log.info("Publishing to: %s", TOPIC_RAW)
    log.info("Press Ctrl+C to stop.")

    # Graceful shutdown on Ctrl+C
    def _shutdown(signum, frame):
        log.info("Shutting down connectors...")
        for connector in connectors:
            connector.stop()
        producer.close()
        log.info("All connectors stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep main thread alive
    while True:
        time.sleep(5)


if __name__ == "__main__":
    run()
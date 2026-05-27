
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
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
KAFKA_BROKERS   = os.environ.get("KAFKA_BROKERS", "localhost:9092")
KAFKA_GROUP_ID  = os.environ.get("KAFKA_GROUP_ID", "iw-ml-service")
ML_SERVICE_URL  = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")
API_KEY         = os.environ.get("API_KEY", "")

TOPIC_RAW        = "iw.raw-posts"
TOPIC_CLASSIFIED = "iw.classified-posts"
TOPIC_HITL       = "iw.hitl-queue"

# Route to HITL queue if model uncertainty is high
UNCERTAINTY_THRESHOLD = 0.45

# Retry settings for FastAPI calls
MAX_RETRIES   = 3
RETRY_DELAY_S = 2


# ---------------------------------------------------------------------------
# Kafka producer — publishes classification results
# ---------------------------------------------------------------------------
def _make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


# ---------------------------------------------------------------------------
# Kafka consumer — reads from iw.raw-posts
# ---------------------------------------------------------------------------
def _make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=KAFKA_BROKERS.split(","),
        group_id=KAFKA_GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


# ---------------------------------------------------------------------------
# Call FastAPI /classify endpoint
# ---------------------------------------------------------------------------
def _classify(post: dict) -> dict:
    """
    Send post to FastAPI /classify and return the result.
    Retries up to MAX_RETRIES times on failure.
    Returns None if all retries fail.
    """
    payload = {
        "post_id":     post.get("post_id", ""),
        "content":     post.get("content", ""),
        "language":    post.get("language"),
        "platform":    post.get("platform", "submission"),
        "kb_snippets": [],
    }

    headers = {
        "Content-Type":  "application/json",
        "X-ML-API-Key":  API_KEY,
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
            return resp.json()
        except requests.RequestException as e:
            log.warning("Classify attempt %d/%d failed: %s",
                        attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    log.error("All classify attempts failed for post_id=%s",
              post.get("post_id"))
    return None


# ---------------------------------------------------------------------------
# Publish to Kafka topic
# ---------------------------------------------------------------------------
def _publish(producer: KafkaProducer, topic: str, message: dict) -> None:
    """Publish a message to a Kafka topic with schema_version."""
    message["schema_version"] = "1.0"
    message["published_at"]   = datetime.now(timezone.utc).isoformat()

    try:
        producer.send(topic, value=message)
        producer.flush()
        log.debug("Published to %s: post_id=%s",
                  topic, message.get("post_id"))
    except KafkaError as e:
        log.error("Failed to publish to %s: %s", topic, e)


# ---------------------------------------------------------------------------
# Route to HITL queue
# ---------------------------------------------------------------------------
def _should_route_to_hitl(result: dict) -> bool:
    if result.get("label") == "misinformation":
        return True
    if result.get("entropy", 0) > UNCERTAINTY_THRESHOLD:
        return True
    return False


# ---------------------------------------------------------------------------
# Process one post
# ---------------------------------------------------------------------------
def _process(post: dict, producer: KafkaProducer) -> None:
    """Classify one post and publish results."""
    post_id  = post.get("post_id", "unknown")
    content  = post.get("content", "")

    if not content or len(content.strip()) < 5:
        log.debug("Skipping empty post: %s", post_id)
        return

    # Classify
    result = _classify(post)
    if result is None:
        return

    # Build classified message
    classified_msg = {
        "post_id":       post_id,
        "label":         result["label"],
        "confidence":    result["confidence"],
        "entropy":       result["entropy"],
        "language":      result.get("language"),
        "state":         result.get("state"),
        "platform":      result.get("platform"),
        "model_version": result.get("model_version"),
        "alternatives":  result.get("alternatives", []),
        "processing_ms": result.get("processing_ms"),
        "original_text": content,
        "ingested_at":   post.get("ingestion_ts"),
    }

    # Always publish to classified-posts
    _publish(producer, TOPIC_CLASSIFIED, classified_msg)

    # Route to HITL queue if needed
    if _should_route_to_hitl(result):
        _publish(producer, TOPIC_HITL, classified_msg)
        log.info("Routed to HITL: post_id=%s label=%s confidence=%.2f",
                 post_id, result["label"], result["confidence"])
    else:
        log.info("Classified: post_id=%s label=%s confidence=%.2f",
                 post_id, result["label"], result["confidence"])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run() -> None:
    """Start the classification worker. Runs until interrupted."""
    log.info("=" * 55)
    log.info("ImmuniWatch — Classification Worker")
    log.info("Broker:    %s", KAFKA_BROKERS)
    log.info("Group:     %s", KAFKA_GROUP_ID)
    log.info("ML URL:    %s", ML_SERVICE_URL)
    log.info("Consuming: %s", TOPIC_RAW)
    log.info("=" * 55)

    producer = _make_producer()
    consumer = _make_consumer()

    log.info("Waiting for posts on %s ...", TOPIC_RAW)

    try:
        for message in consumer:
            post = message.value
            _process(post, producer)
    except KeyboardInterrupt:
        log.info("Worker stopped by user.")
    finally:
        consumer.close()
        producer.close()
        log.info("Connections closed.")


if __name__ == "__main__":
    run()
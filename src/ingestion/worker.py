
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

# Module-level RAG singleton — initialised once at worker startup
_rag = None


# ---------------------------------------------------------------------------
# RAG initialisation — Section 5.3
# ---------------------------------------------------------------------------
def _init_rag() -> None:
    """Initialise RAG retriever once at startup. Graceful if KB not ready."""
    global _rag
    try:
        from src.intelligence.rag import RAGRetriever
        _rag = RAGRetriever()
        if _rag.is_ready():
            log.info("RAG retriever ready.")
        else:
            log.warning(
                "RAG retriever initialised but knowledge base is empty — "
                "run: python -m src.intelligence.ingestion"
            )
    except Exception as e:
        log.warning("RAG retriever failed to initialise: %s — evidence will be empty", e)
        _rag = None


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
        "location":    post.get("location_raw"),
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
# RAG cross-reference — Section 5.3
# Returns evidence records as list of dicts; always returns a list.
# ---------------------------------------------------------------------------
def _retrieve_evidence(content: str, language: str) -> list:
    if _rag is None or not _rag.is_ready():
        return []
    try:
        return _rag.retrieve_as_dicts(content, language)
    except Exception as e:
        log.error("RAG retrieval error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Counter-response generation — Section 6.5
# Only called when label == "misinformation" and evidence is present.
# Returns serialisable dict or None.
# ---------------------------------------------------------------------------
def _build_counter_response(post_id: str, content: str,
                             language: str, evidence: list):
    if not evidence:
        return None
    try:
        from src.intelligence.counter import generate_counter_response
        snippets = [e["snippet"] for e in evidence if e.get("snippet")]
        urls     = [e["source_url"] for e in evidence if e.get("source_url")]
        cr = generate_counter_response(
            post_id=           post_id,
            claim=             content,
            language=          language,
            evidence_snippets= snippets,
            source_urls=       urls,
        )
        return cr.to_dict() if cr else None
    except Exception as e:
        log.error("Counter-response generation error for %s: %s", post_id, e)
        return None


# ---------------------------------------------------------------------------
# Process one post — Section 2.2 pipeline
# ---------------------------------------------------------------------------
def _process(post: dict, producer: KafkaProducer) -> None:
    """
    Full pipeline for one post:
      1. Classify (FastAPI)
      2. RAG cross-reference → evidence records   (HITL posts only)
      3. Counter-response generation               (misinformation + evidence only)
      4. Publish to iw.classified-posts
      5. Route to iw.hitl-queue if needed
    """
    post_id = post.get("post_id", "unknown")
    content = post.get("content", "")

    if not content or len(content.strip()) < 5:
        log.debug("Skipping empty post: %s", post_id)
        return

    # Step 1 — Classify
    result = _classify(post)
    if result is None:
        return

    language = result.get("language") or "en"

    # Build classified message — base fields
    classified_msg = {
        "post_id":           post_id,
        "label":             result["label"],
        "confidence":        result["confidence"],
        "entropy":           result["entropy"],
        "language":          result.get("language"),
        "state":             result.get("state"),
        "platform":          result.get("platform"),
        "model_version":     result.get("model_version"),
        "alternatives":      result.get("alternatives", []),
        "processing_ms":     result.get("processing_ms"),
        "original_text":     content,
        "ingested_at":       post.get("ingestion_ts"),
        "evidence_records":  [],
        "counter_responses": None,
        "hitl_state":        "CLASSIFIED",
    }

    # Steps 2 + 3 — RAG + counter-response for HITL-bound posts
    if _should_route_to_hitl(result):
        # Step 2 — RAG cross-reference (Section 5.3)
        evidence = _retrieve_evidence(content, language)
        classified_msg["evidence_records"] = evidence
        if evidence:
            classified_msg["hitl_state"] = "EVIDENCE_READY"

        # Step 3 — Counter-response (Section 6.5): misinformation + evidence required
        if result.get("label") == "misinformation" and evidence:
            cr = _build_counter_response(post_id, content, language, evidence)
            classified_msg["counter_responses"] = cr

    # Step 4 — Always publish to classified-posts
    _publish(producer, TOPIC_CLASSIFIED, classified_msg)

    # Step 5 — Route to HITL queue if needed
    if _should_route_to_hitl(result):
        _publish(producer, TOPIC_HITL, classified_msg)
        log.info(
            "HITL routed: post_id=%s label=%s confidence=%.2f "
            "evidence=%d counter=%s hitl_state=%s",
            post_id, result["label"], result["confidence"],
            len(classified_msg["evidence_records"]),
            "yes" if classified_msg["counter_responses"] else "no",
            classified_msg["hitl_state"],
        )
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

    _init_rag()

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

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

DB_PATH = Path("data/counter_narratives.db")
_lock   = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _lock, _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS cn_queue (
                post_id           TEXT PRIMARY KEY,
                platform          TEXT NOT NULL,
                author_handle     TEXT DEFAULT '',
                original_post_uri TEXT DEFAULT '',
                original_post_cid TEXT DEFAULT '',
                content_snippet   TEXT DEFAULT '',
                label             TEXT NOT NULL,
                confidence        REAL NOT NULL,
                language          TEXT DEFAULT 'en',
                generated_short   TEXT DEFAULT '',
                generated_medium  TEXT DEFAULT '',
                generated_long    TEXT DEFAULT '',
                sources           TEXT DEFAULT '[]',
                status            TEXT DEFAULT 'pending',
                reply_uri         TEXT DEFAULT '',
                manual_url        TEXT DEFAULT '',
                error_msg         TEXT DEFAULT '',
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            )
        """)
        con.commit()
    log.info("Counter-narrative store initialised at %s", DB_PATH)


def queue_post(
    post_id:           str,
    platform:          str,
    author_handle:     str,
    original_post_uri: str,
    original_post_cid: str,
    content_snippet:   str,
    label:             str,
    confidence:        float,
    language:          str,
    generated_short:   str,
    generated_medium:  str,
    generated_long:    str,
    sources:           List[str],
) -> None:
    now = _now()
    with _lock, _conn() as con:
        con.execute("""
            INSERT OR IGNORE INTO cn_queue
            (post_id, platform, author_handle, original_post_uri, original_post_cid,
             content_snippet, label, confidence, language,
             generated_short, generated_medium, generated_long, sources,
             status, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            post_id, platform, author_handle, original_post_uri, original_post_cid,
            content_snippet[:280], label, confidence, language,
            generated_short, generated_medium, generated_long,
            json.dumps(sources), "pending", now, now,
        ))
        con.commit()


def get_pending(limit: int = 50) -> List[dict]:
    with _lock, _conn() as con:
        rows = con.execute(
            "SELECT * FROM cn_queue WHERE status='pending' ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_history(limit: int = 50) -> List[dict]:
    with _lock, _conn() as con:
        rows = con.execute(
            "SELECT * FROM cn_queue WHERE status != 'pending' ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def mark_deployed(post_id: str, reply_uri: str = "", manual_url: str = "") -> None:
    _update_status(post_id, "deployed", reply_uri=reply_uri, manual_url=manual_url)


def mark_skipped(post_id: str) -> None:
    _update_status(post_id, "skipped")


def mark_failed(post_id: str, error: str) -> None:
    _update_status(post_id, "failed", error_msg=error)


def _update_status(
    post_id:    str,
    status:     str,
    reply_uri:  str = "",
    manual_url: str = "",
    error_msg:  str = "",
) -> None:
    with _lock, _conn() as con:
        con.execute("""
            UPDATE cn_queue
            SET status=?, reply_uri=?, manual_url=?, error_msg=?, updated_at=?
            WHERE post_id=?
        """, (status, reply_uri, manual_url, error_msg, _now(), post_id))
        con.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["sources"] = json.loads(d.get("sources", "[]"))
    except Exception:
        d["sources"] = []
    return d

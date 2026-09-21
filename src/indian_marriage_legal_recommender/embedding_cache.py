"""Tiny SQLite-backed cache of query embeddings.

Lets a query that was embedded **once** be reused later without loading the
embedding model again — the cached vector is served straight from disk. This
avoids re-running (or even re-loading) the model when the embedding model is
unavailable.

The cache is keyed by ``(model_name, text)`` and stores the vector as JSON. It is
consulted transparently by :class:`CachedEmbeddings` in ``embeddings.py``; callers
never interact with this module directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from functools import lru_cache
from typing import List, Optional

from config import config

_LOCK = threading.Lock()


def _key(model_name: str, text: str) -> str:
    return hashlib.sha256(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _conn() -> sqlite3.Connection:
    """Return a cached connection, creating the DB file and table on first use."""
    config.EMBED_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.EMBED_CACHE_DB), check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "  key TEXT PRIMARY KEY,"
        "  model TEXT NOT NULL,"
        "  vector TEXT NOT NULL"
        ")"
    )
    conn.commit()
    return conn


def get(model_name: str, text: str) -> Optional[List[float]]:
    """Return the cached vector for ``(model_name, text)`` or ``None`` on a miss."""
    if not config.EMBED_CACHE_ENABLED:
        return None
    with _LOCK:
        row = _conn().execute(
            "SELECT vector FROM embeddings WHERE key = ?", (_key(model_name, text),)
        ).fetchone()
    return json.loads(row[0]) if row else None


def put(model_name: str, text: str, vector: List[float]) -> None:
    """Store ``vector`` for ``(model_name, text)`` (no-op if caching is disabled)."""
    if not config.EMBED_CACHE_ENABLED:
        return
    with _LOCK:
        _conn().execute(
            "INSERT OR REPLACE INTO embeddings (key, model, vector) VALUES (?, ?, ?)",
            (_key(model_name, text), model_name, json.dumps(vector)),
        )
        _conn().commit()

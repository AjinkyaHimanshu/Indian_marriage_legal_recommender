"""Disk-backed cache for pre-computed RAG results — enables offline demo runs.

Cache file lives at  demo/demo_cache.json  (committed to git).

Workflow
--------
Work laptop  (API access) :  python demo/build_demo_cache.py
Personal laptop (no API)  :  set DEMO_MODE=1  &&  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

# src/indian_marriage_legal_recommender/demo_cache.py → parents[2] == repo root
CACHE_FILE = Path(__file__).resolve().parents[2] / "demo" / "demo_cache.json"


def cache_key(question: str, profile_key: str, top_k: int) -> str:
    raw = f"{question.strip().lower()}|{profile_key}|{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get(question: str, profile_key: str, top_k: int) -> Optional[dict]:
    """Return the cached raw dict for this query, or None on miss."""
    return _load().get(cache_key(question, profile_key, top_k))


def put(question: str, profile_key: str, top_k: int, result) -> None:
    """Serialize a RAGResult and write it to the cache file."""
    data = _load()
    data[cache_key(question, profile_key, top_k)] = {
        "question": result.question,
        "expanded_query": result.expanded_query,
        "answer": result.answer,
        "documents": [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in result.documents
        ],
        "timings": result.timings,
    }
    _save(data)


def list_cached() -> list[str]:
    """Return all cached question strings (for inspection)."""
    return [v["question"] for v in _load().values()]

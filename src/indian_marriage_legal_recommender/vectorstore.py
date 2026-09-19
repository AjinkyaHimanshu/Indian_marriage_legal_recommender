"""Qdrant plumbing shared by ingestion and retrieval.

Collections store a **flat payload** (``text``, ``law_name``, ``section_number``,
``section_title`` and, for chunked collections, ``chunk_part`` / ``total_parts``).
This matches the on-disk database shipped with the repo so retrieval works without
re-embedding. ``as_langchain_store`` also exposes a collection as a
``langchain_qdrant.QdrantVectorStore`` for idiomatic LangChain usage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import config

CONTENT_KEY = "text"


@lru_cache(maxsize=1)
def get_client(path: Optional[str] = None) -> QdrantClient:
    """Return a cached local Qdrant client bound to ``database/qdrant_db``."""
    return QdrantClient(path=str(path or config.DB_DIR))


def ensure_collection(client: QdrantClient, name: str, dim: int, *, recreate: bool = False) -> None:
    """Create ``name`` with cosine distance if missing (or recreate it)."""
    exists = client.collection_exists(name)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def as_langchain_store(collection: str, embed_model: str, path: Optional[str] = None):
    """Wrap a collection as a LangChain ``QdrantVectorStore``.

    The content payload key is ``text`` (flat schema); remaining payload fields are
    surfaced as document metadata.
    """
    from langchain_qdrant import QdrantVectorStore

    from .embeddings import get_embeddings

    return QdrantVectorStore(
        client=get_client(path),
        collection_name=collection,
        embedding=get_embeddings(embed_model),
        content_payload_key=CONTENT_KEY,
    )

"""Embedding models exposed as LangChain ``Embeddings`` objects.

Two encoders are supported, matching the original benchmark:

* ``all-MiniLM-L6-v2`` (384-dim) — lightweight baseline, wrapped with the
  standard ``langchain_huggingface.HuggingFaceEmbeddings``.
* ``jina-embeddings-v3`` (1024-dim) — task-specific long-context encoder. Jina v3
  needs a *different* task prefix for passages vs. queries
  (``retrieval.passage`` when indexing, ``retrieval.query`` when searching), which
  the standard wrapper cannot express, so it gets a small custom class.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from config import config


class JinaV3Embeddings(Embeddings):
    """LangChain ``Embeddings`` for ``jinaai/jina-embeddings-v3``.

    Applies the correct task prefix per call: passages when embedding documents,
    queries when embedding a search string.
    """

    def __init__(self, max_seq_length: int = config.JINA_MAX_TOKENS):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.JINA_EMBED_MODEL, trust_remote_code=True)
        self._model.max_seq_length = max_seq_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(list(texts), task="retrieval.passage")
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, task="retrieval.query").tolist()


@lru_cache(maxsize=2)
def _baseline() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.BASELINE_EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": False},
    )


@lru_cache(maxsize=2)
def _jina() -> Embeddings:
    return JinaV3Embeddings()


def get_embeddings(embed_model: str) -> Embeddings:
    """Return a cached LangChain ``Embeddings`` for the given model id."""
    if embed_model == config.JINA_EMBED_MODEL:
        return _jina()
    if embed_model == config.BASELINE_EMBED_MODEL:
        return _baseline()
    raise ValueError(f"Unsupported embedding model: {embed_model}")


def for_profile(profile: "config.ConfigProfile") -> Embeddings:
    return get_embeddings(profile.embed_model)

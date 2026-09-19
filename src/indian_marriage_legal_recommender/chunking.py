"""Sliding-window, token-level chunker.

Long legal sections silently overflow an embedding model's context window and get
truncated. We split them into overlapping token windows before re-embedding, then
reconstruct the full section at retrieval time (parent-child merge in
``retrieval.py``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from config import config


@lru_cache(maxsize=4)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def chunk_by_tokens(
    text: str,
    model_name: str,
    *,
    max_tokens: int,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Split ``text`` into overlapping windows of at most ``max_tokens`` tokens.

    A section that already fits returns a single-element list. Otherwise windows
    advance by ``max_tokens - overlap`` tokens, preserving ``overlap`` tokens of
    context between neighbours.
    """
    tokenizer = _tokenizer(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]

    step = max_tokens - overlap
    chunks: List[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        chunks.append(tokenizer.decode(window))
        if start + max_tokens >= len(tokens):
            break
    return chunks


def window_for(profile: "config.ConfigProfile") -> int:
    """The chunk window size to use for a given (chunked) profile."""
    if profile.embed_model == config.JINA_EMBED_MODEL:
        return config.JINA_CHUNK_WINDOW
    return config.BASELINE_CHUNK_WINDOW

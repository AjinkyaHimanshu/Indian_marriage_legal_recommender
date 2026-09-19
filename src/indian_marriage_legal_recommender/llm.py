"""Claude LLM factory built on ``langchain_anthropic.ChatAnthropic``.

Every LLM call in the system (query expansion, answer generation, LLM-as-a-judge
evaluation) flows through here, so the provider is Claude and only Claude.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from config import config


def get_chat_model(
    *,
    model: str = config.CLAUDE_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> ChatAnthropic:
    """Create a configured Claude chat model.

    Parameters mirror the settings used across the original notebooks:
    ``temperature=0.2`` for generation/expansion, ``0.0`` for deterministic
    evaluation.
    """
    api_key = config.require_anthropic_key()
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_api_key=api_key,
        timeout=120,
        max_retries=2,
    )


@lru_cache(maxsize=8)
def _cached(model: str, temperature: float, max_tokens: int) -> ChatAnthropic:
    return get_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)


def shared_chat_model(
    *, temperature: float = 0.2, max_tokens: int = 1024, model: str = config.CLAUDE_MODEL
) -> ChatAnthropic:
    """Process-cached variant so the Streamlit app reuses one client."""
    return _cached(model, temperature, max_tokens)

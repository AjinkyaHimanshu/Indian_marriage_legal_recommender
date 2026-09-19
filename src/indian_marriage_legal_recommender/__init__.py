"""Indian Marriage Law — RAG-based Legal Assistant.

A Retrieval-Augmented Generation system over 40+ Indian marriage statutes,
built on **LangChain** with **Anthropic Claude** as the only LLM provider.

The public pipeline entry point is :func:`indian_marriage_legal_recommender.pipeline.answer`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "llm",
    "embeddings",
    "vectorstore",
    "chunking",
    "ingest",
    "query_expansion",
    "retrieval",
    "generation",
    "evaluation",
    "pipeline",
]

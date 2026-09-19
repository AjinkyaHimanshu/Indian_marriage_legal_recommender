"""End-to-end RAG pipeline: expand -> retrieve -> generate.

This is the single entry point the Streamlit app (and any caller) uses. It ties the
LangChain components together and reports per-stage timings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from config import config
from .generation import generate_answer
from .query_expansion import expand_query
from .retrieval import get_retriever


@dataclass
class RAGResult:
    question: str
    expanded_query: str
    answer: str
    documents: List[Document] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    @property
    def sections(self) -> List[str]:
        return [d.metadata.get("section", "") for d in self.documents]


def answer(
    question: str,
    *,
    profile_key: str = config.DEFAULT_CONFIG,
    top_k: int = config.TOP_K_FINAL,
) -> RAGResult:
    """Run the full pipeline for one question and return a structured result."""
    prof = config.profile(profile_key)
    timings: dict = {}

    # 1. Query expansion (rechunked configs retrieve with the expanded query)
    t0 = time.perf_counter()
    expanded = expand_query(question) if prof.use_expanded_query else question
    timings["expand"] = time.perf_counter() - t0

    # 2. Retrieval
    t1 = time.perf_counter()
    retriever = get_retriever(profile_key, top_k_final=top_k)
    docs = retriever.invoke(expanded)
    timings["retrieve"] = time.perf_counter() - t1

    # 3. Grounded generation
    t2 = time.perf_counter()
    if docs:
        text = generate_answer(question, docs)
    else:
        text = (
            "No relevant legal sections were retrieved. The topic may not be "
            "covered in the current corpus."
        )
    timings["generate"] = time.perf_counter() - t2

    return RAGResult(
        question=question,
        expanded_query=expanded,
        answer=text,
        documents=docs,
        timings=timings,
    )

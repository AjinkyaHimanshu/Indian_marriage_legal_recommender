"""End-to-end RAG pipeline: expand -> retrieve -> generate.

This is the single entry point the Streamlit app (and any caller) uses. It ties the
LangChain components together and reports per-stage timings.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from config import config
from .demo_cache import get as _cache_get, put as _cache_put
from .generation import generate_answer
from .query_expansion import expand_query
from .retrieval import get_retriever

DEMO_MODE = os.environ.get("DEMO_MODE", "0").strip() in ("1", "true", "True")


@dataclass
class RAGResult:
    question: str
    expanded_query: str
    answer: str
    documents: List[Document] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    @property
    def sections(self) -> List[str]:
        """Return the section header string for each retrieved document."""
        return [d.metadata.get("section", "") for d in self.documents]


def answer(
    question: str,
    *,
    profile_key: str = config.DEFAULT_CONFIG,
    top_k: int = config.TOP_K_FINAL,
) -> RAGResult:
    """Run the full pipeline for one question and return a structured result."""
    # ── Cache check (always — saves API calls; required in DEMO_MODE) ─────────
    cached = _cache_get(question, profile_key, top_k)
    if cached:
        return RAGResult(
            question=cached["question"],
            expanded_query=cached["expanded_query"],
            answer=cached["answer"],
            documents=[
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in cached["documents"]
            ],
            timings=cached["timings"],
        )

    if DEMO_MODE:
        return RAGResult(
            question=question,
            expanded_query=question,
            answer=(
                "⚠️ This question is not in the demo cache.\n\n"
                "Run `python demo/build_demo_cache.py` on a machine with API access "
                "to add it, then commit `demo/demo_cache.json`."
            ),
            documents=[],
            timings={"expand": 0.0, "retrieve": 0.0, "generate": 0.0},
        )
    # ─────────────────────────────────────────────────────────────────────────

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

    result = RAGResult(
        question=question,
        expanded_query=expanded,
        answer=text,
        documents=docs,
        timings=timings,
    )
    _cache_put(question, profile_key, top_k, result)
    return result

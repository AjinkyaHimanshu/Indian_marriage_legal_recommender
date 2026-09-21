"""Hybrid legal retrieval as a LangChain ``BaseRetriever``.

Rechunked configs run the full pipeline::

    vector search  ->  BM25 keyword scoring  ->  cross-encoder re-ranking
                   ->  weighted fusion (0.3*BM25 + 0.7*CE)
                   ->  parent-child merge (rebuild full sections from chunks)

Unchunked configs use plain semantic top-k with no re-ranking or merge.

The retriever reads the flat Qdrant payload directly, so it works against the
database shipped in the repo without re-embedding.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from config import config
from .embeddings import get_embeddings
from .vectorstore import get_client


@lru_cache(maxsize=1)
def _stopwords() -> set:
    import nltk

    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(nltk.corpus.stopwords.words("english"))


@lru_cache(maxsize=1)
def _cross_encoder():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=config.CROSS_ENCODER_MODEL)


def _tokenize(text: str) -> List[str]:
    stop = _stopwords()
    return [t for t in text.lower().split() if t not in stop and len(t) > 2]


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridLegalRetriever(BaseRetriever):
    """Configurable retriever over one Qdrant collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile_key: str = config.DEFAULT_CONFIG
    top_k_initial: int = config.TOP_K_INITIAL
    top_k_final: int = config.TOP_K_FINAL

    _profile: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._profile = config.profile(self.profile_key)

    # -- vector search -----------------------------------------------------
    def _vector_search(self, query: str) -> List[Dict]:
        prof = self._profile
        query_vec = get_embeddings(prof.embed_model).embed_query(query)
        limit = self.top_k_initial if prof.hybrid else self.top_k_final
        res = get_client().query_points(
            collection_name=prof.collection, query=query_vec, limit=limit
        )
        return [hit.payload for hit in res.points]

    # -- unchunked: plain semantic top-k ----------------------------------
    @staticmethod
    def _format_section(payload: Dict) -> Document:
        header = (
            f"{payload.get('law_name', '')} | "
            f"Section {payload.get('section_number', '')}: {payload.get('section_title', '')}"
        )
        text = f"{header}\n{payload.get('text', '')}"
        return Document(
            page_content=text,
            metadata={
                "section": header,
                "law_name": payload.get("law_name", ""),
                "section_number": payload.get("section_number", ""),
                "section_title": payload.get("section_title", ""),
            },
        )

    # -- rechunked: hybrid re-rank + parent-child merge -------------------
    def _hybrid(self, query: str, payloads: List[Dict]) -> List[Document]:
        if not payloads:
            return []
        chunks = [p["text"] for p in payloads]

        # BM25 keyword relevance over the candidate pool
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([_tokenize(c) for c in chunks])
        bm25_scores = list(bm25.get_scores(_tokenize(query)))

        # Cross-encoder pairwise relevance (LangChain community wrapper)
        ce_scores = _cross_encoder().score([(query, c) for c in chunks])

        fused = [
            config.BM25_WEIGHT * b + config.CE_WEIGHT * c
            for b, c in zip(_normalize(bm25_scores), _normalize(list(ce_scores)))
        ]
        ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[: self.top_k_final]

        # Parent-child merge: rebuild full sections from the winning chunks
        merged: Dict[str, list] = defaultdict(list)
        for idx in ranked:
            p = payloads[idx]
            key = (
                f"{p['law_name']} | Section {p['section_number']}: {p['section_title']}"
            )
            part = p.get("chunk_part", 1)
            if part not in [x[0] for x in merged[key]]:
                merged[key].append((part, p["text"]))

        docs: List[Document] = []
        for key, frags in merged.items():
            frags.sort(key=lambda x: x[0])
            body = "".join(f"[Part {n}]: {t}\n" for n, t in frags)
            docs.append(
                Document(
                    page_content=f"{key}\n{body}",
                    metadata={"section": key},
                )
            )
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        payloads = self._vector_search(query)
        if self._profile.hybrid:
            return self._hybrid(query, payloads)
        return [self._format_section(p) for p in payloads]


def get_retriever(profile_key: str = config.DEFAULT_CONFIG, top_k_final: int | None = None) -> HybridLegalRetriever:
    kwargs: Dict[str, Any] = {"profile_key": profile_key}
    if top_k_final is not None:
        kwargs["top_k_final"] = top_k_final
        kwargs["top_k_initial"] = max(top_k_final * 2, config.TOP_K_INITIAL)
    return HybridLegalRetriever(**kwargs)

"""Corpus loading and Qdrant collection building (notebooks 2 & 3).

Reads the per-law JSON files in ``data/base_chunked_marriage_laws`` and populates
the four Qdrant collections used by the benchmark:

* unchunked: ``legal_acts_baseline`` (MiniLM), ``legal_acts_jina`` (Jina)
* rechunked: ``legal_baseline_chunked``, ``legal_jina_chunked``

Embeddings go through the LangChain ``Embeddings`` objects in :mod:`embeddings`, so
the same encoders are used everywhere.
"""

from __future__ import annotations

import json
import uuid
from typing import Dict, Iterable, List

from config import config
from qdrant_client.models import PointStruct
from tqdm import tqdm

from . import chunking
from .embeddings import get_embeddings
from .vectorstore import ensure_collection, get_client


# ── Corpus loading ───────────────────────────────────────────────────────────
def load_sections() -> List[Dict]:
    """Flatten every law JSON into a list of section records.

    Each record: ``{law_name, section_number, section_title, text}``.
    """
    sections: List[Dict] = []
    for path in sorted(config.LAWS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        for chunk in doc.get("chunks", []):
            content = (chunk.get("content") or "").strip()
            if not content or content == "***":
                continue
            sections.append(
                {
                    "law_name": chunk.get("act_number")
                    or doc.get("document_metadata", {}).get("document_title", path.stem),
                    "section_number": str(chunk.get("section_number", "")).strip(),
                    "section_title": (chunk.get("section_title") or "").strip(),
                    "text": content,
                }
            )
    return sections


def save_sections_checkpoint(sections: List[Dict]) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CHUNKED_LAWS_JSON
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(sections, fh, ensure_ascii=False, indent=2)


# ── Collection builders ───────────────────────────────────────────────────────
def _upsert(collection: str, points: List[PointStruct], batch: int = 100) -> None:
    client = get_client()
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i : i + batch])


def build_unchunked(profile_key: str, sections: List[Dict] | None = None, *, recreate: bool = True) -> int:
    """Build an unchunked collection (one vector per full section)."""
    prof = config.profile(profile_key)
    assert not prof.chunked, f"{profile_key} is a chunked profile"
    sections = sections if sections is not None else load_sections()
    embeddings = get_embeddings(prof.embed_model)

    ensure_collection(get_client(), prof.collection, prof.dim, recreate=recreate)

    texts = [s["text"] for s in sections]
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(
            id=idx,
            vector=vec,
            payload={
                "text": s["text"],
                "law_name": s["law_name"],
                "section_number": s["section_number"],
                "section_title": s["section_title"],
            },
        )
        for idx, (s, vec) in enumerate(zip(sections, vectors))
    ]
    _upsert(prof.collection, points)
    return len(points)


def build_chunked(profile_key: str, sections: List[Dict] | None = None, *, recreate: bool = True) -> int:
    """Build a rechunked collection (sliding-window chunks with parent metadata)."""
    prof = config.profile(profile_key)
    assert prof.chunked, f"{profile_key} is not a chunked profile"
    sections = sections if sections is not None else load_sections()
    embeddings = get_embeddings(prof.embed_model)
    window = chunking.window_for(prof)

    ensure_collection(get_client(), prof.collection, prof.dim, recreate=recreate)

    records: List[Dict] = []
    for s in tqdm(sections, desc=f"chunking:{profile_key}"):
        parts = chunking.chunk_by_tokens(
            s["text"], prof.embed_model, max_tokens=window, overlap=config.CHUNK_OVERLAP
        )
        total = len(parts)
        for i, part_text in enumerate(parts, start=1):
            records.append(
                {
                    "text": part_text,
                    "law_name": s["law_name"],
                    "section_number": s["section_number"],
                    "section_title": s["section_title"],
                    "chunk_part": i,
                    "total_parts": total,
                }
            )

    vectors = embeddings.embed_documents([r["text"] for r in records])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload=rec)
        for rec, vec in zip(records, vectors)
    ]
    _upsert(prof.collection, points)
    return len(points)


def build_all(*, recreate: bool = True) -> Dict[str, int]:
    """Rebuild all four collections from scratch."""
    sections = load_sections()
    save_sections_checkpoint(sections)
    counts = {}
    for key in ("baseline_unchunked", "jina_unchunked"):
        counts[key] = build_unchunked(key, sections, recreate=recreate)
    for key in ("baseline_rechunked", "jina_rechunked"):
        counts[key] = build_chunked(key, sections, recreate=recreate)
    return counts

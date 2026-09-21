"""Central configuration — paths, model IDs, retrieval params, and API keys.

Claude (Anthropic) is the **only** supported LLM provider. The old repo's Groq /
Llama fallback has been intentionally removed.

API key resolution order (first non-empty wins):
    1. ``ANTHROPIC_API_KEY`` environment variable (incl. values loaded from ``.env``)
    2. ``anthropic.api_key`` in ``config.yaml`` at the repo root
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# ── Paths ───────────────────────────────────────────────────────────────────
# <repo root>/config/config.py  ->  parents[1] == repo root
ROOT_DIR = Path(os.environ.get("IMLR_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT_DIR / "data"
LAWS_DIR = DATA_DIR / "base_chunked_marriage_laws"
PROCESSED_DIR = DATA_DIR / "processed_data"
DB_DIR = ROOT_DIR / "database" / "qdrant_db"
# On-disk cache of query embeddings, so a query embedded once can be retrieved
# again without re-loading the embedding model.
EMBED_CACHE_DB = ROOT_DIR / "database" / "embedding_cache.sqlite"
# Toggle the transparent embedding cache (set IMLR_EMBED_CACHE=0 to force re-embed).
EMBED_CACHE_ENABLED = os.environ.get("IMLR_EMBED_CACHE", "1").strip() not in ("0", "false", "False", "")
RESULTS_DIR = ROOT_DIR / "results"
CONTEXTS_DIR = RESULTS_DIR / "contexts"
EVALUATED_DIR = RESULTS_DIR / "evaluated"
CONFIG_FILE = ROOT_DIR / "config.yaml"

# Benchmark files
BENCHMARK_CSV = PROCESSED_DIR / "benchmarking_data.csv"
BENCHMARK_EXPANDED_CSV = PROCESSED_DIR / "benchmarking_data_expanded.csv"

# Data-gathering artifacts (produced by notebooks/1_data_gathering.ipynb).
# Centralised here so notebooks/scripts never hard-code a path or filename.
LAWS_CSV = PROCESSED_DIR / "laws.csv"                        # master list scraped from India Code
LAWS_WITH_LINKS_CSV = PROCESSED_DIR / "laws_with_links.csv"  # raw Indian Kanoon links (auto)
FINAL_LAW_LINKS_CSV = PROCESSED_DIR / "final_law_with_links.csv"  # manually verified links
CHUNKED_LAWS_JSON = PROCESSED_DIR / "chunked_laws.json"      # re-chunked corpus (see chunking.py)
EXPANSION_CHECKPOINT_JSON = PROCESSED_DIR / "expansion_checkpoint.json"


# ── Models ──────────────────────────────────────────────────────────────────
# The single LLM used for query expansion, answer generation, and LLM-as-a-judge.
CLAUDE_MODEL = "claude-sonnet-4-6"

BASELINE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
JINA_EMBED_MODEL = "jinaai/jina-embeddings-v3"                    # 1024-dim
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BASELINE_DIM = 384
JINA_DIM = 1024
BASELINE_MAX_TOKENS = 512
JINA_MAX_TOKENS = 1024


# ── Qdrant collections ──────────────────────────────────────────────────────
COLLECTIONS = {
    "baseline_unchunked": "legal_acts_baseline",
    "jina_unchunked": "legal_acts_jina",
    "baseline_rechunked": "legal_baseline_chunked",
    "jina_rechunked": "legal_jina_chunked",
}

# Chunking (sliding window, token-level) — see chunking.py
CHUNK_OVERLAP = 50
BASELINE_CHUNK_WINDOW = 400
JINA_CHUNK_WINDOW = 1024

# Hybrid retrieval scoring: final = BM25_W * bm25 + CE_W * cross_encoder
BM25_WEIGHT = 0.3
CE_WEIGHT = 0.7
TOP_K_INITIAL = 10   # candidate pool pulled from the vector store
TOP_K_FINAL = 5      # sections returned after re-ranking + parent-child merge

# The configuration the production app / pipeline uses.
DEFAULT_CONFIG = "jina_rechunked"


@dataclass
class ConfigProfile:
    """One of the four evaluated retrieval configurations."""

    key: str
    collection: str
    embed_model: str
    dim: int
    chunked: bool
    use_expanded_query: bool  # rechunked configs retrieve with the expanded query
    hybrid: bool              # rechunked configs add BM25 + cross-encoder re-ranking


PROFILES: dict[str, ConfigProfile] = {
    "jina_unchunked": ConfigProfile(
        "jina_unchunked", COLLECTIONS["jina_unchunked"],
        JINA_EMBED_MODEL, JINA_DIM, chunked=False, use_expanded_query=False, hybrid=False,
    ),
    "baseline_unchunked": ConfigProfile(
        "baseline_unchunked", COLLECTIONS["baseline_unchunked"],
        BASELINE_EMBED_MODEL, BASELINE_DIM, chunked=False, use_expanded_query=False, hybrid=False,
    ),
    "jina_rechunked": ConfigProfile(
        "jina_rechunked", COLLECTIONS["jina_rechunked"],
        JINA_EMBED_MODEL, JINA_DIM, chunked=True, use_expanded_query=True, hybrid=True,
    ),
    "baseline_rechunked": ConfigProfile(
        "baseline_rechunked", COLLECTIONS["baseline_rechunked"],
        BASELINE_EMBED_MODEL, BASELINE_DIM, chunked=True, use_expanded_query=True, hybrid=True,
    ),
}


# ── Secrets ─────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    anthropic_api_key: str = ""
    huggingface_api_key: str = ""
    _yaml: dict = field(default_factory=dict, repr=False)


def _load_yaml() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def load_settings() -> Settings:
    """Resolve API keys from the environment first, then ``config.yaml``."""
    cfg = _load_yaml()
    anthropic_key = (
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or (cfg.get("anthropic") or {}).get("api_key", "").strip()
    )
    hf_key = (
        os.environ.get("HUGGINGFACE_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or (cfg.get("huggingface") or {}).get("api_key", "").strip()
    )
    return Settings(anthropic_api_key=anthropic_key, huggingface_api_key=hf_key, _yaml=cfg)


def require_anthropic_key(settings: Optional[Settings] = None) -> str:
    """Return the Claude API key or raise a clear, actionable error."""
    settings = settings or load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment "
            "variable (or add it to a .env file), or set `anthropic.api_key` in "
            f"{CONFIG_FILE}."
        )
    return settings.anthropic_api_key


def profile(key: str = DEFAULT_CONFIG) -> ConfigProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown config '{key}'. Choose from {list(PROFILES)}.")
    return PROFILES[key]

"""Pre-populate demo/demo_cache.json with canned questions.

Run this on a machine that has ANTHROPIC_API_KEY set (your work laptop).
Commit the resulting demo/demo_cache.json so the personal laptop can run
the app offline with DEMO_MODE=1.

Usage
-----
    cd <repo root>
    python demo/build_demo_cache.py
"""

import sys
from pathlib import Path

# Make sure the src package and repo-root config are importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from indian_marriage_legal_recommender.pipeline import answer  # noqa: E402
from indian_marriage_legal_recommender.demo_cache import list_cached, CACHE_FILE  # noqa: E402

# ── Add your demo questions here ─────────────────────────────────────────────
DEMO_QUESTIONS = [
    (
        "I am a Hindu woman from Kerala and my partner is a Christian man. "
        "Can we marry without converting?",
        "jina_rechunked",
        5,
    ),
    (
        "What are the grounds for divorce under the Hindu Marriage Act?",
        "jina_rechunked",
        5,
    ),
    (
        "Can a Muslim man have more than one wife in India?",
        "jina_rechunked",
        5,
    ),
    (
        "What is the minimum age of marriage for a girl under Indian law?",
        "jina_rechunked",
        5,
    ),
    (
        "How can two people of different religions marry under the Special Marriage Act?",
        "jina_rechunked",
        5,
    ),
    (
        "What are the rights of a Hindu woman after divorce?",
        "jina_rechunked",
        5,
    ),
    (
        "Can a Parsi marry a non-Parsi in India?",
        "jina_rechunked",
        5,
    ),
]
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Cache file: {CACHE_FILE}\n")
    for i, (question, profile, top_k) in enumerate(DEMO_QUESTIONS, 1):
        print(f"[{i}/{len(DEMO_QUESTIONS)}] {question[:70]}...")
        answer(question, profile_key=profile, top_k=top_k)
        print("  cached.\n")

    print(f"Done. {len(list_cached())} question(s) in cache.")


if __name__ == "__main__":
    main()

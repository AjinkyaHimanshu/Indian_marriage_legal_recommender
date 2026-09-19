"""Evaluation — LLM-as-a-judge relevance scoring and nDCG (notebooks 6 & 7).

Claude scores each retrieved chunk 0-5 for how well it answers the query (batched
in a single deterministic call), and nDCG measures whether the most relevant
chunks were ranked at the top.
"""

from __future__ import annotations

import json
import math
from typing import List, Sequence, Union

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import config
from .llm import shared_chat_model

_JUDGE_SYSTEM = "You are an expert Evaluator for a Legal AI System. Output ONLY valid JSON."

_JUDGE_HUMAN = """For EACH of the {n} recommendations below, rate how well it answers the user query.

RELEVANCE (0-5):
  5 = Perfectly resolves the query
  4 = Highly relevant, answers most of the query
  3 = Somewhat relevant, partially addresses the query
  2 = Tangentially related, minimal direct relevance
  1 = Barely related, vague connection
  0 = Completely irrelevant

USER QUERY: {query}

{recommendations}

Output ONLY valid JSON of the form:
{{"relevance_scores": [score_1, score_2, ..., score_{n}]}}"""


def _texts(contexts: Sequence[Union[str, Document]]) -> List[str]:
    return [c.page_content if isinstance(c, Document) else str(c) for c in contexts]


def score_relevance(query: str, contexts: Sequence[Union[str, Document]]) -> List[float]:
    """Return one 0-5 relevance score per context, judged by Claude."""
    texts = _texts(contexts)
    if not texts:
        return []
    block = "\n\n".join(f"RECOMMENDATION {i + 1}:\n{t}" for i, t in enumerate(texts))
    prompt = ChatPromptTemplate.from_messages(
        [("system", _JUDGE_SYSTEM), ("human", _JUDGE_HUMAN)]
    )
    llm = shared_chat_model(temperature=0.0, max_tokens=512, model=config.CLAUDE_MODEL)
    raw = (prompt | llm).invoke(
        {"n": len(texts), "query": query, "recommendations": block}
    ).content
    scores = json.loads(_extract_json(raw)).get("relevance_scores", [])
    # pad/truncate defensively so downstream nDCG is well-defined
    scores = [float(s) for s in scores][: len(texts)]
    scores += [0.0] * (len(texts) - len(scores))
    return scores


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text


def _dcg(scores: Sequence[float]) -> float:
    total = 0.0
    for i, s in enumerate(scores):
        total += s if i == 0 else s / math.log2(i + 1)
    return total


def calculate_ndcg(scores: Sequence[float]) -> float:
    """Normalized Discounted Cumulative Gain in [0, 1]."""
    if not scores:
        return 0.0
    ideal = _dcg(sorted(scores, reverse=True))
    return round(_dcg(scores) / ideal, 4) if ideal > 0 else 0.0

"""Query expansion — rewrite a plain-English question into legal terminology.

Implemented as a LangChain LCEL chain (prompt | Claude | string parser). Used both
by the live app and by notebook 4 to expand the 15 benchmark questions.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import config
from .llm import shared_chat_model

_SYSTEM = (
    "You are an Indian Legal Query Expansion Assistant. "
    "Output only the expanded query, with no explanations or preamble."
)

_HUMAN = """A user provides a plain-English legal query related to marriage in India.
Rewrite it using formal legal phrasing and add closely related legal terms, synonyms, and procedural keywords.

STRICT RULES:
- Preserve the original intent exactly — do NOT introduce new legal concepts not present in the query.
- Do NOT infer missing context (religion, gender, jurisdiction, specific grounds).
- Do NOT broaden the scope beyond the original question.
- Do NOT answer or interpret the query.
- Return a SINGLE expanded query of 100 words maximum, with no bullet points or headings.

USER QUESTION: "{question}"
EXPANDED SEARCH QUERY:"""


def build_chain(temperature: float = 0.2):
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    llm = shared_chat_model(temperature=temperature, max_tokens=256, model=config.CLAUDE_MODEL)
    return prompt | llm | StrOutputParser()


def expand_query(question: str, temperature: float = 0.2) -> str:
    """Expand a single question and return the cleaned expanded query."""
    return build_chain(temperature).invoke({"question": question}).strip()

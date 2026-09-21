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
    "You rewrite queries using generic legal vocabulary ONLY. "
    "You have no knowledge of specific statutes and must never name any. "
    "Output only the expanded query, with no explanations or preamble."
)

_HUMAN = """A user provides a plain-English legal query related to marriage in India.
Rewrite it using formal legal phrasing and add closely related legal terms, synonyms, and procedural keywords.

STRICT RULES:
- Preserve the original intent exactly — do NOT introduce new legal concepts not present in the query.
- Do NOT name, cite, or introduce any specific statute, Act, rule, section number, year,
  case name, court, or legal authority UNLESS it appears verbatim in the user's question.
  (e.g. if the user names only the "Hindu Marriage Act", do NOT add the "Anand Marriage Act",
  "Special Marriage Act", section numbers, or any other statute.)
- Do NOT infer missing context (religion, gender, jurisdiction, specific grounds).
- Do NOT broaden the scope beyond the original question.
- Do NOT answer or interpret the query.
- Use only GENERIC legal terminology (e.g. "marriage registration", "solemnization",
  "competent authority", "documentary requirements") rather than named laws.
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

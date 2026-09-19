"""Grounded answer generation — Claude answers strictly from retrieved context.

LangChain LCEL chain: prompt | Claude | string parser. The context is the list of
retrieved legal sections (LangChain ``Document`` objects or plain strings).
"""

from __future__ import annotations

from typing import List, Sequence, Union

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import config
from .llm import shared_chat_model

_SYSTEM = "You are a helpful Legal AI. Provide plain-English, accurate legal advice."

_HUMAN = """You are a highly professional Indian Marriage Attorney.
Based STRICTLY on the retrieved legal context below, answer the user's query.
If the context does not contain the answer, state clearly that you cannot answer based on the provided laws.

USER QUERY: {question}

RETRIEVED CONTEXT:
{context}"""


def _to_text(contexts: Sequence[Union[str, Document]]) -> str:
    parts = [c.page_content if isinstance(c, Document) else str(c) for c in contexts]
    return "\n\n".join(parts)


def build_chain(temperature: float = 0.2, max_tokens: int = 1024):
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    llm = shared_chat_model(temperature=temperature, max_tokens=max_tokens, model=config.CLAUDE_MODEL)
    return prompt | llm | StrOutputParser()


def generate_answer(
    question: str,
    contexts: Sequence[Union[str, Document]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    chain = build_chain(temperature, max_tokens)
    return chain.invoke({"question": question, "context": _to_text(contexts)}).strip()

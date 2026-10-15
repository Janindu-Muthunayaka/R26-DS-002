# query_si.py
import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

_PROMPT = ChatPromptTemplate.from_template(
    "Translate the question below into Sinhala, written the way a Sinhala "
    "newspaper would write it.\n"
    "Rules:\n"
    "- Write names of people, places and organisations in their usual Sinhala "
    "newspaper spelling.\n"
    "- Keep numbers as they are.\n"
    "- Do not answer the question and do not add anything.\n"
    "- Return only the Sinhala question, nothing else.\n\n"
    "Question: {question}"
)

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        _chain = _PROMPT | ChatOpenAI(model=model_name, temperature=0)
    return _chain


@lru_cache(maxsize=1024)
def _translate(english_query: str) -> str:
    response = _get_chain().invoke({"question": english_query})
    return (response.content or "").strip().strip('"')


def get_sinhala_query(english_query: str, provided_sinhala: str | None = None) -> str | None:
    if provided_sinhala and provided_sinhala.strip():
        return provided_sinhala.strip()
    if not english_query or not english_query.strip():
        return None
    try:
        return _translate(english_query.strip()) or None
    except Exception as e:
        print(f"[query_si] translation failed: {e!r} - BM25 skipped for this query")
        return None

# query_si.py
# BM25 needs a SINHALA query (the corpus is Sinhala script), but Component 4
# only sends an English translation. This module produces the Sinhala query:
#   1. if Component 4 ever sends the original Sinhala text (voice field
#      "original_text"), that is used as-is - no LLM call;
#   2. otherwise the English query is translated to Sinhala by the LLM.
# On any failure it returns None and the pipeline falls back to dense-only
# retrieval, so this step can never break a request.
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
    # Created lazily so importing this module never needs the API key.
    global _chain
    if _chain is None:
        from langchain_openai import ChatOpenAI
        _chain = _PROMPT | ChatOpenAI(model="gpt-5.4-mini", temperature=0)
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

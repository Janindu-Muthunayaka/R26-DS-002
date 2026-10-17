# websearch.py
import os
import unicodedata

from langchain_core.documents import Document

MAX_RESULTS = 3
REFUSAL_PHRASE = "ප්‍රමාණවත් තොරතුරු නොමැත"

# Appended to prompt_modifier on the second (web) generation pass.
WEB_PROMPT_NOTE = (
    "වැදගත්: මෙම තොරතුරු කියවන ලද පාඨයේ හෝ ලිපිවල නොමැති බවත්, වෙබ් සෙවුමකින් "
    "ලබාගත් බවත් පිළිතුරේ මුල් වාක්‍යයෙන් සඳහන් කරන්න. "
    "'[වෙබ්]' ලෙස සලකුණු කළ සාක්ෂි පමණක් භාවිතා කර පිළිතුරු දෙන්න."
)

_client = None


def _norm(s: str) -> str:
    """Strip invisible Sinhala joiners so refusal matching is reliable."""
    s = unicodedata.normalize("NFC", s or "")
    for ch in ("\u200d", "\u200c", "\u200b", "\u00ad"):
        s = s.replace(ch, "")
    return " ".join(s.split())


def is_refusal(answer: str) -> bool:
    return _norm(REFUSAL_PHRASE) in _norm(answer)


def _get_client():
    # Created lazily so importing this module never fails when the key is
    # missing (e.g. evaluation runs that never use web search).
    global _client
    if _client is None:
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                _client = TavilyClient(api_key=api_key)
        except ImportError:
            _client = None
    return _client


def web_search(query: str) -> list[Document]:
    """Returns web results as Documents. Returns [] on any failure (missing
    key, network error, no results) so the pipeline never breaks."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[web_search] TAVILY_API_KEY not set - skipping web search")
        return []
    client = _get_client()
    if client is None:
        return []
    try:
        res = client.search(
            query=query, max_results=MAX_RESULTS, search_depth="basic"
        )
    except Exception as e:
        print(f"[web_search] failed: {e!r}")
        return []

    docs = []
    for r in res.get("results", []):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        docs.append(Document(
            page_content=f"[වෙබ්] {r.get('title', '')}\n{content}",
            metadata={
                "source_type": "web",
                "title": r.get("title", ""),
                "source_url": r.get("url", ""),
                "score": r.get("score"),
            },
        ))
    return docs

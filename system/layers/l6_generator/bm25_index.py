# bm25_index.py
# Keyword (sparse) index over the SAME paragraph chunks the dense index uses.
#
# Sinhala is highly inflected (ප්‍රහාරය / ප්‍රහාරයේ / ප්‍රහාරයට ...), so plain
# whole-word BM25 misses many matches. Each word is therefore indexed as:
#     the whole word  +  all character 3-grams  +  all character 4-grams
# (set use_ngrams=False to get plain word-level BM25 for comparison).
#
# The index is built in memory from ingest.build_documents() the first time it
# is needed (takes a few seconds) and then reused for the whole process.
import heapq
import re
import threading
import unicodedata

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from .ingest import build_documents
except (ImportError, ValueError):
    from ingest import build_documents

NGRAM_SIZES = (3, 4)

# Invisible characters are stripped for MATCHING only (ZWJ included, so the
# query and the corpus are compared in the same form). The text shown to the
# LLM / user is never changed.
_STRIP = dict.fromkeys(map(ord, "\u200d\u200c\u200b\u00ad\ufeff"), None)
_WORD = re.compile(r"[\u0D80-\u0DFF]+|[A-Za-z]+|[0-9]+")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").translate(_STRIP).lower()


def tokenize(text: str, use_ngrams: bool = True) -> list[str]:
    words = _WORD.findall(normalize(text))
    if not use_ngrams:
        return words
    tokens = []
    for w in words:
        tokens.append(w)
        for n in NGRAM_SIZES:
            if len(w) > n:
                tokens.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return tokens


class BM25Index:
    def __init__(self, docs, use_ngrams: bool = True):
        self.docs = docs
        self.use_ngrams = use_ngrams
        self.by_id = {d.metadata["chunk_id"]: d for d in docs}
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi([tokenize(d.page_content, use_ngrams) for d in docs])
        else:
            self.bm25 = None

    def search(self, query: str, k: int = 20):
        """Returns [(Document, score), ...] best first. Only positive scores."""
        if self.bm25 is None:
            return []
        tokens = tokenize(query, self.use_ngrams)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
        return [(self.docs[i], float(scores[i])) for i in top if scores[i] > 0]


_indexes: dict[bool, BM25Index] = {}
_lock = threading.Lock()


def get_bm25(use_ngrams: bool = True) -> BM25Index:
    with _lock:
        if use_ngrams not in _indexes:
            print(f"[bm25] building index (use_ngrams={use_ngrams}) ...")
            _indexes[use_ngrams] = BM25Index(build_documents(), use_ngrams)
            print(f"[bm25] done: {len(_indexes[use_ngrams].docs)} chunks")
        return _indexes[use_ngrams]

# ingest.py
# Paragraph-level chunking of the article corpus (matches thesis Table 3.1):
#   chunk 0      = headline            -> id  {article_id}::p000
#   chunk 1..n   = paragraphs          -> ids {article_id}::p001, p002, ...
#
# Paragraphs that are very short are merged into the next one, and very long
# paragraphs are split on sentence boundaries, so no chunk is tiny or huge.
import json
import re
import unicodedata

from langchain_core.documents import Document

CORPUS_PATH = "corpus/articles_fixed.jsonl"
MIN_CHARS = 60     # shorter paragraphs are merged with the next one
MAX_CHARS = 900    # longer paragraphs are split at sentence boundaries

# Invisible junk removed from the text. ZWJ (U+200D) is KEPT because Sinhala
# conjuncts such as "ප්‍රහාරය" need it to render correctly.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u00ad\ufeff"), None)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_INVISIBLE)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_long(paragraph: str) -> list[str]:
    """Split one over-long paragraph into pieces of at most ~MAX_CHARS,
    cutting only at sentence ends (a single very long sentence stays whole)."""
    if len(paragraph) <= MAX_CHARS:
        return [paragraph]
    pieces, current = [], ""
    for sentence in _SENT_SPLIT.split(paragraph):
        if current and len(current) + 1 + len(sentence) > MAX_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_into_paragraphs(body: str) -> list[str]:
    paras = [" ".join(p.split()) for p in re.split(r"\n+", _clean(body))]
    paras = [p for p in paras if p]

    expanded = []
    for p in paras:
        expanded.extend(_split_long(p))

    merged, carry = [], ""
    for p in expanded:
        if carry:
            p = f"{carry} {p}"
            carry = ""
        if len(p) < MIN_CHARS:
            carry = p
        else:
            merged.append(p)
    if carry:                       # leftover short tail
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def build_documents():
    docs = []
    for row in load_jsonl(CORPUS_PATH):
        article_id = str(row["article_id"])
        headline = _clean(row.get("headline") or "").strip()
        paras = split_into_paragraphs(row.get("clean_body") or row.get("raw_body", ""))

        base_meta = {
            "source_type": "article",
            "article_id": article_id,
            "headline": headline,
            "section_category": row.get("section_category") or "",
            "publication_date": row.get("publication_date") or "",
            "source_url": row.get("source_url") or "",
            "total_paras": len(paras),
        }

        if headline:
            docs.append(Document(
                page_content=headline,
                metadata={**base_meta,
                          "chunk_id": f"{article_id}::p000",
                          "para_index": 0,
                          "is_headline": True,
                          "is_lead": False},
            ))

        for i, para in enumerate(paras, start=1):
            docs.append(Document(
                page_content=para,
                metadata={**base_meta,
                          "chunk_id": f"{article_id}::p{i:03d}",
                          "para_index": i,
                          "is_headline": False,
                          "is_lead": i == 1},
            ))
    return docs


def list_chunk_ids(limit=10):
    """Utility for testing — prints real chunk_ids you can drop into
    contracts.py's retrieved_chunk_id."""
    docs = build_documents()
    for d in docs[:limit]:
        print(d.metadata["chunk_id"], "->", d.page_content[:40].replace("\n", " "))

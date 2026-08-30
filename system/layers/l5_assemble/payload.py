"""
LAYER 5 — the payload Component 3 (RAG) receives.  OWNER: Ishara.

WHY THIS FILE EXISTS, AND WHY IT DOES NOT MATCH THE ORIGINAL SPEC
-----------------------------------------------------------------
`TempFormatPleaseRead.txt` in this folder, and `contracts.py` in Nadee's
component, both specify the payload as:

    {"corrected_text": "...",
     "tokens": [{"original", "corrected", "label", "confidence",
                 "was_changed"}]}

**Layer 4B cannot produce `label` or `confidence`, and must not pretend to.**

The deployed corrector is plain full-sequence mT5-small. There is no per-token
classifier anywhere in the reading path. The model that WOULD have produced a
label and a confidence per token is the SinBERT-gated span corrector — which
UNDERPERFORMED plain mT5. It is the project's negative result and it is not in
the pipeline.

Filling those two fields would mean generating numbers with nothing behind
them, in the one project whose strongest contribution is a carefully reported
negative.

WHAT THE TOKENS ARE INSTEAD: a word-level diff of `body_raw` against `body`,
computed with difflib. It says truthfully which words the corrector changed
and carries no confidence, because none was measured.

The alignment is APPROXIMATE: `correct()` splits into sentences, corrects
each, joins and dedups, so token counts need not match. Read `was_changed` as
"these strings differ here", not "the model decided to change this token".

SAFE EITHER WAY: Nadee's `parse_ocr_input()` requires only `corrected_text`
and defaults `tokens` to `[]` — and never reads `tokens` at all.
"""
from __future__ import annotations

import difflib
from typing import List

from core.schemas import Document

# A diff over a long article is O(n^2)-ish and the result is only ever
# provenance in a log. Cap it, and SAY when the cap bit — a screening step
# that does not report what it dropped is how the `g0` null marker survived.
MAX_DIFF_WORDS = 4000


def diff_tokens(raw: str, corrected: str,
                max_words: int = MAX_DIFF_WORDS) -> List[dict]:
    """Word-level diff of OCR output against corrected output.

    Returns [{'original', 'corrected', 'was_changed'}]. No label. No
    confidence. See the module docstring for why.
    """
    a = (raw or '').split()
    b = (corrected or '').split()
    if not a and not b:
        return []
    if not b:                      # correction skipped — nothing changed
        return [{'original': w, 'corrected': w, 'was_changed': False}
                for w in a[:max_words]]
    if not a:                      # no OCR to compare against
        return [{'original': '', 'corrected': w, 'was_changed': True}
                for w in b[:max_words]]

    a, b = a[:max_words], b[:max_words]
    out: List[dict] = []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            out += [{'original': w, 'corrected': w, 'was_changed': False}
                    for w in a[i1:i2]]
            continue
        left, right = a[i1:i2], b[j1:j2]
        # Pair positionally inside the changed block; the overhang on either
        # side is a deletion or an insertion. Crude, and honest about it.
        for k in range(max(len(left), len(right))):
            out.append({
                'original':  left[k] if k < len(left) else '',
                'corrected': right[k] if k < len(right) else '',
                'was_changed': True,
            })
    return out


def article_text(article) -> str:
    """What this article's text ACTUALLY IS, for every consumer.

    ONE definition, in one place. The phone, the RAG payload, the local
    commands and the readability gate must never disagree about what was read
    — and there are three candidate fields.

    Precedence: the accepted LLM post-edit if there is one, else the mT5
    correction, else the raw OCR.
    """
    cand = (article.body_polished or '').strip()
    if cand == "[DISCARD]":
        return ""
    return cand or (article.body or '').strip() or (article.body_raw or '').strip()


def article_title(article) -> str:
    """The title to use, preferring the polished output."""
    cand = (article.title_polished or '').strip()
    if cand == "[DISCARD]":
        return ""
    return cand or (article.title or '').strip() or (article.title_raw or '').strip()


def rag_payload(document: Document, with_tokens: bool = True) -> dict:
    """Flatten a Document into what Component 3 consumes."""
    bodies, raws, articles = [], [], []
    formatted_bodies = []
    for idx, art in enumerate(document.articles):
        body = article_text(art)
        raw = (art.body_raw or '').strip()
        if body:
            bodies.append(body)
        if raw:
            raws.append(raw)
        articles.append({
            'index': art.index,
            'title': article_title(art),
            'body': body,
            'polished': bool((art.body_polished or '').strip()),
            'glyph_p75': art.glyph_p75,
            'ocr_scale': art.ocr_scale,
            'verdict': art.verdict,
        })
        
        num = idx + 1
        t = article_title(art)
        header = f"Article {num}"
        if t:
            header += f": {t}"
        if body:
            formatted_bodies.append(f"{header}\n{body}")
        else:
            formatted_bodies.append(f"{header}\n(Empty)")

    corrected_text = '\n\n'.join(formatted_bodies)
    return {
        'corrected_text': corrected_text,
        'tokens': (diff_tokens('\n'.join(raws), corrected_text)
                   if with_tokens else []),
        # The field that keeps this honest. Do not remove it.
        'token_source': 'diff',
        'articles': articles,
        'warnings': list(document.warnings),
    }

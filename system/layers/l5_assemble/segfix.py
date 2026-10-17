"""
LAYER 5 — fix segmentation mistakes L4C reported: one story cut into two
consecutive articles (merge), or two stories in one article (split).

L4C's LLM only REPORTS (polish.py, "SEGMENTATION CHECK"); it is told to do so
only when it is obvious. Nothing it says is trusted as-is. Every hint must
pass the checks below against the final text, or it is ignored and the
segments stay exactly as they were:

  merge   only with the IMMEDIATELY NEXT article; both must have been seen by
          the LLM and both must still have text.
  split   `split_at` must be an exact phrase of the article's text, start right
          after a sentence end (or a line break), and leave at least
          MIN_PART_CHARS on each side. `split_title` is used only if it occurs
          word for word at the start of the second part.
  both    an article with a merge AND a split hint gets neither.

`body` is the research artifact (mT5 output) and is never overwritten: a
merged or split article is a NEW Article; the original segments are kept,
untouched, in Document.all_articles. Every change is returned as a note for
the document warnings, so it shows in the UI.

`SINHALA_ARTICLE_FIX=off` disables it (core/config.py).
"""
from __future__ import annotations

import re

from core.schemas import Box
from layers.l5_assemble.payload import article_text, article_title

MIN_PART_CHARS = 60          # each side of a split must be a real piece of text
MIN_SPLIT_PHRASE = 8         # chars; shorter phrases match by accident
SPLIT_TITLE_WINDOW = 200     # split_title must sit at the start of part 2
SENTENCE_END = '.!?।"\'”’)'


def _union(a: Box, b: Box) -> Box:
    return Box(x1=min(a.x1, b.x1), y1=min(a.y1, b.y1), x2=max(a.x2, b.x2), y2=max(a.y2, b.y2))


def _join(a: str, b: str) -> str:
    a, b = (a or '').strip(), (b or '').strip()
    return f'{a}\n\n{b}'.strip() if a and b else (a or b)


def _merge(a, b):
    """One Article from two consecutive segments of the same story."""
    polished = bool((a.body_polished or '').strip() or (b.body_polished or '').strip())
    title_from = a if article_title(a) else b
    return a.model_copy(deep=True, update={
        'box': _union(a.box, b.box),
        'regions': list(a.regions) + list(b.regions),
        'body_raw': _join(a.body_raw, b.body_raw),
        'body': _join(a.body, b.body),
        'body_polished': _join(article_text(a), article_text(b)) if polished else '',
        'title': title_from.title, 'title_raw': title_from.title_raw,
        'title_hybrid': title_from.title_hybrid, 'title_mat': title_from.title_mat,
        'title_polished': title_from.title_polished,
        'title_images': dict(title_from.title_images or {}),
        'note': (a.note + '; ' if a.note else '') + f'segments {a.index + 1}+{b.index + 1} merged (one story)',
    })


def _split_pos(text: str, phrase: str):
    """Index where part 2 starts, or None if the phrase fails any check."""
    if len(phrase) < MIN_SPLIT_PHRASE or text.count(phrase) != 1:
        return None
    pos = text.index(phrase)
    if pos < MIN_PART_CHARS or len(text) - pos < MIN_PART_CHARS:
        return None
    before = text[:pos]
    if '\n' not in before[-3:] and not before.rstrip().endswith(tuple(SENTENCE_END)):
        return None                                   # not at the start of a sentence
    return pos


def _cut(field: str, phrase: str):
    """(part1, part2) of a text field at `phrase` if it is there exactly once."""
    field = field or ''
    if phrase and field.count(phrase) == 1:
        k = field.index(phrase)
        return field[:k].strip(), field[k:].strip()
    return field, ''


def _split(a, phrase: str, pos: int, split_title: str):
    text = article_text(a)
    part1, part2 = text[:pos].strip(), text[pos:].strip()
    raw1, raw2 = _cut(a.body_raw, phrase)
    mt1, mt2 = _cut(a.body, phrase)
    title2 = split_title if split_title and split_title in part2[:SPLIT_TITLE_WINDOW] else ''
    note = f'segment {a.index + 1} split in two (two stories)'
    first = a.model_copy(deep=True, update={
        'body_raw': raw1, 'body': mt1, 'body_polished': part1,
        'note': (a.note + '; ' if a.note else '') + note,
    })
    second = a.model_copy(deep=True, update={
        'body_raw': raw2, 'body': mt2, 'body_polished': part2,
        'title': title2, 'title_raw': '', 'title_hybrid': '', 'title_mat': '',
        'title_polished': title2, 'title_images': {},
        'note': note + ('' if title2 else '; no headline of its own'),
    })
    return first, second


def apply(articles: list, results: list) -> tuple[list, list]:
    """(articles after the checked fixes, notes). `results` is polish_articles()
    output, aligned with `articles` (same order). Never raises: on any error the
    articles come back unchanged."""
    try:
        return _apply(list(articles), list(results or []))
    except Exception as e:
        return list(articles), [f'segmentation check skipped: {type(e).__name__}: {e}']


def _hint(results, i):
    r = results[i] if i < len(results) else None
    return (r or {}).get('segment')        # None = the LLM never saw this article


def _apply(articles, results):
    n, notes = len(articles), []
    hints = [_hint(results, i) for i in range(n)]

    # Merges first, left to right; chains (1+2+3) are allowed.
    out, i = [], 0
    while i < n:
        cur = articles[i]
        used = {i}
        while i + 1 < n:
            h, nh = hints[i] or {}, hints[i + 1]
            if 'merge_with' not in h or 'split_at' in h:
                break
            ok = (h['merge_with'] == str(i + 1) and nh is not None
                  and 'split_at' not in (nh or {})
                  and article_text(cur) and article_text(articles[i + 1]))
            if not ok:
                notes.append(f'Segmentation: ignored merge hint on article {i + 1} '
                             f'(must point to the next article, both readable)')
                break
            notes.append(f'Segmentation: articles {articles[i].index + 1} and '
                         f'{articles[i + 1].index + 1} merged - one story')
            cur = _merge(cur, articles[i + 1])
            i += 1
            used.add(i)
        out.append((cur, used))
        i += 1

    # Splits, only on articles that were not part of a merge.
    final = []
    for art, used in out:
        if len(used) == 1:
            k = next(iter(used))
            h = hints[k] or {}
            if 'split_at' in h and 'merge_with' not in h:
                text = article_text(art)
                pos = _split_pos(text, h['split_at'])
                if pos is None:
                    notes.append(f'Segmentation: ignored split hint on article {art.index + 1} '
                                 f'(phrase not found once at a sentence start, or a part too short)')
                else:
                    first, second = _split(art, h['split_at'], pos, h.get('split_title', ''))
                    notes.append(f'Segmentation: article {art.index + 1} split in two - two stories')
                    final.extend([first, second])
                    continue
        final.append(art)

    for idx, a in enumerate(final):
        a.index = idx
    return final, notes

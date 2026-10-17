"""
Getting text into the store.

THE PROBLEM THIS SOLVES. `Work/Nadee/ingest.py` reads `corpus/articles.jsonl`.
That file does not exist anywhere in the repository, has never existed, and
nobody has produced it — which is why Component 3 has never run end to end.

Two answers, and the system uses both:

**1. It indexes what it reads.** Every article the reader captures is stored,
so the corpus builds itself out of actual use. For a personal reading
assistant that is the more useful corpus, and it needs nobody else to deliver.

**2. A seed corpus, if there is one.** Point `--seed` at a folder of `.txt`
files or a `.jsonl` in the shape Nadee's ingest expected, and it loads.

Chunking is 400 characters with 50 of overlap — the same values as
`Work/Nadee/ingest.py`, on sentence boundaries rather than the recursive
character splitter, because Sinhala sentence ends are unambiguous here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CHUNK_CHARS = 400
CHUNK_OVERLAP = 50
SENT_END = re.compile(r'(?<=[.!?।])\s+')


def split_chunks(text: str, size: int = CHUNK_CHARS,
                 overlap: int = CHUNK_OVERLAP) -> list:
    text = re.sub(r'\s+', ' ', text or '').strip()
    if not text:
        return []
    sents = [s.strip() for s in SENT_END.split(text) if s.strip()] or [text]

    out, buf = [], ''
    for s in sents:
        if buf and len(buf) + 1 + len(s) > size:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ''
            buf = (tail + ' ' + s).strip() if tail else s
        else:
            buf = f'{buf} {s}'.strip()
    if buf:
        out.append(buf)
    return out


def records_from_text(text: str, source_type: str, base_id: str,
                      metadata: dict = None) -> list:
    meta = dict(metadata or {})
    meta['source_type'] = source_type
    recs = []
    for i, chunk in enumerate(split_chunks(text)):
        m = dict(meta)
        m['chunk_id'] = f'chunk_{base_id}_{i}'
        recs.append({'chunk_id': m['chunk_id'], 'text': chunk, 'metadata': m})
    return recs


def records_from_jsonl(path: Path) -> list:
    """The shape Work/Nadee/ingest.py expected, so a real corpus drops in."""
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        text = row.get('clean_body') or row.get('raw_body') or row.get('text')
        if not text:
            continue
        aid = str(row.get('article_id') or row.get('id') or len(out))
        out += records_from_text(text, 'article', aid, {
            'article_id': aid,
            'headline': row.get('headline', ''),
            'section_category': row.get('section_category', ''),
            'publication_date': row.get('publication_date', ''),
            'source_url': row.get('source_url', ''),
        })
    return out


def records_from_folder(folder: Path) -> list:
    """Every .txt and .jsonl under `folder`. One .txt is one article."""
    folder = Path(folder)
    out = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob('*')):
        if p.suffix.lower() == '.jsonl':
            out += records_from_jsonl(p)
        elif p.suffix.lower() == '.txt':
            try:
                text = p.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue
            out += records_from_text(text, 'article', p.stem,
                                     {'article_id': p.stem, 'headline': '',
                                      'source_url': str(p)})
    return out

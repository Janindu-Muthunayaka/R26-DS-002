"""
LAYER 5 — document assembly.  OWNER: shared.

Collects finished Articles into a Document in reading order, drops rejected
ones, and records warnings the phone app can speak.
"""
from core.schemas import Article, Document


def assemble(articles, source_frames, timings=None) -> Document:
    from layers.l5_assemble.payload import article_text, article_title
    warnings = []
    keep = []
    for a in sorted(articles, key=lambda x: x.index):
        if a.verdict == 'reject':
            warnings.append(f'Article {a.index + 1} skipped: {a.note}')
            continue
        if a.verdict == 'warn' and a.note:
            warnings.append(f'Article {a.index + 1}: {a.note}')
        
        body = article_text(a).strip()
        title = article_title(a).strip()
        
        if not body or body == "[DISCARD]" or title == "[DISCARD]":
            continue
            
        keep.append(a)
    for i, a in enumerate(keep):
        a.index = i
    return Document(source_frames=source_frames, articles=keep,
                    timings=timings or {}, warnings=warnings)

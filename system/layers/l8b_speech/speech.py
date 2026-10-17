"""
LAYER 8B — Speech / TTS.  OWNER: Bumal.

Extracts and formats TTS text data from the Document and generator output
to be sent back to the phone application for reading aloud.
"""
from __future__ import annotations
from typing import Optional
from core.schemas import Document
from layers.l5_assemble.payload import article_text


def get_tts_text(document: Document) -> Optional[str]:
    """
    Extracts the speakable TTS text data from the document.
    Prioritizes latest generator answer if follow-up generations exist, otherwise
    generates the initial scan summary (number of articles + title of each article with number).
    """
    if not document:
        return None

    # Check for latest generation answer in document
    if getattr(document, 'generations', None) and len(document.generations) > 0:
        latest = document.generations[-1]
        text = latest.get('answer') or latest.get('answer_si')
        if text:
            return text.strip()

    if not document.articles:
        return "කිසිදු ලිපියක් හඳුනාගෙන නොමැත."

    # Initial scan summary: number of articles + title of each article along with its article no.
    n = len(document.articles)
    titles_with_no = []
    for idx, a in enumerate(document.articles):
        t = (a.title_polished or a.title or a.title_raw or '').strip()
        if t:
            titles_with_no.append(f"ලිපිය {idx + 1}: {t}")

    msg = f"ලිපි {n} ක් හඳුනාගෙන ඇත."
    if titles_with_no:
        msg += " එම ලිපිවල ශීර්ෂයන් වන්නේ: " + ", ".join(titles_with_no)
    else:
        msg += " නමුත් ඒවායේ ශීර්ෂයන් හඳුනාගත නොහැක."

    return msg


def speak(document: Document) -> Optional[str]:
    """
    Speech endpoint for Layer 8B.

    Returns audio URL if audio file exists on disk, otherwise None
    while TTS text data is handled via get_tts_text() and sent to the phone app.
    """
    if document is None:
        return None
    return None


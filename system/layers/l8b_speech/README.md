# Layer 6 — RAG and TTS

**OWNER: Bumal (Components 3 and 4). Do not edit this folder.**

## Contract

Input : `Document` (see `core/schemas.py`)
Output: audio file path, or a URL the phone can fetch

Implement `speak(document) -> str` in `speech.py`.

## Latency note

Return the **title audio first** if you can. A Sinhala headline takes three
to four seconds to speak, which covers the body correction time, so the user
perceives almost no wait. This is why the pipeline keeps title and body
separate.

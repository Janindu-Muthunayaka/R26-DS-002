"""
LAYER 0 — voice interaction.  COMPONENT 4. OWNER: Bumal.

Layer 0 because it runs BEFORE everything else in the question path: the user
speaks, and what they said decides whether anything downstream runs at all.

IN : Sinhala text (transcribed on the phone) + a user id
OUT: the dict Component 4 already returns from `handle_voice_command()`:

    {"route": "GENERATE" | "TTS_REPLAY",
     "intent": "SUMMARIZE" | "ELABORATE" | "REPEAT" | ...,
     "english_translation": "...",
     "style_class": "Simple" | "Detailed" | "StepByStep",
     "prompt_modifier": "...",
     "personalization_flags": {...},
     "retrieved_chunk_id": None,
     "correction_applied": None}

That shape is not invented here. It is what `main_flow.handle_voice_command`
returns and exactly what Nadee's `adapters.parse_voice_input` requires, which
is the one place in this project where two components already agreed on an
interface without being asked to.

TWO MODES
---------
`http`  POST to VOICE_URL/interpret. Component 4 runs in its own process with
        its own venv — it pins numpy 2.4.4 and transformers 5.7.0 against this
        system's 1.26.4 / 5.1.0, and its intent path loads NLLB-600M and calls
        a local Ollama daemon. See core/config.py.

`stub`  The default. See below.

WHAT THE STUB IS, AND WHAT IT IS NOT
------------------------------------
**It is plumbing. It is not intent detection.** It exists so the `/ask` loop
can be built, tested and demonstrated before Component 4 is wrapped, and so a
Component 4 outage degrades one feature instead of killing the demo.

It recognises a handful of LITERAL command words and routes everything else to
GENERATE. It does not translate, it does not classify, and it does not learn a
style. Every dict it returns carries `"source": "stub"` so that nothing
downstream — and no figure in any chapter — can mistake it for Component 4's
classifier output.

Do not report anything measured through this stub as an intent-detection
result.
"""
from __future__ import annotations

from core import svc
from core.config import (VOICE_MODE, VOICE_TIMEOUT_S, VOICE_URL,
                         VOICE_USER_ID)

# Fields `adapters.parse_voice_input` raises on if they are missing.
REQUIRED = ('route', 'intent', 'english_translation', 'style_class',
            'prompt_modifier', 'personalization_flags')

# ==========================================================================
# THE LOCAL COMMAND VOCABULARY
# ==========================================================================
# Literal words. Sinhala first, then the English a bilingual user may reach
# for — and which makes these testable from a keyboard. Anything not on this
# list goes to GENERATE; the stub never guesses.
#
# ORDER MATTERS and the list is ordered, not a dict. "මුල සිට කියවන්න" (read
# from the start) contains "කියවන්න" (read), so FIRST must be tested before
# READ_ALOUD or every navigation command collapses into "read it all again".
#
# THE SINHALA HERE WAS NOT WRITTEN BY A NATIVE SPEAKER. The English aliases
# work regardless, so testing is never blocked on getting the Sinhala right —
# but have the Sinhala checked before the viva, and treat a command that does
# not trigger as a vocabulary problem before suspecting the code.
_COMMANDS = [
    ('STOP',     ('නවත්වන්න', 'නවතන්න', 'නවත්තන්න', 'stop', 'quiet')),
    ('NEXT',     ('ඊළඟ', 'මීළඟ', 'next', 'continue')),
    ('PREVIOUS', ('කලින්', 'පෙර ', 'previous', 'back', 'go back')),
    ('FIRST',    ('මුල සිට', 'මුලින්', 'ආරම්භයේ', 'from the start',
                  'start again', 'beginning')),
    ('LENGTH',   ('කොපමණ', 'දිගද', 'වචන කීයද', 'how long', 'how many words')),
    ('TITLE',    ('ශීර්ෂය', 'මාතෘකාව', 'headline', 'title')),
    ('WARNINGS', ('මඟ හැරුණ', 'මගහැරුණ', 'what did i miss', 'did i miss',
                  'missed')),
    ('REPEAT',   ('නැවත', 'ආපසු', 'යළි', 'repeat', 'again')),
    # 'කියන්න' ("say / tell me") is NOT here, deliberately. It swallowed
    # "මේ ලිපිය ගැන කියන්න" — "tell me about this article" — and routed a
    # real question to READ_ALOUD, so the listener got the article back
    # instead of an answer. A keyword that appears in ordinary questions is
    # not a command word.
    ('READ_ALOUD', ('කියවන්න', 'read aloud', 'read it', 'read the')),
]


def _stub(text: str) -> dict:
    t = (text or '').strip()
    low = t.lower()
    intent, route = 'ASK', 'GENERATE'
    for name, words in _COMMANDS:
        if any(w in low for w in words):
            intent, route = name, 'LOCAL'
            break
    return {
        'route': route,
        'intent': intent,
        # NOT a translation. The stub does not translate; it passes the
        # Sinhala through under the field name the contract uses, so the shape
        # is right and the content is visibly untranslated.
        'english_translation': t,
        'style_class': 'Detailed',
        'prompt_modifier': '',
        'personalization_flags': {},
        'retrieved_chunk_id': None,
        'correction_applied': None,
        'source': 'stub',
    }


def _degraded(text: str, reason: str) -> dict:
    """Component 4 was asked and could not answer. Fall back to the stub, and
    carry the reason so `/ask` can log why the personalisation is missing."""
    out = _stub(text)
    out['source'] = 'stub-fallback'
    out['warning'] = f'voice service unavailable ({reason})'
    return out


def interpret(text: str, user_id: str = None,
              retrieved_chunk_id: str = None, mode: str = None) -> dict:
    """Sinhala utterance -> Component 4's routing dict. Never raises."""
    mode = (mode or VOICE_MODE or 'stub').lower()
    if mode != 'http':
        return _stub(text)

    body = {'text': text or '',
            'user_id': user_id or VOICE_USER_ID,
            'retrieved_chunk_id': retrieved_chunk_id}
    reply, reason = svc.post_json(f'{VOICE_URL.rstrip("/")}/interpret', body,
                                  timeout_s=VOICE_TIMEOUT_S)
    if reply is None:
        return _degraded(text, reason)

    missing = [k for k in REQUIRED if k not in reply]
    if missing:
        # Loudly, rather than half-using a reply that will fail two layers
        # further on inside parse_voice_input.
        return _degraded(text, f'reply missing {missing}')

    reply.setdefault('retrieved_chunk_id', retrieved_chunk_id)
    reply.setdefault('correction_applied', None)
    # tools/stub_services.py answers this endpoint with `stub: true`. Without
    # this line a stub service passes validation and is stamped 'component4',
    # and the "nothing routed through a stub is a result" guarantee quietly
    # stops holding for exactly the setup most likely to be used in testing.
    reply['source'] = 'stub-service' if reply.get('stub') else 'component4'
    return reply

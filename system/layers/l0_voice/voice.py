"""
LAYER 0 — voice interaction.  COMPONENT 4. OWNER: Bumal.

Layer 0 because it runs BEFORE everything else in the question path: what the
user said decides whether anything downstream runs at all.

IN : Sinhala text (transcribed on the phone) + a user id
OUT: the dict Component 4 already returns from `handle_voice_command()` —
     {route, intent, english_translation, style_class, prompt_modifier,
      personalization_flags, retrieved_chunk_id, correction_applied}

That shape is not invented here. It is what `main_flow.handle_voice_command`
returns and exactly what Nadee's `adapters.parse_voice_input` requires — the
one place in this project where two components agreed on an interface without
being asked to.

TWO MODES. `http` POSTs to VOICE_URL/interpret (Component 4 in its own venv:
it pins numpy 2.4.4 / transformers 5.7.0 against this system's 1.26.4 / 5.1.0,
loads NLLB-600M and calls a local Ollama daemon). `stub` is the default.

WHAT THE STUB IS, AND WHAT IT IS NOT. **It is plumbing. It is not intent
detection.** It recognises LITERAL command words and routes everything else to
GENERATE. Every dict it returns carries `"source": "stub"` so nothing
downstream — and no figure in any chapter — can mistake it for Component 4's
classifier output.
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
# for — and which makes these testable from a keyboard.
#
# ORDER MATTERS and the list is ordered, not a dict. "මුල සිට කියවන්න" (read
# from the start) contains "කියවන්න" (read), so FIRST must be tested before
# READ_ALOUD or every navigation command collapses into "read it all again".
#
# THE SINHALA HERE WAS NOT WRITTEN BY A NATIVE SPEAKER. The English aliases
# work regardless, so testing is never blocked on getting the Sinhala right.
_COMMANDS = [
    ('STOP',      ('නවත්වන්න', 'නවතන්න', 'නවත්තන්න', 'stop', 'quiet')),
    ('NEXT',      ('ඊළඟ', 'මීළඟ', 'next', 'continue')),
    ('PREVIOUS',  ('කලින්', 'පෙර ', 'previous', 'back', 'go back')),
    ('FIRST',     ('මුල සිට', 'මුලින්', 'ආරම්භයේ', 'from the start',
                   'start again', 'beginning')),
    ('LENGTH',    ('කොපමණ', 'දිගද', 'වචන කීයද', 'how long', 'how many words')),
    ('TITLE',     ('ශීර්ෂය', 'මාතෘකාව', 'headline', 'title')),
    ('WARNINGS',  ('මඟ හැරුණ', 'මගහැරුණ', 'what did i miss', 'did i miss',
                   'missed')),
    ('REPEAT',    ('නැවත', 'ආපසු', 'යළි', 'repeat', 'again')),
    # Article numbers. Every ordinal form is tied to ලිපිය (article), so
    # "දෙවැනි තැන කාටද" (who came second?) stays a question. "පළමුවන ලිපිය"
    # was missing until 17 Oct 2026 and "read the first article" went to RAG.
    ('ARTICLE_1', ('ලිපිය 1', 'ලිපිය එක', 'ලිපිය අංක 1', 'ලිපි අංක 1', 'පළමු ලිපිය',
                   'පළමුවන ලිපිය', 'පළමුවැනි ලිපිය', 'පළමුවෙනි', 'පළවෙනි ලිපිය',
                   'පලමු ලිපිය', 'පලමුවන ලිපිය', '1 වන ලිපිය', '1 වෙනි ලිපිය',
                   'article 1', 'article one', 'first article')),
    ('ARTICLE_2', ('ලිපිය 2', 'ලිපිය දෙක', 'ලිපිය අංක 2', 'ලිපි අංක 2', 'දෙවන ලිපිය',
                   'දෙවැනි ලිපිය', 'දෙවෙනි ලිපිය', 'දෙවනි ලිපිය', '2 වන ලිපිය',
                   '2 වෙනි ලිපිය', 'article 2', 'article two', 'second article')),
    ('ARTICLE_3', ('ලිපිය 3', 'ලිපිය තුන', 'ලිපිය අංක 3', 'ලිපි අංක 3', 'තුන්වන ලිපිය',
                   'තුන්වැනි ලිපිය', 'තුන්වෙනි ලිපිය', '3 වන ලිපිය', '3 වෙනි ලිපිය',
                   'article 3', 'article three', 'third article')),
    ('ARTICLE_4', ('ලිපිය 4', 'ලිපිය හතර', 'ලිපිය අංක 4', 'ලිපි අංක 4', 'සිව්වන ලිපිය',
                   'සිව්වැනි ලිපිය', 'හතරවන ලිපිය', 'හතරවෙනි ලිපිය', '4 වන ලිපිය',
                   '4 වෙනි ලිපිය', 'article 4', 'article four', 'fourth article')),
    ('ARTICLE_5', ('ලිපිය 5', 'ලිපිය පහ', 'ලිපිය අංක 5', 'ලිපි අංක 5', 'පස්වන ලිපිය',
                   'පස්වැනි ලිපිය', 'පස්වෙනි ලිපිය', 'පහවෙනි ලිපිය', '5 වන ලිපිය',
                   '5 වෙනි ලිපිය', 'article 5', 'article five', 'fifth article')),
    # 'කියන්න' ("say / tell me") is NOT here, deliberately. It swallowed
    # "මේ ලිපිය ගැන කියන්න" — "tell me about this article" — and routed a real
    # question to READ_ALOUD. A word that appears in ordinary questions is not
    # a command word.
    ('READ_ALOUD', ('කියවන්න', 'read aloud', 'read it', 'read the')),
]


# Component 4's own modifier checks (pure regex, no model): the stub must
# recognise "speak faster", "more pauses", "again", "simply" exactly as the
# real service does, or a fallback sends a command to RAG as a question
# (TURN_CONDITIONS.xlsx G7).
try:
    from layers.l7_personalization.intent_detection.modifiers import build_frame as _build_frame
except Exception:                                    # pragma: no cover
    _build_frame = None

_STYLE_INTENT = {'Simple': 'SIMPLIFY', 'Detailed': 'ELABORATE', 'StepByStep': 'STEP_BY_STEP'}
_REPHRASE_WORDS = ('වෙනත් විදියකට', 'වෙන විදියකට', 'වෙන විදිහට', 'වෙනත් වචන', 'වෙන වචන',
                   'another way', 'other words', 'rephrase')
# Commands that keep their own action when a delivery word rides along:
# "ඉක්මනින් ඊළඟ එක" is NEXT at a faster rate, not a bare speed change.
_KEEP_WITH_DELIVERY = {'STOP', 'NEXT', 'PREVIOUS', 'FIRST', 'LENGTH', 'TITLE', 'WARNINGS',
                       'REPEAT', 'READ_ALOUD', 'ARTICLE_1', 'ARTICLE_2', 'ARTICLE_3',
                       'ARTICLE_4', 'ARTICLE_5'}


def _frame(text: str) -> dict:
    if _build_frame is None:
        return {'intent': 'UNKNOWN', 'modifiers': {}, 'evidence': {}}
    try:
        # Typed English (testing from a keyboard) goes through the English checks.
        if any('඀' <= ch <= '෿' for ch in text):
            return _build_frame(text, '', 'UNKNOWN')
        return _build_frame('', text, 'UNKNOWN')
    except Exception:
        return {'intent': 'UNKNOWN', 'modifiers': {}, 'evidence': {}}


def _command(low: str, allow_prefix: bool):
    # Prefix / exact word check for article numbers in Sinhala. Off when a
    # delivery word is present: "එක දිගට කියන්න" (say it in one go) starts
    # with එක but is not "article one".
    if allow_prefix:
        for n, word in ((3, 'තුන'), (1, 'එක'), (2, 'දෙක'), (4, 'හතර'), (5, 'පහ')):
            if low.startswith(word):
                return f'ARTICLE_{n}'
    for n in (3, 1, 2, 4, 5):
        if low == str(n) or f' {n}' in low:
            return f'ARTICLE_{n}'
    for name, words in _COMMANDS:
        if any(w in low for w in words):
            return name
    return None


def _stub(text: str) -> dict:
    t = (text or '').strip()
    low = t.lower()
    frame = _frame(t)
    mods = frame.get('modifiers') or {}
    delivery = bool(mods.get('speed') or mods.get('pause') or mods.get('chunk')
                    or mods.get('volume'))
    cmd = _command(low, allow_prefix=not delivery)
    intent, route = 'ASK', 'GENERATE'

    if mods.get('style'):
        # A style word needs new words: GENERATE. A navigation command keeps
        # its name so the turn policy knows WHICH text to restyle.
        if cmd and (cmd.startswith('ARTICLE_') or cmd == 'NEXT'):
            intent, route = cmd, 'GENERATE'
        else:
            intent, route = _STYLE_INTENT[mods['style']], 'GENERATE'
    elif mods.get('summary') and cmd and cmd.startswith('ARTICLE_'):
        intent, route = cmd, 'GENERATE'          # "ලිපිය 1 සාරාංශය"
    elif any(w in low for w in _REPHRASE_WORDS):
        intent, route = 'REPHRASE', 'GENERATE'
    elif mods.get('repeat') and cmd in (None, 'REPEAT', 'READ_ALOUD'):
        intent, route = 'REPEAT', 'LOCAL'        # "හෙමින් ආයෙ කියන්න" = repeat, slower
    elif cmd:
        if delivery and cmd not in _KEEP_WITH_DELIVERY:
            intent, route = 'CHANGE_SPEED', 'LOCAL'
        else:
            intent, route = cmd, 'LOCAL'
    elif delivery:
        intent, route = ('VOLUME' if mods.get('volume') and not mods.get('speed')
                         else 'CHANGE_SPEED'), 'LOCAL'
    elif mods.get('speed_mentioned') and len(low.split()) <= 4:
        intent, route = 'CHANGE_SPEED', 'LOCAL'  # "වේගය වෙනස් කරන්න": ask which way
    return {
        'route': route,
        'intent': intent,
        'frame': frame,
        'sinhala_input': t,
        # NOT a translation. The stub does not translate; it passes the
        # Sinhala through under the contract's field name, so the shape is
        # right and the content is visibly untranslated.
        'english_translation': t,
        'style_class': 'Detailed',
        'prompt_modifier': 'Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.',
        'personalization_flags': {},
        'retrieved_chunk_id': None,
        'correction_applied': None,
        'source': 'stub',
        'style_source': 'cold_start',
        'user_profile': {
            'n_confirmed': 0,
            'history_weights': {'Simple': 0.0, 'Detailed': 0.0, 'StepByStep': 0.0},
            'dominant_preference': None
        },
        'speed': 1.0,
        'speech_rate': 1.0,
        'learned': False,
    }


def _degraded(text: str, reason: str) -> dict:
    """Component 4 was asked and could not answer. Fall back to the stub, and
    carry the reason so `/ask` can log why the personalisation is missing."""
    out = _stub(text)
    out['source'] = 'stub-fallback'
    out['style_source'] = 'degraded_fallback'
    out['warning'] = f'voice service unavailable ({reason})'
    return out


def interpret(text: str, user_id: str = None,
              retrieved_chunk_id: str = None, mode: str = None,
              dry_run: bool = False, last_spoken_ctx: dict = None) -> dict:
    """Sinhala utterance -> Component 4's routing dict. Never raises.

    dry_run: Component 4 decides the style for `user_id` but logs and learns
    nothing. Used to re-generate an old answer under another user.
    last_spoken_ctx: {has_last, next_turn, age_s} — this session holds what
    was last spoken; Component 4 needs it for its REPEAT learning rule."""
    # Connect l8a_listen: if no text was provided (e.g. debugging/testing), 
    # optionally grab it from the local microphone if PyAudio is installed.
    if not text:
        try:
            from layers.l8a_listen.stt.google_stt_live import listen_and_transcribe
            print("[l8a_listen] Connecting to microphone...")
            captured = listen_and_transcribe()
            if captured:
                text = captured
        except ImportError:
            pass

    mode = (mode or VOICE_MODE or 'stub').lower()
    if mode != 'http':
        return _stub(text)

    body = {'text': text or '',
            'user_id': user_id or VOICE_USER_ID,
            'retrieved_chunk_id': retrieved_chunk_id,
            'dry_run': bool(dry_run),
            'last_spoken_ctx': last_spoken_ctx}
    reply, reason = svc.post_json(f'{VOICE_URL.rstrip("/")}/interpret', body,
                                  timeout_s=VOICE_TIMEOUT_S)
    if reply is None:
        return _degraded(text, reason)

    # For SYSTEM_COMMAND or navigation routes, personalization style fields
    # do not apply; populate sensible defaults so validation succeeds and
    # the command is mapped for local session handling.
    if reply.get('route') == 'SYSTEM_COMMAND':
        reply.setdefault('style_class', 'Detailed')
        reply.setdefault('prompt_modifier', 'Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.')
        reply.setdefault('personalization_flags', {})
        reply['route'] = 'LOCAL'
        cmd = reply.get('command')
        if cmd == 'GO_FORWARD':
            reply['intent'] = 'NEXT'
        elif cmd == 'STOP':
            reply['intent'] = 'STOP'

    missing = [k for k in REQUIRED if k not in reply]
    if missing:
        # Loudly, rather than half-using a reply that will fail two layers
        # further on inside parse_voice_input.
        return _degraded(text, f'reply missing {missing}')

    reply.setdefault('retrieved_chunk_id', retrieved_chunk_id)
    reply.setdefault('correction_applied', None)
    reply.setdefault('speed', 1.0)
    reply.setdefault('speech_rate', 1.0)
    reply.setdefault('pause_scale', 1.0)
    reply.setdefault('chunk_size', 3)
    reply['sinhala_input'] = text
    # tools/stub_services.py answers this endpoint with `stub: true`. Without
    # this line a stub service passes validation and is stamped 'component4',
    # and the "nothing routed through a stub is a result" guarantee quietly
    # stops holding for exactly the setup most likely to be used in testing.
    reply['source'] = 'stub-service' if reply.get('stub') else 'component4'
    return reply

# Same text as Component 4's STYLE_PROMPT_MODIFIERS (l7 main_flow.py). Kept
# here so a turn the server starts itself (the first scan after a capture)
# can be styled for the user without running intent detection on it.
STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.",
    "StepByStep": "Break the explanation into clear numbered steps, one action or idea per step.",
}


def style_for_user(user_id: str) -> dict:
    """The user's current dominant style, read from their profile. Nothing is
    logged or learned. Falls back to Detailed if the profile is unavailable."""
    profile = get_profile(user_id)
    style = (profile or {}).get('dominant_preference')
    source = 'user_profile'
    if style not in STYLE_PROMPT_MODIFIERS:
        style, source = 'Detailed', 'default'
    return {'style_class': style, 'prompt_modifier': STYLE_PROMPT_MODIFIERS[style],
            'style_source': source, 'user_profile': profile}


def get_profile(user_id: str) -> dict:
    mode = (VOICE_MODE or 'stub').lower()
    if mode == 'http':
        reply, _ = svc.get_json(f'{VOICE_URL.rstrip("/")}/profile/{user_id}', timeout_s=5)
        if reply: return reply
    return {
        'n_confirmed': 0,
        'history_weights': {'Simple': 0.0, 'Detailed': 0.0, 'StepByStep': 0.0},
        'dominant_preference': None
    }



# ==========================================================================
# USER REGISTRY (Component 4). Proxied so the web UI talks to one server.
# Returns (reply, '') or (None, reason); never raises.
# ==========================================================================
def _voice_url(path: str) -> str:
    return f'{VOICE_URL.rstrip("/")}{path}'


def _need_http():
    if (VOICE_MODE or 'stub').lower() != 'http':
        return 'user profiles need SINHALA_VOICE_MODE=http (the L7 service)'
    return ''


def list_users():
    why = _need_http()
    if why:
        return None, why
    return svc.get_json(_voice_url('/users'), timeout_s=10)


def add_user(base_weights: dict, description: str = '', name: str = '', tag: str = ''):
    why = _need_http()
    if why:
        return None, why
    return svc.post_json(_voice_url('/users'), {
        'base_weights': base_weights, 'description': description,
        'name': name, 'tag': tag}, timeout_s=10)


def delete_user(user_id: str):
    why = _need_http()
    if why:
        return None, why
    return svc.post_json(_voice_url(f'/users/{user_id}/delete'), {}, timeout_s=10)


def reset_user(user_id: str):
    why = _need_http()
    if why:
        return None, why
    return svc.post_json(_voice_url(f'/users/{user_id}/reset'), {}, timeout_s=10)

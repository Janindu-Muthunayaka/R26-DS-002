"""
TURN POLICY — what one user turn does. Spec: system/layers/TURN_CONDITIONS.xlsx.

Layer 0 says WHAT the user said (intent + modifiers); this decides WHAT TO DO
with it, using the conversation state the session holds; Layer 6 does it.

Why a separate step: on 17 Oct 2026 "වේගයෙන් කතා කරන්න" (speak fast) was
recognised as CHANGE_SPEED, the speed went up, and the sentence was then sent
to RAG as a question, which answered "not enough information". Routing was
spread over L7's route, L6's LOCAL_INTENTS and the stub's word list, and a
turn that matched none of them fell through to GENERATE. Here every turn is
classified once, in a fixed order, and only a turn that needs new words
reaches the generator.

THE RULES (sheet "Reference", G1-G10):
  G1  Only GENERATE when new words are needed (a question, a style change, a
      summary). Speed, pause, repeat, navigation and stop are LOCAL.
  G2  Never send a command phrase as the question: a style request re-asks
      the previous REAL question (or the current article if there is none).
  G3  last_spoken = the last CONTENT spoken. Confirmations, clarifying
      questions and error messages are never saved, so REPEAT never replays
      "හරි".
  G4  A speed / pause word anywhere in the sentence is applied as a modifier.
  G5  Faster pairs with fewer pauses, slower (and repeat) with more.
  G7  The stub fallback goes through exactly the same path.
  G8  No readable document: nothing goes to RAG.
  G9  A failure says so in one sentence and is not saved as last_spoken.
  G10 The session has a "current article"; repeat / next / summarise act on it.

State (core/session.SessionStore.state, one dict per job):
  turn            completed user turns (0 = only the first scan so far)
  last_spoken     {text, style, base, at, turn}   the last CONTENT (G3)
  last_question   {si, en}                        the last REAL question (G2)
  last_output     {kind, text, pending}           whatever was said last
  last_error      {si, en, article}               the question that failed
  last_style_req  {style, turn}                   for "even simpler" (F2)
  current_article int | None                      (G10)
  pending         {about: speed | read, article}  a question we asked
  delivery        {rate, pause_scale, chunk_size} last known (stub mirror, E3)
"""
from __future__ import annotations

import re
import time
from datetime import datetime

from core.schemas import Document
from layers.l0_voice import voice as l0voice
from layers.l5_assemble.payload import article_text
from layers.l6_generator import generate as l6gen
from layers.l7_personalization.intent_detection.modifiers import build_frame
from layers.l7_personalization.delivery.feedback import apply_turn, yes_no, speed_answer
from layers.l7_personalization.delivery.voice_profile import new_profile, REPEAT_MAX_AGE

# ── What is spoken (TURN_CONDITIONS.xlsx, "What to say / details") ─────────
SI_OK = 'හරි.'
SI_OK_FASTER = 'හරි, ඉක්මනින් කියන්නම්.'
SI_OK_SLOWER = 'හරි, හෙමින් කියන්නම්.'
SI_OK_FASTER_MORE = 'හරි, තවත් ඉක්මනින් කියන්නම්.'
SI_OK_SLOWER_MORE = 'හරි, තවත් හෙමින් කියන්නම්.'
SI_AT_FASTEST = 'මෙය උපරිම වේගයයි.'
SI_AT_SLOWEST = 'මෙය අඩුම වේගයයි.'
SI_OK_MORE_PAUSE = 'හරි, නවත්තමින් කියන්නම්.'
SI_OK_LESS_PAUSE = 'හරි, නොනවත්වා කියන්නම්.'
SI_OK_SMALLER = 'හරි, කෑලි කෑලි කියන්නම්.'
SI_VOLUME_UP = 'කරුණාකර දුරකථනයේ ශබ්දය වැඩි කරන්න.'
SI_VOLUME_DOWN = 'කරුණාකර දුරකථනයේ ශබ්දය අඩු කරන්න.'
SI_ASK_SPEED = 'හෙමින්ද ඉක්මනින්ද?'
SI_OFFER_READ = 'ලිපිය කියවන්නද?'
SI_STALE = 'කලින් කියපු දේ දැන් නැත. ලිපිය කියවන්නද?'
SI_NEXT_ARTICLE = 'ඊළඟ ලිපිය කියවන්නද?'
SI_SAY_AGAIN = 'කරුණාකර නැවත කියන්න.'
SI_SCAN_FIRST = 'කරුණාකර පළමුව පිටුවක් ස්කෑන් කරන්න.'

STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.",
    "StepByStep": "Break the explanation into clear numbered steps, one action or idea per step.",
}
EVEN_MORE = {   # F2: the same style asked twice in a row = "not enough"
    "Simple": " Make it even simpler and shorter than the previous answer.",
    "Detailed": " Give even more detail than the previous answer.",
    "StepByStep": " Use smaller, more detailed steps than the previous answer.",
}
REPHRASE_MOD = " Say it in different words than the previous answer."

STYLE_INTENTS = {'SIMPLIFY': 'Simple', 'SIMPLE': 'Simple', 'MAKE_SIMPLE': 'Simple',
                 'MAKE_SIMPLER': 'Simple', 'SUMMARIZE': 'Simple', 'SUMMARY': 'Simple',
                 'SHORTEN': 'Simple', 'BRIEF': 'Simple',
                 'ELABORATE': 'Detailed', 'MORE_DETAIL': 'Detailed', 'EXPAND': 'Detailed',
                 'STEP_BY_STEP': 'StepByStep'}
INTENT_ALIASES = {'GO_FORWARD': 'NEXT', 'CONTINUE': 'NEXT', 'READ_NEXT': 'NEXT',
                  'GO_BACK': 'PREVIOUS', 'BACK': 'PREVIOUS', 'GO_TO_START': 'FIRST',
                  'READ_HEADLINES': 'TITLE', 'REPLAY': 'REPEAT', 'REPEAT_AUDIO': 'REPEAT',
                  'REPEAT_LAST': 'REPEAT', 'SAY_AGAIN': 'REPEAT', 'TTS_REPLAY': 'REPEAT'}
NAV_INTENTS = {'NEXT', 'PREVIOUS', 'FIRST'}
LOCAL_READ_INTENTS = {'READ_ALOUD', 'TITLE', 'LENGTH', 'WARNINGS', 'FIRST_SCAN',
                      'GO_TO_PAGE', 'GO_TO_SECTION'} | NAV_INTENTS
QUESTION_INTENTS = {'ASK', 'ASK_QUESTION', 'IDENTIFY_CONTENT', 'EXPLAIN', 'UNKNOWN'}
QUESTION_WORDS_SI = ('මොකක්ද', 'මොනවද', 'කවුද', 'කොහෙද', 'ඇයි', 'කොහොමද', 'කීයද',
                     'කවදාද', 'කුමක්ද', 'කාටද', 'කොතැනද', 'කාගෙද', 'මොකද', '?')
QUESTION_WORDS_EN = {'what', 'who', 'whom', 'whose', 'where', 'why', 'how', 'when', 'which'}
# Intents where "ලිපිය 2" in the sentence means "that article" rather than
# being part of a question about it ("ලිපිය 2 ගැන කවුද කිව්වේ" stays a question).
ARTICLE_TEXT_INTENTS = {'READ_ALOUD', 'UNKNOWN', 'CHANGE_SPEED'} | set(STYLE_INTENTS)
ARTICLE_WORDS = [(n, w) for name, words in l0voice._COMMANDS if name.startswith('ARTICLE_')
                 for n in [int(name.split('_')[1])] for w in words]
# "Read it" words. With an article reference they mean READ THAT ARTICLE,
# whatever label L7 chose (it has used READ_ALOUD, GO_TO_PAGE, ASK_QUESTION).
READ_WORDS_SI = ('කියවන්න', 'කියවනවද', 'කියවන්නකෝ', 'කියවපන්', 'කියවලා', 'කියවල',
                 'කියවමු', 'කියවනවා')
SAY_WORDS_SI = ('කියන්න', 'කියපන්', 'කියන්නකෝ', 'කියනවද')
# "...ගැන කියන්න" = tell me ABOUT it: answer about that article, do not read it out.
ABOUT_WORDS = ('ගැන', 'පිළිබඳ', 'about')
NEXT_ARTICLE_WORDS = ('ඊළඟ ලිපිය', 'මීළඟ ලිපිය', 'ඊලඟ ලිපිය', 'මීලඟ ලිපිය', 'next article')
PREV_ARTICLE_WORDS = ('කලින් ලිපිය', 'පෙර ලිපිය', 'previous article')
LAST_ARTICLE_WORDS = ('අවසාන ලිපිය', 'අන්තිම ලිපිය', 'last article')
NUMBER_WORD_ENDINGS = ('එක', 'දෙක', 'තුන', 'හතර', 'පහ')

# Spoken lines that are not CONTENT (G3): never saved as last_spoken.
NOT_CONTENT = {l6gen.SI_END, l6gen.SI_START, l6gen.SI_NOTHING_TO_REPEAT,
               l6gen.SI_NO_TITLE, l6gen.SI_NOTHING_MISSED, l6gen.SI_NO_SERVICE,
               l6gen.SI_FAILED}
ERRORS = {l6gen.SI_NO_SERVICE, l6gen.SI_FAILED}


def _sinhala(text):
    return any('඀' <= ch <= '෿' for ch in text or '')


def _frame(si, en, intent):
    """Component 4's modifiers for this sentence. Sinhala first; typed English
    (keyboard testing) goes through the English checks."""
    if _sinhala(si):
        return build_frame(si, en if en != si else '', intent)
    return build_frame('', en or si, intent)


def _is_question(si, en):
    low = f'{si} {en}'.lower()
    if any(w in low for w in QUESTION_WORDS_SI):
        return True
    return bool(QUESTION_WORDS_EN & set(re.findall(r'[a-z]+', low)))


def _article_no(intent, text):
    m = re.match(r'ARTICLE_(\d+)$', intent)
    if m:
        return int(m.group(1))
    low = (text or '').lower()
    if intent in ('GO_TO_PAGE', 'GO_TO_SECTION'):
        # L7 labels "දෙවනි ලිපිය කියවන්න" (read the second article) GO_TO_PAGE:
        # a digit if there is one, else the article words (ordinals included).
        d = re.search(r'\d+', low)
        return int(d.group()) if d else _mentioned_article(low)
    if intent in ARTICLE_TEXT_INTENTS:
        return _mentioned_article(low)
    return None


def _mentioned_article(text):
    """The article a sentence names ("ලිපිය 2", "පළමුවන ලිපිය"), or None.
    For a question this does not change the action — it is still answered by
    RAG — it limits the evidence to that article (article_no)."""
    low = (text or '').lower()
    for n, w in ARTICLE_WORDS:
        start = low.find(w)
        while start != -1:
            if _whole(low, start, w):
                return n
            start = low.find(w, start + 1)
    return None


def _whole(low, start, w):
    """The match is the article reference itself, not part of a longer word:
    "ලිපිය 1" is not "ලිපිය 10"; "ලිපිය එක" is not "ලිපිය එකක්" (an article)
    or "ලිපිය දෙකම" (both). Case endings stay allowed: "ලිපියට", "ලිපියේ"."""
    nxt = low[start + len(w):start + len(w) + 1]
    if not nxt:
        return True
    if w[-1].isdigit() and nxt.isdigit():
        return False
    if w.endswith(NUMBER_WORD_ENDINGS) and '\u0d85' <= nxt <= '\u0dc6':
        return False                       # a consonant follows: a longer word
    return True


def _wants_read(si, en):
    """A read / say request ("කියවන්න", "කියන්න", "read"), but not "tell me ABOUT"."""
    low = f'{si} {en}'.lower()
    if any(w in low for w in ABOUT_WORDS):
        return False
    if any(w in (si or '') for w in READ_WORDS_SI + SAY_WORDS_SI):
        return True
    return bool(re.search(r'\b(read|say)\b', (en or '').lower()))


def _has_wh_word(si, en):
    """A real question word. A "?" or a polite "...පුළුවන්ද" alone is still a request."""
    low = f'{si} {en}'.lower()
    if any(w in low for w in QUESTION_WORDS_SI if w != '?'):
        return True
    return bool(QUESTION_WORDS_EN & set(re.findall(r'[a-z]+', low)))


def _title(art):
    return (art.title_polished or art.title or art.title_raw or '').strip()


def _fresh(ls, now):
    return bool(ls) and now - ls.get('at', 0) <= REPEAT_MAX_AGE.total_seconds()


# ── Called by the server ───────────────────────────────────────────────────

def context(state: dict, now: float = None) -> dict:
    """What Component 4 needs for its REPEAT rule: is there something to
    replay, and is this turn the very next one after it was spoken."""
    now = now or time.time()
    ls = state.get('last_spoken')
    if not _fresh(ls, now):
        return {'has_last': False, 'next_turn': False, 'age_s': None}
    return {'has_last': True, 'next_turn': ls.get('turn') == state.get('turn', 0),
            'age_s': round(now - ls['at'], 1)}


def seed_first_scan(state: dict, text: str, style: str = None, now: float = None) -> None:
    """The title list spoken right after a capture is the first CONTENT."""
    state.clear()
    state['turn'] = 0
    if text and text not in NOT_CONTENT:
        state['last_spoken'] = {'text': text, 'style': style, 'base': 'low',
                                'at': now or time.time(), 'turn': 0}
        state['last_output'] = {'kind': 'content', 'text': text}


def run(document: Document, voice: dict, state: dict, cursor: int = 0,
        mode: str = None, now: float = None) -> dict:
    """One turn. Returns the L6 answer dict plus: action, kind, style_class,
    prompt_modifier, question, delivery {rate, pause_scale, chunk_size},
    pause_base. An exception propagates: /ask turns it into a spoken
    sentence AND a 500 with the error, so a crash is never silent and never
    hidden."""
    now = now or time.time()
    ctx = context(state, now)
    state['turn'] = state.get('turn', 0) + 1
    t = _Turn(document, voice or {}, state, max(0, int(cursor or 0)), mode, now, ctx)
    res = t.decide()
    t.remember(res)
    return res


class _Turn:
    def __init__(self, doc, voice, st, cursor, mode, now, ctx):
        self.doc, self.voice, self.st, self.cursor = doc, voice, st, cursor
        self.mode, self.now, self.ctx = mode, now, ctx
        self.si = (voice.get('sinhala_input') or voice.get('text') or '').strip()
        self.en = (voice.get('english_translation') or '').strip()
        raw = re.sub(r'[\s\-]+', '_', str(voice.get('intent') or '').strip().upper()) or 'UNKNOWN'
        self.frame = _frame(self.si, self.en, raw)
        intent = self.frame.get('intent') or raw
        self.intent = INTENT_ALIASES.get(intent, intent)
        self.mods = self.frame.get('modifiers') or {}
        self.warnings = []
        if voice.get('warning'):
            self.warnings.append(voice['warning'])
        if str(voice.get('source', '')).startswith('stub'):
            self.warnings.append(f"voice routing: {voice['source']}")
        self.before, self.delivery = self._delivery()

    # ── delivery: rate / pause / chunk after this turn ──────────────────
    def _delivery(self):
        cur = self.st.setdefault('delivery', {'rate': 1.0, 'pause_scale': 1.0, 'chunk_size': 3})
        before = dict(cur)
        m = self.mods
        if str(self.voice.get('source', 'stub')).startswith('stub'):
            # Component 4 is not answering: apply ITS rules to the session copy
            # (E3) so the rate is kept, not reset to 1.0.
            # Every turn: one with no delivery request drifts back towards
            # normal speed (delivery/feedback.py, RELAX_FACTOR).
            prof = new_profile()
            prof.update(rate=cur['rate'], pause_scale=cur['pause_scale'],
                        chunk_size=cur['chunk_size'])
            if self.ctx['has_last']:
                at = datetime.fromtimestamp(self.now - (self.ctx['age_s'] or 0))
                prof['last_spoken'] = {'text': '', 'kind': 'answer', 'style': None,
                                       'completed': True, 'resume_at': None,
                                       'finished_at': at.isoformat(timespec='seconds'),
                                       'turns_since': 0 if self.ctx['next_turn'] else 1}
            prof, _ = apply_turn(prof, self.frame, datetime.fromtimestamp(self.now))
            cur.update(rate=prof['rate'], pause_scale=prof['pause_scale'],
                       chunk_size=prof['chunk_size'])
        else:
            b = self.voice.get('delivery_before')
            if isinstance(b, dict) and b.get('rate') is not None:
                before = {'rate': float(b['rate']),
                          'pause_scale': float(b.get('pause_scale') or 1.0),
                          'chunk_size': int(b.get('chunk_size') or 3)}
            cur.update(rate=float(self.voice.get('speed') or cur['rate']),
                       pause_scale=float(self.voice.get('pause_scale') or cur['pause_scale']),
                       chunk_size=int(self.voice.get('chunk_size') or cur['chunk_size']))
        cur['rate'] = round(max(0.8, min(1.3, float(cur['rate']))), 2)
        return before, dict(cur)

    # ── output ───────────────────────────────────────────────────────────
    def out(self, action, text, kind, ok=True, route='LOCAL', style=None, base='none',
            cursor=None, res=None, pending=None, warnings=(), question=None,
            prompt_modifier=None):
        r = dict(res or {})
        r.update({
            'ok': bool(r.get('ok', ok)) if res else bool(ok),
            'route': route, 'intent': self.intent,
            'speakable': text or '', 'answer_si': r.get('answer_si', '') if res else '',
            'sources': r.get('sources', []),
            'warnings': self.warnings + list(r.get('warnings', [])) + list(warnings),
            'cursor': cursor if cursor is not None else (
                r['cursor'] if res and r.get('cursor') is not None else self.cursor),
            'action': action, 'kind': kind,
            'style_class': style or self.voice.get('style_class') or '',
            'prompt_modifier': prompt_modifier if prompt_modifier is not None
                               else self.voice.get('prompt_modifier', ''),
            'question': question, 'delivery': self.delivery,
            'pause_base': base, 'pending': pending,
        })
        # de-duplicate warnings, keep order
        r['warnings'] = list(dict.fromkeys(r['warnings']))
        return r

    def remember(self, res):
        st, kind = self.st, res.get('kind')
        if res.get('pending'):
            st['pending'] = res['pending']
        st['last_output'] = {'kind': kind, 'text': res.get('speakable', ''),
                             'pending': res.get('pending')}
        if kind == 'content' and res.get('speakable'):
            text = res.get('replay_text') or res['speakable']
            st['last_spoken'] = {'text': text, 'style': res.get('style_class'),
                                 'base': res.get('pause_base'), 'at': self.now,
                                 'turn': st['turn']}

    # ── the decision, in a fixed order ───────────────────────────────────
    def decide(self):
        m, intent, st = self.mods, self.intent, self.st

        if not any(article_text(a) for a in self.doc.articles):            # G8 / E4
            return self.out('MSG', SI_SCAN_FIRST, 'error', ok=False)

        if intent == 'STOP':                                               # N10
            return self.out('STOP', '', 'stop')

        # Nothing was said (an empty transcript). Never a question: it was
        # stored as the "last question", and every later "say it simpler"
        # re-asked an empty question that RAG rejected.
        if not (self.si or '').strip() and not (self.en or '').strip():
            return self.out('ASK', SI_SAY_AGAIN, 'ask', pending=None)

        pending = st.pop('pending', None)
        if pending and pending.get('about') == 'read':                     # R3/R4/N5 answer
            yn = yes_no(self.si, self.en)
            if intent == 'NEXT' and not m.get('style'):   # "ඊළඟ" to "ඊළඟ ලිපිය කියවන්නද?"
                yn = 'yes'
            if yn == 'yes':
                return self._read_article(pending.get('article') or 0)
            if yn == 'no':
                return self.out('CONFIRM', SI_OK, 'confirm')
        if pending and pending.get('about') == 'speed' and not m.get('speed'):
            sp = speed_answer(self.si, self.en)                            # D4
            if sp:
                self.mods = m = {**m, 'speed': sp}

        style = m.get('style') or STYLE_INTENTS.get(intent)
        summary = bool(m.get('summary')) or intent in ('SUMMARIZE', 'SUMMARY')
        if summary:
            style = style or 'Simple'
        art_no = _article_no(intent, self.si)
        # Which article the WORDS name (any label): "දෙවනි ලිපිය", "ලිපිය 2",
        # "second article", or "ඊළඟ / කලින් / අන්තිම ලිපිය" (next / previous / last).
        ref = (_mentioned_article(self.si) or _mentioned_article(self.en)
               or self._relative_article())
        wants_read = _wants_read(self.si, self.en)
        wh = _has_wh_word(self.si, self.en)
        if art_no is None and ref is not None and (
                wants_read or style or intent in NAV_INTENTS or intent in ARTICLE_TEXT_INTENTS
                or intent in ('GO_TO_PAGE', 'GO_TO_SECTION')):
            art_no = ref
        n_words = len(self.si.split())
        delivery_words = bool(m.get('speed') or m.get('pause') or m.get('chunk'))
        question = _is_question(self.si, self.en)

        # Volume is a phone setting (D11).
        if m.get('volume') and not (style or art_no or intent in NAV_INTENTS or question):
            down = bool(re.search(r'අඩු|quieter|lower', f'{self.si} {self.en}'.lower()))
            return self.out('CONFIRM', SI_VOLUME_DOWN if down else SI_VOLUME_UP, 'confirm')

        # New words needed: style / summary / rephrase / bare "explain" (C1-C6, X3-X20).
        if style or intent == 'REPHRASE' or (intent == 'EXPLAIN' and n_words <= 2):
            return self._restyle(style, summary, art_no)

        # An article + a read word = read THAT article, whatever the label
        # (READ_ALOUD, GO_TO_PAGE, ASK_QUESTION, REPEAT, NEXT ...). Only a real
        # question word makes it a question about the article instead.
        if art_no is not None and wants_read and not wh:
            return self._read_article(art_no - 1)

        # Repeat (R1-R7, X1-X2).
        if m.get('repeat') or intent == 'REPEAT':
            return self._repeat()

        # Delivery only: change how the last answer sounds (D1-D10).
        reads_article = 'ලිපිය' in self.si or 'article' in self.si.lower()
        delivery_only = intent in ('CHANGE_SPEED', 'CHANGE_DELIVERY', 'VOLUME') or (
            delivery_words and (
                intent in ('UNKNOWN', 'CHANGE_SPEED')
                or (intent == 'READ_ALOUD' and not reads_article)
                or (intent in QUESTION_INTENTS and n_words <= 4 and not question)))
        if delivery_only:
            return self._delivery_only()

        # Reading and navigation, with any speed word applied (N1-N11, X16-X17).
        if art_no is not None:
            if question and 1 <= art_no <= len(self.doc.articles):
                # "ලිපිය 2 ගැන කවුද කිව්වේ": a question ABOUT article 2, not "read it".
                return self._generate(self.si, self.en or self.si, style=None,
                                      real_question=True, article=art_no - 1)
            return self._read_article(art_no - 1)
        if intent in LOCAL_READ_INTENTS:
            return self._local(intent)

        # Unclear and short (F7): ask again rather than send noise to RAG.
        if intent == 'UNKNOWN' and n_words <= 3 and not question:
            return self.out('ASK', SI_SAY_AGAIN, 'ask', pending=None)

        # A real question (C7, C8). If it names an article, RAG uses only that one.
        named = _mentioned_article(self.si) or _mentioned_article(self.en)
        if named is not None and not 1 <= named <= len(self.doc.articles):
            named = None
        return self._generate(self.si, self.en or self.si, style=None, real_question=True,
                              article=None if named is None else named - 1)

    # ── branches ─────────────────────────────────────────────────────────
    def _delivery_only(self):
        m = self.mods
        if not (m.get('speed') or m.get('pause') or m.get('chunk')):        # D3
            return self.out('ASK', SI_ASK_SPEED, 'ask', pending={'about': 'speed'})
        msgs = []
        b, a = self.before, self.delivery
        if m.get('speed') == 'faster':
            msgs.append(SI_AT_FASTEST if b['rate'] >= 1.3 - 1e-6 else
                        SI_OK_FASTER_MORE if 'තව' in self.si else SI_OK_FASTER)
        elif m.get('speed') == 'slower':
            msgs.append(SI_AT_SLOWEST if b['rate'] <= 0.8 + 1e-6 else
                        SI_OK_SLOWER_MORE if 'තව' in self.si else SI_OK_SLOWER)
        if m.get('pause') == 'more':
            msgs.append(SI_OK_MORE_PAUSE)
        elif m.get('pause') == 'less':
            msgs.append(SI_OK_LESS_PAUSE)
        if m.get('chunk') == 'smaller':
            msgs.append(SI_OK_SMALLER)
        confirm = ' '.join(msgs) or SI_OK
        ls = self.st.get('last_spoken')
        if not _fresh(ls, self.now):
            return self.out('CONFIRM', confirm, 'confirm')
        return self.out('CONFIRM + REPLAY', f"{confirm} {ls['text']}", 'content',
                        style=ls.get('style'), base=ls.get('base') or 'low',
                        res={'replay_text': ls['text']})

    def _repeat(self):
        st, lo = self.st, self.st.get('last_output') or {}
        if lo.get('kind') == 'ask' and lo.get('pending'):                  # R6: the question again
            return self.out('ASK', lo['text'], 'ask', pending=lo['pending'])
        if lo.get('kind') == 'error' and st.get('last_error'):             # R7: retry once
            e = st['last_error']
            return self._generate(e['si'], e['en'], style=e.get('style'),
                                  article=e.get('article'), real_question=e.get('real', True))
        ls = st.get('last_spoken')
        if _fresh(ls, self.now):                                           # R1 / R2 / R5
            return self.out('REPLAY', ls['text'], 'content', style=ls.get('style'),
                            base=ls.get('base') or 'low')
        art = st.get('current_article') or 0
        if ls:                                                             # R3: stale
            return self.out('ASK', SI_STALE, 'ask', pending={'about': 'read', 'article': art})
        return self.out('MSG', f'{l6gen.SI_NOTHING_TO_REPEAT} {SI_OFFER_READ}', 'ask',
                        pending={'about': 'read', 'article': art})         # R4

    def _relative_article(self):
        """1-based article for "next / previous / last article", or None."""
        low = f'{self.si} {self.en}'.lower()
        cur = self.st.get('current_article')
        n = len(self.doc.articles)
        if any(w in low for w in NEXT_ARTICLE_WORDS):
            return (cur + 1 if cur is not None else 0) + 1
        if any(w in low for w in PREV_ARTICLE_WORDS):
            return max(1, cur if cur is not None else 1)
        if any(w in low for w in LAST_ARTICLE_WORDS):
            return n or None
        return None

    def _local_answer(self, doc, intent, cursor):
        v = {**self.voice, 'intent': intent, 'route': 'LOCAL', 'warning': None, 'source': 'policy'}
        return l6gen.answer(doc, v, mode=self.mode, cursor=cursor)

    def _sub(self, idx):
        """The current article alone, so parts / length / read act on it (G10)."""
        idx = idx if idx is not None and 0 <= idx < len(self.doc.articles) else 0
        return idx, Document(articles=[self.doc.articles[idx]], warnings=self.doc.warnings)

    def _read_article(self, idx):
        n = idx + 1
        res = self._local_answer(self.doc, f'ARTICLE_{n}', 0)
        if res.get('ok') and 0 <= idx < len(self.doc.articles):
            self.st['current_article'] = idx
            # The whole article was just read: "next" goes on to the next
            # article, "previous" back to its last part.
            done = len(l6gen._parts(self._sub(idx)[1]))
            return self.out('READ', res['speakable'], 'content', res=res, cursor=done, base='low')
        return self.out('MSG', res['speakable'], 'msg', ok=False, res=res)   # E5

    def _local(self, intent):
        st = self.st
        if intent in ('GO_TO_PAGE', 'GO_TO_SECTION'):                     # N11, no number
            intent = 'FIRST_SCAN'
        if intent == 'READ_ALOUD':                                         # N1
            return self._read_article(st.get('current_article') or 0)
        if intent in NAV_INTENTS or intent == 'LENGTH':                    # N5-N8
            idx, sub = self._sub(st.get('current_article'))
            st['current_article'] = idx
            res = self._local_answer(sub, intent, self.cursor)
            text = res['speakable']
            if intent == 'LENGTH':
                return self.out('MSG', text, 'msg', res=res)
            if text == l6gen.SI_END and idx + 1 < len(self.doc.articles):
                return self.out('NAV', f'{text} {SI_NEXT_ARTICLE}', 'ask', res=res,
                                pending={'about': 'read', 'article': idx + 1})
            kind = 'content' if text not in NOT_CONTENT else 'msg'
            return self.out('NAV', text, kind, res=res, base='low')
        res = self._local_answer(self.doc, intent, self.cursor)            # TITLE / WARNINGS / FIRST_SCAN
        kind = 'content' if intent in ('TITLE', 'FIRST_SCAN') and res['speakable'] not in NOT_CONTENT \
            else 'msg'
        return self.out('READ' if kind == 'content' else 'MSG', res['speakable'], kind,
                        res=res, base='low' if kind == 'content' else 'none')

    def _default_question(self, summary, idx=None):
        """No previous real question: act on the current article (G2, G10)."""
        idx = self.st.get('current_article') if idx is None else idx
        if idx is not None and 0 <= idx < len(self.doc.articles):
            n, t = idx + 1, _title(self.doc.articles[idx])
            named = f" titled '{t}'" if t else ''
            if summary:
                return f'ලිපිය {n} සාරාංශ කරන්න.', f'Summarize article {n}{named}.', idx
            return f'ලිපිය {n} පැහැදිලි කරන්න.', f'Explain article {n}{named}.', idx
        if summary:
            return 'මෙම පිටුවේ ලිපි සාරාංශ කරන්න.', 'Summarize the articles on this page.', None
        return 'මෙම පිටුවේ ලිපි පැහැදිලි කරන්න.', 'Explain the articles on this page.', None

    def _restyle(self, style, summary, art_no):
        st = self.st
        extra = ''
        if self.intent == 'REPHRASE':
            extra += REPHRASE_MOD
        if style and (st.get('last_style_req') or {}).get('style') == style \
                and st['last_style_req'].get('turn') == st['turn'] - 1:
            extra += EVEN_MORE[style]                                       # F2
        if style:
            st['last_style_req'] = {'style': style, 'turn': st['turn']}

        if art_no is not None:                                              # X18 / X19
            idx = art_no - 1
            if not 0 <= idx < len(self.doc.articles):
                return self._read_article(idx)                              # E5 message
            si, en, idx = self._default_question(summary, idx)
            return self._generate(si, en, style, extra, article=idx)
        if self.intent == 'NEXT':                                           # X20
            idx, sub = self._sub(st.get('current_article'))
            st['current_article'] = idx
            nxt = self._local_answer(sub, 'NEXT', self.cursor)
            if nxt['speakable'] in NOT_CONTENT:
                return self.out('NAV', nxt['speakable'], 'msg', res=nxt)
            part = nxt['speakable']
            return self._generate(f'මෙම කොටස පැහැදිලි කරන්න: {part}',
                                  f'Explain this part of the article: {part}',
                                  style, extra, article=idx, cursor=nxt['cursor'])
        lq = st.get('last_question')
        if lq and not (lq.get('si') or lq.get('en') or '').strip():
            lq = None
        if summary or not lq:
            si, en, idx = self._default_question(summary)
            return self._generate(si, en, style, extra, article=idx)
        return self._generate(lq['si'], lq['en'], style, extra,             # C1/C3/C4/C5, X3-X15
                              article=lq.get('article'))

    def _generate(self, si, en, style=None, extra='', article=None, cursor=None,
                  real_question=False):
        st = self.st
        style_eff = style or self.voice.get('style_class') or 'Detailed'
        if style_eff not in STYLE_PROMPT_MODIFIERS:
            style_eff = 'Detailed'
        modifier = (STYLE_PROMPT_MODIFIERS[style_eff] if style
                    else self.voice.get('prompt_modifier') or STYLE_PROMPT_MODIFIERS[style_eff]) + extra
        # intent ASK_QUESTION, route GENERATE: an ARTICLE_n / NEXT turn that is
        # restyled must not be answered locally by Layer 6.
        v = {**self.voice, 'route': 'GENERATE', 'intent': 'ASK_QUESTION',
             'sinhala_input': si, 'english_translation': en,
             'style_class': style_eff, 'prompt_modifier': modifier,
             'warning': None, 'source': 'policy',
             # 1-based; RAG then uses only that article's text as evidence.
             'article_no': None if article is None else article + 1}
        res = l6gen.answer(self.doc, v, mode=self.mode,
                           cursor=self.cursor if cursor is None else cursor)
        text = res.get('speakable') or ''
        q = {'si': si, 'en': en, 'article': article}
        if not text or text in ERRORS or res.get('ok') is False:          # E1 / E2
            st['last_error'] = {**q, 'style': style, 'real': real_question}
            return self.out('MSG', text or l6gen.SI_FAILED, 'error', ok=False, res=res,
                            route='GENERATE', style=style_eff, question=q,
                            prompt_modifier=modifier)
        st.pop('last_error', None)
        if real_question:
            st['last_question'] = q
        if article is not None:
            st['current_article'] = article
        out = self.out('REGEN', text, 'content', res=res, route='GENERATE',
                       style=style_eff, base=None, question=q, prompt_modifier=modifier,
                       cursor=res.get('cursor') if cursor is None else cursor)
        out['style_forced'] = bool(style)
        return out

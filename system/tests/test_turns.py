"""The turn policy (layers/l6_generator/turns.py) against TURN_CONDITIONS.xlsx.

THE BUG THIS FILE EXISTS FOR (17 Oct 2026): "වේගයෙන් කතා කරන්න" (speak fast)
was recognised as CHANGE_SPEED, the speed went up, and the sentence was then
sent to RAG as a question. RAG answered "කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට
ප්‍රමාණවත් තොරතුරු නොමැත." Every test that names a sheet row checks the
same thing first: what, if anything, reached the generator.

The generator is faked: it records the question it was given and echoes it,
so "what was asked" is visible in the answer.
"""
import pytest

from core.schemas import Article, Box, Document
from layers.l0_voice.voice import _stub
from layers.l6_generator import generate as l6gen
from layers.l6_generator import turns

BOX = Box(x1=0, y1=0, x2=10, y2=10)
NO_INFO = 'කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත.'


def _art(i, title, body):
    return Article(index=i, box=BOX, body_raw=body, body=body, title=title, verdict='ok')


def _doc():
    long = ' '.join(f'පළමු ලිපියේ {i} වන වාක්‍යයයි එය තරමක් දිගු වේ.' for i in range(30))
    return Document(articles=[_art(0, 'බෝක්කුව කැඩිලා', long),
                              _art(1, 'විකිරණ යන්ත්‍රය ක්‍රියා විරහිතයි', 'දෙවන ලිපියේ පෙළ මෙයයි.')])


class Chat:
    """One session: the stub for L0, a recording fake for RAG."""

    def __init__(self, monkeypatch, rag_reply=None):
        self.doc, self.state, self.cursor, self.asked = _doc(), {}, 0, []
        real = l6gen.answer

        def fake(doc, voice, mode=None, cursor=0):
            if str(voice.get('route')).upper() == 'GENERATE' \
                    and str(voice.get('intent')).upper() not in l6gen.LOCAL_INTENTS:
                self.asked.append(voice)
                text = rag_reply(voice) if rag_reply else f"[RAG {voice['style_class']}] {voice['english_translation']}"
                return l6gen._out(text not in (l6gen.SI_FAILED, l6gen.SI_NO_SERVICE), 'GENERATE',
                                  voice['intent'], text, answer_si=text, cursor=cursor)
            return real(doc, voice, mode='off', cursor=cursor)

        monkeypatch.setattr(l6gen, 'answer', fake)
        first = real(self.doc, {'intent': 'FIRST_SCAN', 'route': 'GENERATE'}, mode='off')
        self.titles = first['speakable']
        turns.seed_first_scan(self.state, self.titles)

    def say(self, text):
        self.asked.clear()
        r = turns.run(self.doc, _stub(text), self.state, cursor=self.cursor)
        self.cursor = r['cursor']
        return r


# ── 1. Delivery: never a question (D1-D11) ────────────────────────────────
@pytest.mark.parametrize('text', ['වේගයෙන් කතා කරන්න', 'ඉක්මනින් කියන්න', 'හෙමින් කියන්න',
                                  'වේගය වෙනස් කරන්න', 'නවත්තලා නවත්තලා කියන්න',
                                  'නොනවත්වා කියන්න', 'කෑලි කෑලි කියන්න', 'ශබ්දය වැඩි කරන්න',
                                  'speak faster', 'speak slower'])
def test_a_delivery_request_never_reaches_the_generator(monkeypatch, text):
    c = Chat(monkeypatch)
    c.say('කවුද බෝක්කුව හැදුවේ')
    r = c.say(text)
    assert c.asked == [], f'{text!r} was sent to RAG as a question'
    assert r['route'] == 'LOCAL'
    assert NO_INFO not in r['speakable']


def test_d1_speak_fast_confirms_replays_faster_and_with_fewer_pauses(monkeypatch):
    c = Chat(monkeypatch)
    answer = c.say('කවුද බෝක්කුව හැදුවේ')['speakable']
    r = c.say('වේගයෙන් කතා කරන්න')
    assert r['action'] == 'CONFIRM + REPLAY'
    assert r['speakable'].startswith(turns.SI_OK_FASTER) and answer in r['speakable']
    assert r['delivery']['rate'] > 1.0 and r['delivery']['pause_scale'] < 1.0


def test_d2_speak_slower_confirms_replays_slower_and_with_more_pauses(monkeypatch):
    c = Chat(monkeypatch)
    r = c.say('හෙමින් කියන්න')
    assert r['speakable'].startswith(turns.SI_OK_SLOWER) and c.titles in r['speakable']
    assert r['delivery']['rate'] < 1.0 and r['delivery']['pause_scale'] > 1.0


def test_d3_d4_change_speed_with_no_direction_asks_then_obeys(monkeypatch):
    c = Chat(monkeypatch)
    r = c.say('වේගය වෙනස් කරන්න')
    assert r['action'] == 'ASK' and r['speakable'] == turns.SI_ASK_SPEED
    assert r['delivery']['rate'] == 1.0
    r = c.say('හෙමින්')
    assert r['action'] == 'CONFIRM + REPLAY' and r['delivery']['rate'] < 1.0
    assert c.titles in r['speakable'], 'the question itself is never replayed (G3)'


def test_d6_d7_at_the_limit_says_so(monkeypatch):
    c = Chat(monkeypatch)
    for _ in range(8):
        r = c.say('හෙමින් කියන්න')
    assert r['delivery']['rate'] == 0.8 and r['speakable'].startswith(turns.SI_AT_SLOWEST)
    for _ in range(12):
        r = c.say('ඉක්මනින් කියන්න')
    assert r['delivery']['rate'] == 1.3 and r['speakable'].startswith(turns.SI_AT_FASTEST)


def test_d8_d9_pause_requests_move_the_pause_and_not_the_rate(monkeypatch):
    c = Chat(monkeypatch)
    more = c.say('නවත්තලා නවත්තලා කියන්න')
    assert more['delivery']['rate'] == 1.0 and more['delivery']['pause_scale'] > 1.0
    less = c.say('නොනවත්වා කියන්න')
    assert less['delivery']['pause_scale'] < more['delivery']['pause_scale']


def test_d11_volume_points_to_the_phone(monkeypatch):
    r = Chat(monkeypatch).say('ශබ්දය වැඩි කරන්න')
    assert r['speakable'] == turns.SI_VOLUME_UP and r['action'] == 'CONFIRM'


# ── 2. Repeat (R1-R7) ──────────────────────────────────────────────────────
def test_r1_repeat_replays_the_last_answer_not_the_article(monkeypatch):
    c = Chat(monkeypatch)
    answer = c.say('කවුද බෝක්කුව හැදුවේ')['speakable']
    r = c.say('නැවත කියන්න')
    assert c.asked == []
    assert r['action'] == 'REPLAY' and r['speakable'] == answer
    assert r['delivery']['rate'] < 1.0 and r['delivery']['pause_scale'] > 1.0


def test_r2_a_late_repeat_replays_without_learning(monkeypatch):
    c = Chat(monkeypatch)
    answer = c.say('කවුද බෝක්කුව හැදුවේ')['speakable']
    c.say('වචන කීයද')                         # a message, not content (G3)
    r = c.say('නැවත කියන්න')
    assert r['speakable'] == answer
    assert r['delivery']['rate'] == 1.0


def test_r3_r4_nothing_to_repeat_offers_to_read(monkeypatch):
    c = Chat(monkeypatch)
    c.state.pop('last_spoken')
    c.state['last_output'] = {}
    r = c.say('නැවත කියන්න')
    assert turns.SI_OFFER_READ in r['speakable']
    r = c.say('ඔව්')
    assert r['action'] == 'READ' and 'පළමු ලිපියේ' in r['speakable']


def test_r6_repeat_after_a_question_repeats_the_question(monkeypatch):
    c = Chat(monkeypatch)
    c.say('වේගය වෙනස් කරන්න')
    r = c.say('ආයෙ කියන්න')
    assert r['speakable'] == turns.SI_ASK_SPEED and c.state['pending'] == {'about': 'speed'}


def test_r7_repeat_after_a_failure_retries_the_question(monkeypatch):
    replies = iter([l6gen.SI_FAILED, 'දැන් පිළිතුර.'])
    c = Chat(monkeypatch, rag_reply=lambda v: next(replies))
    assert c.say('කවුද බෝක්කුව හැදුවේ')['kind'] == 'error'
    r = c.say('නැවත කියන්න')
    assert r['speakable'] == 'දැන් පිළිතුර.'
    assert c.asked[0]['english_translation'] == 'කවුද බෝක්කුව හැදුවේ'


def test_an_error_is_never_saved_as_the_thing_to_repeat(monkeypatch):
    c = Chat(monkeypatch, rag_reply=lambda v: l6gen.SI_FAILED)
    c.say('කවුද බෝක්කුව හැදුවේ')
    assert c.state['last_spoken']['text'] == c.titles


# ── 3. Style: re-ask the previous REAL question (G2) ──────────────────────
@pytest.mark.parametrize('text,style', [('සරලව කියන්න', 'Simple'),
                                        ('විස්තරාත්මකව කියන්න', 'Detailed'),
                                        ('පියවරෙන් පියවර කියන්න', 'StepByStep'),
                                        ('ආයෙ සරලව කියන්න', 'Simple')])
def test_a_style_request_re_asks_the_previous_question(monkeypatch, text, style):
    c = Chat(monkeypatch)
    c.say('කවුද බෝක්කුව හැදුවේ')
    r = c.say(text)
    assert len(c.asked) == 1
    assert c.asked[0]['english_translation'] == 'කවුද බෝක්කුව හැදුවේ', \
        'the command phrase itself must never be the question'
    assert c.asked[0]['style_class'] == style and r['style_class'] == style


def test_a_style_request_with_no_question_yet_explains_the_page(monkeypatch):
    c = Chat(monkeypatch)
    c.say('සරලව කියන්න')
    assert c.asked[0]['english_translation'] == 'Explain the articles on this page.'


def test_f2_the_same_style_twice_asks_for_even_more(monkeypatch):
    c = Chat(monkeypatch)
    c.say('කවුද බෝක්කුව හැදුවේ')
    c.say('සරලව කියන්න')
    c.say('සරලව කියන්න')
    assert turns.EVEN_MORE['Simple'] in c.asked[0]['prompt_modifier']


def test_x3_slower_and_simpler_does_both(monkeypatch):
    c = Chat(monkeypatch)
    c.say('කවුද බෝක්කුව හැදුවේ')
    r = c.say('හෙමින් සරලව කියන්න')
    assert c.asked[0]['style_class'] == 'Simple' and r['delivery']['rate'] < 1.0


def test_x18_x19_an_article_with_a_style_is_generated_about_that_article(monkeypatch):
    c = Chat(monkeypatch)
    c.say('ලිපිය 2 සරලව කියන්න')
    assert c.asked[0]['english_translation'].startswith('Explain article 2')
    assert c.asked[0]['intent'] == 'ASK_QUESTION', 'ARTICLE_2 would be read locally by L6'
    c.say('ලිපිය 1 සාරාංශය')
    assert c.asked[0]['english_translation'].startswith('Summarize article 1')
    assert c.state['current_article'] == 0


# ── 4. Reading and navigation (G10) ────────────────────────────────────────
def test_x16_read_an_article_slowly(monkeypatch):
    c = Chat(monkeypatch)
    r = c.say('හෙමින් ලිපිය 2 කියවන්න')
    assert c.asked == [] and r['action'] == 'READ'
    assert 'දෙවන ලිපියේ' in r['speakable'] and r['delivery']['rate'] < 1.0


def test_next_walks_the_current_article_then_offers_the_next(monkeypatch):
    c = Chat(monkeypatch)
    n = len(l6gen._parts(Document(articles=[c.doc.articles[0]])))
    assert n >= 3, 'the fixture must be long enough to actually walk'
    seen = [c.say('ඊළඟ')['speakable'] for _ in range(n)]
    assert len(set(seen)) == n and all('පළමු ලිපියේ' in s for s in seen)
    r = c.say('ඊළඟ')
    assert r['speakable'] == f'{l6gen.SI_END} {turns.SI_NEXT_ARTICLE}'
    r = c.say('ඊළඟ')                        # "next" answers "read the next one?"
    assert 'දෙවන ලිපියේ' in r['speakable'] and c.state['current_article'] == 1


def test_titles_are_numbered(monkeypatch):
    r = Chat(monkeypatch).say('ශීර්ෂය මොකක්ද')
    assert 'ලිපිය 1: බෝක්කුව කැඩිලා' in r['speakable']


def test_stop_keeps_what_to_repeat(monkeypatch):
    c = Chat(monkeypatch)
    answer = c.say('කවුද බෝක්කුව හැදුවේ')['speakable']
    assert c.say('නවත්වන්න')['speakable'] == ''
    assert c.say('නැවත කියන්න')['speakable'] == answer


# ── 5. The L6 safety net and the L7 rules ──────────────────────────────────
def test_l6_answers_a_delivery_intent_locally_even_without_the_policy():
    v = {'route': 'GENERATE', 'intent': 'CHANGE_SPEED', 'english_translation': 'Speak fast'}
    r = l6gen.answer(_doc(), v, mode='http')
    assert r['route'] == 'LOCAL' and r['speakable'] == l6gen.SI_OK


def test_l7_faster_pairs_with_fewer_pauses_and_slower_with_more():
    from layers.l7_personalization.delivery.feedback import apply_turn
    from layers.l7_personalization.delivery.voice_profile import new_profile
    from layers.l7_personalization.intent_detection.modifiers import build_frame
    fast, _ = apply_turn(new_profile(), build_frame('වේගයෙන් කතා කරන්න', '', 'CHANGE_SPEED'), None)
    slow, _ = apply_turn(new_profile(), build_frame('හෙමින් කියන්න', '', 'CHANGE_SPEED'), None)
    assert fast['rate'] > 1.0 and fast['pause_scale'] < 1.0
    assert slow['rate'] < 1.0 and slow['pause_scale'] > 1.0


def test_the_pause_scale_reaches_the_spoken_text():
    from layers.l7_personalization.delivery.planner import format_text_with_tts_pauses, pause_level
    text = 'පළමු වාක්‍යය. දෙවන වාක්‍යය?'
    assert pause_level('Detailed', 1.0) == 'low' and pause_level('Simple', 1.0) == 'mid'
    assert pause_level('StepByStep', 1.0) == 'mid'
    assert pause_level('Detailed', 0.7) == 'none' and pause_level('Detailed', 1.6) == 'high'
    assert '...' in format_text_with_tts_pauses(text, 'Detailed', 1.6)
    assert format_text_with_tts_pauses(text, level='none').endswith('?'), 'a question keeps its "?"'


# ── 6. Article numbers (17 Oct 2026) ───────────────────────────────────────
@pytest.mark.parametrize('text', ['පළමුවන ලිපිය කියවන්න', 'පළමුවැනි ලිපිය කියවන්න',
                                  'ලිපිය එක කියවන්න', 'read the first article'])
def test_every_way_of_saying_article_one_reads_only_article_one(monkeypatch, text):
    c = Chat(monkeypatch)
    r = c.say(text)
    assert c.asked == [] and r['action'] == 'READ'
    assert 'පළමු ලිපියේ' in r['speakable'] and 'දෙවන ලිපියේ' not in r['speakable']


def test_an_ordinal_without_article_is_still_a_question(monkeypatch):
    c = Chat(monkeypatch)
    c.say('දෙවැනි තැන කාටද')
    assert len(c.asked) == 1 and c.asked[0]['article_no'] is None


def test_a_request_about_one_article_tells_rag_which_one(monkeypatch):
    c = Chat(monkeypatch)
    c.say('ලිපිය 2 සරලව කියන්න')
    assert c.asked[0]['article_no'] == 2
    c.say('ලිපිය 2 ගැන කවුද කිව්වේ')            # a question that names an article
    assert c.asked[0]['article_no'] == 2
    c.say('සරලව කියන්න')                         # re-ask keeps the article
    assert c.asked[0]['article_no'] == 2
    c.say('කවුද බෝක්කුව හැදුවේ')                # a page-wide question is not limited
    assert c.asked[0]['article_no'] is None


def test_in_process_rag_gets_only_the_named_article(monkeypatch):
    from layers.l6_generator import pipeline as l6pipe
    seen = {}

    def fake_run(**kw):
        seen['text'] = kw['raw_ocr_input']['corrected_text']
        return {'ok': True, 'answer_si': 'පිළිතුර.', 'speakable_text': 'පිළිතුර.'}
    monkeypatch.setattr(l6pipe, 'run_pipeline', fake_run)
    v = {'route': 'GENERATE', 'intent': 'ASK_QUESTION', 'english_translation': 'x',
         'article_no': 2}
    l6gen.answer(_doc(), v, mode='local')
    assert seen['text'].startswith('Article 2: විකිරණ') and 'පළමු ලිපියේ' not in seen['text']


@pytest.mark.parametrize('text', ['දෙවනි ලිපිය කියවන්න', 'දෙවන ලිපිය කියවන්න', 'ලිපිය දෙක කියවන්න'])
def test_l7_go_to_page_with_an_article_ordinal_reads_that_article(monkeypatch, text):
    """17 Oct 2026: L7 labelled "දෙවනි ලිපිය කියවන්න" GO_TO_PAGE; with no
    digit the page overview (both titles) was read instead of article 2."""
    c = Chat(monkeypatch)
    v = {'intent': 'GO_TO_PAGE', 'route': 'LOCAL', 'sinhala_input': text,
         'english_translation': 'Read the second article', 'source': 'component4',
         'style_class': 'Detailed', 'prompt_modifier': '', 'speed': 1.0}
    r = turns.run(c.doc, v, c.state, cursor=0)
    assert c.asked == [] and r['action'] == 'READ'
    assert 'දෙවන ලිපියේ' in r['speakable'] and 'පළමු ලිපියේ' not in r['speakable']


# ── 7. Article requests survive any L7 label (17 Oct 2026) ─────────────────
def _l7(c, text, intent, en='Read the second article'):
    v = {'intent': intent, 'route': 'GENERATE', 'sinhala_input': text, 'english_translation': en,
         'source': 'component4', 'style_class': 'Detailed', 'prompt_modifier': '', 'speed': 1.0}
    c.asked.clear()
    r = turns.run(c.doc, v, c.state, cursor=c.cursor)
    c.cursor = r['cursor']
    return r


ARTICLE_2_WAYS = ['දෙවනි ලිපිය කියවන්න', 'දෙවන ලිපිය කියවන්න පුළුවන්ද?', 'දෙවැනි ලිපිය කියවන්න',
                  'ලිපිය දෙක කියවන්න', 'ලිපිය 2 කියවන්න', 'දෙවන ලිපිය කියන්න', 'ලිපිය අංක 2 කියවන්න']
ANY_LABEL = ['READ_ALOUD', 'GO_TO_PAGE', 'ASK_QUESTION', 'IDENTIFY_CONTENT', 'UNKNOWN',
             'REPEAT', 'NEXT', 'ASK', 'EXPLAIN']


@pytest.mark.parametrize('text', ARTICLE_2_WAYS)
@pytest.mark.parametrize('label', ANY_LABEL)
def test_reading_article_two_works_whatever_l7_calls_it(monkeypatch, text, label):
    c = Chat(monkeypatch)
    r = _l7(c, text, label)
    assert c.asked == [], f'{label}: sent to RAG'
    assert r['action'] == 'READ'
    assert 'දෙවන ලිපියේ' in r['speakable'] and 'පළමු ලිපියේ' not in r['speakable']


def test_a_question_about_an_article_is_answered_not_read(monkeypatch):
    c = Chat(monkeypatch)
    _l7(c, 'දෙවන ලිපියේ කවුද කතා කරන්නේ', 'READ_ALOUD', en='Who is speaking in the second article')
    assert len(c.asked) == 1 and c.asked[0]['article_no'] == 2


def test_tell_me_about_an_article_is_answered_not_read(monkeypatch):
    c = Chat(monkeypatch)
    _l7(c, 'ලිපිය 2 ගැන කියන්න', 'ASK_QUESTION', en='Tell me about article 2')
    assert len(c.asked) == 1 and c.asked[0]['article_no'] == 2


def test_a_style_on_an_article_works_whatever_the_label(monkeypatch):
    c = Chat(monkeypatch)
    _l7(c, 'දෙවනි ලිපිය සරලව කියන්න', 'ASK_QUESTION', en='Say the second article simply')
    assert c.asked[0]['article_no'] == 2 and c.asked[0]['style_class'] == 'Simple'


def test_next_previous_and_last_article(monkeypatch):
    c = Chat(monkeypatch)
    _l7(c, 'පළමුවන ලිපිය කියවන්න', 'READ_ALOUD', en='Read the first article')
    r = _l7(c, 'ඊළඟ ලිපිය කියවන්න', 'NEXT', en='Read the next article')
    assert 'දෙවන ලිපියේ' in r['speakable'], 'next ARTICLE, not the next part'
    r = _l7(c, 'කලින් ලිපිය', 'PREVIOUS', en='Previous article')
    assert 'පළමු ලිපියේ' in r['speakable']
    r = _l7(c, 'අන්තිම ලිපිය කියවන්න', 'READ_ALOUD', en='Read the last article')
    assert 'දෙවන ලිපියේ' in r['speakable']


def test_next_part_is_still_next_part(monkeypatch):
    c = Chat(monkeypatch)
    r = _l7(c, 'ඊළඟ', 'NEXT', en='Next')
    assert r['action'] == 'NAV' and 'පළමු ලිපියේ 0 වන' in r['speakable']


@pytest.mark.parametrize('text', ['ලිපිය එකක් කියවන්න', 'ලිපි දෙකම කියවන්න'])
def test_an_article_or_both_articles_is_not_article_one_or_two(monkeypatch, text):
    assert turns._mentioned_article(text) is None


def test_case_endings_still_name_the_article():
    assert turns._mentioned_article('දෙවන ලිපියට යන්න') == 2
    assert turns._mentioned_article('ලිපිය 1 සහ') == 1 and turns._mentioned_article('ලිපිය 10') is None


def test_a_headline_gets_its_own_pause_but_a_line_wrap_does_not():
    from layers.l7_personalization.delivery.planner import split_segments
    art = 'ලිපිය 2: බෝක්කුව කැඩිලා\n\nපාර අසල බෝක්කුව කැඩී ඇත. ප්‍රදේශවාසීන්\nඅපහසුතාවට පත්ව සිටිති.'
    assert split_segments(art) == ['ලිපිය 2: බෝක්කුව කැඩිලා', 'පාර අසල බෝක්කුව කැඩී ඇත.',
                                   'ප්‍රදේශවාසීන් අපහසුතාවට පත්ව සිටිති.']
    assert split_segments('ප.ව. 4ට රැස්වීම. මිල 25.50කි.') == ['ප.ව. 4ට රැස්වීම.', 'මිල 25.50කි.']

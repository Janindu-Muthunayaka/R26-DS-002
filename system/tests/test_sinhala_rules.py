"""Component 4's Sinhala command rules and the two modifier fixes (18 Oct 2026).

THE BUGS. Short control commands were broken by the translation before the
LLM saw them, then sent to RAG as questions and answered "not enough
information":
    "පස්සට යන්න" (go back)      -> "Go back to the drawing board" -> REPHRASE
    "ඊළඟ එකට යන්න" (go to next) -> "Next time we go together"     -> PLAN_OUTING
And two modifier mistakes the turn policy then acted on:
    "ආයෙත් කලින් ප්‍රවෘත්තියට යන්න" (go back to the previous news again) was a REPLAY
    "සද්දෙ නිසා මට ඇහුනෙ නෑ" (didn't hear, because of the noise) was "turn the volume up"

Measured on 556 commands (tools/bench/replay_turns.py): the rules fixed 40
labels and broke none; control commands reaching RAG went from 17 to 1.
"""
import pytest

from core.schemas import Article, Box, Document
from layers.l6_generator import generate as l6gen
from layers.l6_generator import turns
from layers.l7_personalization.intent_detection.modifiers import build_frame
from layers.l7_personalization.intent_detection.sinhala_rules import rule_intent


@pytest.mark.parametrize('text, intent', [
    # back
    ('පස්සට යන්න', 'PREVIOUS'),
    ('ආපස්සට යන්න', 'PREVIOUS'),
    ('ආපහු පස්සට යන්න', 'PREVIOUS'),
    ('පෙර පිටුව', 'PREVIOUS'),
    ('පෙර සිරස්තලයට යන්න', 'PREVIOUS'),
    ('මීට කලින් තිබ්බ එක', 'PREVIOUS'),
    ('ආයෙත් කලින් ප්‍රවෘත්තියට යන්න', 'PREVIOUS'),
    ('ඊළඟ එකට යන්න නෑ නෑ කලින් එකට යන්න', 'PREVIOUS'),     # the last direction said
    # next
    ('ඊළඟ එකට යන්න', 'NEXT'),
    ('මේකට පස්සේ තියෙන ලිපිය කියවන්න', 'NEXT'),
    ('වෙන පුවතක් අරගන්න', 'NEXT'),
    ('පිටුව පෙරලන්න', 'NEXT'),                               # turn the page
    ('නවත්තලා ඊළඟ එක කියවන්න', 'NEXT'),                      # not STOP
    # pages, stop, speed, repeat, wording, headlines
    ('අංක පහ පිටුව පෙරලන්න', 'GO_TO_PAGE'),
    ('පිටු අංක දෙකට ආපහු යන්න', 'GO_TO_PAGE'),
    ('නවත්වන්න', 'STOP'),
    ('වේගෙන් කියන්න', 'CHANGE_SPEED'),
    ('තේරුනේ නෑ නැවත හෙමින් කියන්න', 'REPEAT'),
    ('කලින් කියපු කොටස තව පාරක් කියන්න', 'REPEAT'),
    ('වෙනත් ආකාරයකට මේ වාක්‍යය කියවන්න', 'REPHRASE'),
    ('ඔක්කොම නෙවෙයි සාරාංශය විතරක් ඇති', 'SUMMARIZE'),
    ('වෙනත් මාතෘකා මොනවද තියෙන්නේ', 'READ_HEADLINES'),
    # found on the running system (bench_glitch.py)
    ('ම්ම් අහ්', 'UNKNOWN'),                                 # hesitation: say it again
    ('නවත්තලා නවත්තලා කියන්න', 'CHANGE_SPEED'),              # pauses, not STOP
    ('කෑලි කෑලි කියන්න', 'CHANGE_SPEED'),
])
def test_an_unambiguous_command_is_decided_in_sinhala(text, intent):
    assert rule_intent(text) == intent


@pytest.mark.parametrize('text', [
    'දේශපාලන පිටුව පෙරලන්න',            # a SECTION, not "turn the page"
    'හෙමින් ලිපිය කියවන්න',              # read the article (slowly), not a bare speed change
    'මේ පිටුවේ තියෙන මාතෘකා මොනවද',      # a question about the article
    'මේ කරුණ තවදුරටත් පැහැදිලි කරන්න',   # content: the LLM decides
    'කලින් ජනාධිපති කවුද',               # a question that says "previous"
    'සරලව කියන්න',
])
def test_anything_unsure_is_left_to_the_llm(text):
    assert rule_intent(text) is None


def test_going_back_again_is_navigation_not_a_replay():
    f = build_frame('ආයෙත් කලින් ප්‍රවෘත්තියට යන්න', '', 'PREVIOUS')
    assert f['intent'] == 'PREVIOUS' and f['modifiers']['repeat'] is False
    f = build_frame('කලින් කියපු කොටස තව පාරක් කියන්න', '', 'UNKNOWN')
    assert f['modifiers']['repeat'] is True                  # no movement: still a repeat


def test_noise_is_not_a_volume_request():
    f = build_frame('සද්දෙ නිසා මට ඇහුනෙ නෑ', '', 'REPEAT')
    assert f['modifiers']['volume'] is False and f['modifiers']['repeat'] is True
    for text in ('ශබ්දය වැඩි කරන්න', 'සද්දෙ ටිකක් අඩු කරන්න', 'හයියෙන් කියන්න'):
        assert build_frame(text, '', 'UNKNOWN')['modifiers']['volume'] is True, text


# ── Through the turn policy: what the user hears ──────────────────────────
BOX = Box(x1=0, y1=0, x2=10, y2=10)


@pytest.fixture
def session(monkeypatch):
    long = ' '.join(f'පළමු ලිපියේ {i} වන වාක්‍යයයි.' for i in range(30))
    doc = Document(articles=[Article(index=i, box=BOX, body_raw=long, body=long,
                                     title=f'මාතෘකාව {i}', verdict='ok') for i in range(3)])
    real, asked = l6gen.answer, []

    def fake(d, voice, mode=None, cursor=0):
        if str(voice.get('route')).upper() == 'GENERATE' \
                and str(voice.get('intent')).upper() not in l6gen.LOCAL_INTENTS:
            asked.append(voice)
            return l6gen._out(True, 'GENERATE', voice['intent'], '[RAG]', answer_si='[RAG]', cursor=cursor)
        return real(d, voice, mode='off', cursor=cursor)

    monkeypatch.setattr(l6gen, 'answer', fake)
    state = {}
    turns.seed_first_scan(state, real(doc, {'intent': 'FIRST_SCAN', 'route': 'GENERATE'}, mode='off')['speakable'])

    def say(text, llm_label):
        intent = rule_intent(text) or llm_label
        voice = {'route': 'GENERATE', 'intent': intent, 'english_translation': text,
                 'sinhala_input': text, 'frame': build_frame(text, '', intent),
                 'style_class': 'Detailed', 'prompt_modifier': ''}
        asked.clear()
        return turns.run(doc, voice, state, cursor=0), list(asked)

    say('පළමු ලිපිය කියවන්න', 'READ_ALOUD')
    return say


@pytest.mark.parametrize('text, broken_llm_label', [
    ('පස්සට යන්න', 'REPHRASE'),
    ('ඊළඟ එකට යන්න', 'PLAN_OUTING'),
    ('පෙර පිටුව', 'IDENTIFY_CONTENT'),
    ('වේගෙන් කියන්න', 'SUMMARIZE'),
])
def test_a_mistranslated_command_never_reaches_rag(session, text, broken_llm_label):
    r, asked = session(text, broken_llm_label)
    assert asked == [], f'{text!r} was sent to RAG'
    assert r['route'] == 'LOCAL'


def test_didnt_hear_because_of_noise_replays(session):
    r, asked = session('සද්දෙ නිසා මට ඇහුනෙ නෑ', 'REPEAT')
    assert r['action'] == 'REPLAY' and asked == []

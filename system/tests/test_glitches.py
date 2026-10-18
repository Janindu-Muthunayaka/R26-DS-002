"""Glitches found by tools/bench/bench_glitch.py on the running system (18 Oct 2026).

1. A failed RAG answer was replaced by ARTICLE TEXT ("0390 290 .... ෴෴") and
   presented as the answer to "say it simpler".
2. An empty utterance was stored as the "last question"; every later style
   request re-asked it and RAG rejected the empty string.
3. A failed answer was saved as the last thing spoken, so "repeat" replayed it.
4. An empty upload to /capture was an HTTP 500.
"""
import pytest

from core.schemas import Article, Box, Document
from layers.l6_generator import generate as l6gen
from layers.l6_generator import turns

BOX = Box(x1=0, y1=0, x2=10, y2=10)
BODY = 'පළමු ලිපියේ පෙළ මෙයයි. ' * 20
RAG_FAILED = 'පිළිතුර ලබාගැනීමට නොහැකි විය.'


def _doc():
    return Document(articles=[Article(index=i, box=BOX, body=BODY, body_raw=BODY,
                                      title=f'මාතෘකාව {i}', verdict='ok') for i in range(2)])


def test_a_rag_failure_is_spoken_as_a_failure_not_as_article_text(monkeypatch):
    monkeypatch.setattr(l6gen.svc, 'post_json',
                        lambda *a, **k: ({'ok': False, 'speakable_text': RAG_FAILED,
                                          'notes': ['retrieval failed']}, ''))
    for q in ('සරලව කියන්න', 'ලිපිය එකේ කරුණ මොකක්ද', 'මේක 2 වතාවක් කියන්න'):
        out = l6gen.answer(_doc(), {'route': 'GENERATE', 'intent': 'SIMPLIFY',
                                    'english_translation': q, 'sinhala_input': q}, mode='http')
        assert out['ok'] is False
        assert BODY.strip()[:20] not in out['speakable'], f'{q!r}: article text read as the answer'
        assert out['speakable'] == RAG_FAILED


def _session(monkeypatch, reply):
    real = l6gen.answer
    asked = []

    def fake(doc, voice, mode=None, cursor=0):
        if str(voice.get('route')).upper() == 'GENERATE' \
                and str(voice.get('intent')).upper() not in l6gen.LOCAL_INTENTS:
            asked.append(voice)
            return reply(voice, cursor)
        return real(doc, voice, mode='off', cursor=cursor)

    monkeypatch.setattr(l6gen, 'answer', fake)
    doc, state = _doc(), {}
    turns.seed_first_scan(state, real(doc, {'intent': 'FIRST_SCAN', 'route': 'GENERATE'}, mode='off')['speakable'])

    def say(si, intent, en=None):
        asked.clear()
        v = {'route': 'GENERATE', 'intent': intent, 'sinhala_input': si,
             'english_translation': si if en is None else en, 'style_class': 'Detailed',
             'prompt_modifier': ''}
        return turns.run(doc, v, state, cursor=0), list(asked), state
    return say


def test_an_empty_utterance_is_never_a_question(monkeypatch):
    ok = lambda v, c: l6gen._out(True, 'GENERATE', v['intent'], '[RAG] ' + v['english_translation'],
                                 answer_si='x', cursor=c)
    say = _session(monkeypatch, ok)
    r, asked, state = say('', 'ASK_QUESTION', en='')
    assert asked == [] and r['action'] == 'ASK'
    assert not state.get('last_question')
    # ...so "say it simpler" afterwards acts on the article, not on ''.
    r, asked, _ = say('සරලව කියන්න', 'SIMPLIFY')
    assert asked and (asked[0]['english_translation'] or '').strip()


def test_a_failed_answer_is_not_replayed_as_content(monkeypatch):
    fail = lambda v, c: l6gen._out(False, 'GENERATE', v['intent'], RAG_FAILED, cursor=c)
    say = _session(monkeypatch, fail)
    say('පළමු ලිපිය කියවන්න', 'READ_ALOUD')
    r, _, state = say('බෝක්කුව හැදුවේ කවුද', 'ASK_QUESTION')
    assert r['ok'] is False
    assert state['last_spoken']['text'] != RAG_FAILED, 'the failure sentence became "the last thing said"'


def test_an_empty_upload_is_refused_not_a_500(tmp_path):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from app.server import build

    class Pipe:
        dev = 'cpu'

        def run(self, paths, **kw):
            raise AssertionError('the pipeline must not run on an empty upload')

    client = TestClient(build(Pipe(), tmp_path))
    for data in (b'', b'not an image'):
        r = client.post('/capture', files=[('frames', ('x.jpg', data, 'image/jpeg'))])
        assert r.status_code == 400, data
        assert r.json()['ok'] is False

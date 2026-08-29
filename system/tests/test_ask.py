"""`/ask` end to end, without models, without a phone, without a network.

The conversation is the part of this system a viva actually exercises, and it
is the part with the most ways to fail silently. Three of those are asserted
here because each one has already happened once during development:

  1. `/ask` on a job that was never captured, or whose session expired,
     returned a 404 with an empty body. The phone spoke nothing. A blind user
     cannot tell "no answer" from "the app froze", so every failure path here
     must carry a non-empty `speakable`.

  2. A crash inside the generator became a 500 with a stack trace. Same
     outcome: silence. `/ask` is allowed to fail; it is not allowed to fail
     quietly.

  3. `next` walked the article from the start every time, because the cursor
     lived in the request instead of the session. The article never advanced
     and nobody noticed, because each part on its own sounded correct.

The pipeline is faked. These tests are about the endpoint's promises.
"""
import io

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient          # noqa: E402

from core.schemas import Article, Box, Document, Region   # noqa: E402
from app.server import build                              # noqa: E402


@pytest.fixture(autouse=True)
def isolate_ask_tests(monkeypatch):
    monkeypatch.setattr('layers.l0_voice.voice.VOICE_MODE', 'stub')
    monkeypatch.setattr('layers.l6_generator.generate.RAG_MODE', 'off')
LONG = ' '.join(
    f'කුරුණෑගල නගර සභාව අද පැවති {i} වන රැස්වීමේදී නව තීරණයක් ගත් බව '
    f'සභාපතිවරයා පැවසීය.'
    for i in range(1, 13))          # long enough to need several parts


class FakePipeline:
    dev = 'cpu'

    def __init__(self, doc=None, boom=False):
        self.doc, self.boom = doc, boom

    def run(self, paths, **kw):
        if self.boom:
            raise RuntimeError('pipeline exploded')
        return self.doc


def _doc(body=LONG, title='නගර සභා තීරණය', warnings=()):
    box = Box(x1=0, y1=0, x2=100, y2=100)
    return Document(
        articles=[Article(index=0, box=box,
                          regions=[Region(box=box, label='text')],
                          body=body, title=title, verdict='ok')],
        warnings=list(warnings), timings={'total': 1.0})


def _jpeg(w=60, h=40):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 230, np.uint8)).save(buf, format='JPEG')
    return buf.getvalue()


def _captured(tmp_path, doc=None):
    """A live client with one article already in session, and its job id."""
    c = TestClient(build(FakePipeline(doc or _doc()), tmp_path))
    r = c.post('/capture',
               files=[('frames', ('f0.jpg', _jpeg(), 'image/jpeg'))])
    assert r.status_code == 200 and r.json()['ok'] is True
    return c, r.json()['job']


def _ask(c, job, text):
    return c.post('/ask', json={'job': job, 'text': text})


# ---- failure must still be speakable -------------------------------------
def test_unknown_job_is_a_spoken_sentence_not_an_empty_404(tmp_path):
    c = TestClient(build(FakePipeline(_doc()), tmp_path))
    r = _ask(c, 'never-captured', 'නැවත කියවන්න')
    assert r.status_code == 404
    j = r.json()
    assert j['ok'] is False
    assert j['speakable'].strip(), (
        'the phone speaks `speakable`; an empty one is silence, and silence '
        'is indistinguishable from a frozen app to a blind user')
    assert j['error']


def test_expired_session_answers_exactly_like_a_miss(tmp_path):
    """Expiry and a wrong job id are the same sentence on purpose: the user's
    next action is identical either way — point the camera and capture."""
    c, job = _captured(tmp_path)
    miss = _ask(c, 'no-such-job', 'නැවත කියවන්න').json()

    c.app.state.sessions._items.clear()          # simulate the TTL elapsing
    expired = _ask(c, job, 'නැවත කියවන්න').json()

    assert expired['speakable'] == miss['speakable']
    assert expired['ok'] is False


def test_a_crash_in_the_generator_is_spoken_not_traced(tmp_path, monkeypatch):
    from app import server as srv
    c, job = _captured(tmp_path)
    monkeypatch.setattr(srv.l6gen, 'answer',
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError('generator exploded')))
    r = _ask(c, job, 'නැවත කියවන්න')
    assert r.status_code == 500
    j = r.json()
    assert j['ok'] is False
    assert j['speakable'].strip()
    assert 'RuntimeError' in j['error']


def test_every_field_the_phone_reads_is_present(tmp_path):
    c, job = _captured(tmp_path)
    j = _ask(c, job, 'නැවත කියවන්න').json()
    for k in ('ok', 'job', 'route', 'intent', 'speakable', 'answer_si',
              'sources', 'warnings', 'timings'):
        assert k in j, f'QuestionListener/MainActivity reads {k}'
    assert j['job'] == job


# ---- the local intents, which must work with nothing running -------------
def test_repeat_returns_the_article_with_no_service(tmp_path):
    c, job = _captured(tmp_path)
    j = _ask(c, job, 'නැවත කියවන්න').json()
    assert j['ok'] is True
    assert j['route'] == 'LOCAL'
    assert 'කුරුණෑගල' in j['speakable']


def test_title_and_length_are_answered_from_session(tmp_path):
    c, job = _captured(tmp_path)
    t = _ask(c, job, 'ශීර්ෂය කියන්නද').json()
    assert t['ok'] is True and 'නගර සභා තීරණය' in t['speakable']

    n = _ask(c, job, 'වචන කීයද').json()
    assert n['ok'] is True and any(ch.isdigit() for ch in n['speakable'])


def test_warnings_intent_says_so_when_nothing_was_missed(tmp_path):
    c, job = _captured(tmp_path)
    j = _ask(c, job, 'මොනවා හරි මඟ හැරුණාද').json()
    assert j['ok'] is True
    assert j['speakable'].strip()


# ---- the cursor, which lives in the session -----------------------------
def test_next_advances_and_does_not_restart(tmp_path):
    """The bug this guards: the cursor lived in the request, so every `next`
    replayed part one. Each part sounded right on its own."""
    c, job = _captured(tmp_path)
    first = _ask(c, job, 'මුල සිට කියවන්න').json()['speakable']
    second = _ask(c, job, 'ඊළඟ කොටස').json()['speakable']
    assert second and second != first, 'next replayed the same part'


def test_previous_walks_back_to_where_it_was(tmp_path):
    c, job = _captured(tmp_path)
    a = _ask(c, job, 'මුල සිට කියවන්න').json()['speakable']
    _ask(c, job, 'ඊළඟ කොටස')
    back = _ask(c, job, 'කලින් කොටස').json()['speakable']
    assert back == a


def test_the_cursor_is_per_job_not_global(tmp_path):
    """Two captures in one session. Advancing one must not move the other —
    the cursor belongs to the article, and expires with it."""
    c = TestClient(build(FakePipeline(_doc()), tmp_path))
    jobs = []
    for _ in range(2):
        r = c.post('/capture',
                   files=[('frames', ('f.jpg', _jpeg(), 'image/jpeg'))])
        jobs.append(r.json()['job'])
    assert jobs[0] != jobs[1]

    _ask(c, jobs[0], 'මුල සිට කියවන්න')
    _ask(c, jobs[0], 'ඊළඟ කොටස')
    fresh = _ask(c, jobs[1], 'ඊළඟ කොටස').json()['speakable']
    start = _ask(c, jobs[0], 'මුල සිට කියවන්න').json()['speakable']
    assert fresh == start, 'the second job inherited the first job\'s cursor'


# ---- diagnostics endpoints ----------------------------------------------
def test_session_endpoint_reports_the_held_article(tmp_path):
    c, job = _captured(tmp_path)
    j = c.get(f'/session/{job}').json()
    assert j['ok'] is True and j['job'] == job
    assert c.get('/session/nope').status_code == 404


def test_document_endpoint_returns_the_full_document(tmp_path):
    """web/debug.html read `j.document` from /capture, which stopped being
    returned on 21 Aug and broke the page silently. This is its replacement."""
    c, job = _captured(tmp_path)
    j = c.get(f'/document/{job}').json()
    assert j['ok'] is True
    assert j['document']['articles'][0]['body'].startswith('කුරුණෑගල')
    assert c.get('/document/nope').status_code == 404


def test_a_failed_capture_is_not_remembered(tmp_path):
    """Nothing was read, so there is nothing to ask about. Storing an empty
    Document would make "read that again" replay silence instead of saying
    there is nothing to repeat."""
    c = TestClient(build(FakePipeline(_doc(body='', title='')), tmp_path))
    r = c.post('/capture',
               files=[('frames', ('f.jpg', _jpeg(), 'image/jpeg'))])
    job = r.json()['job']
    assert r.json()['ok'] is False
    assert c.get(f'/session/{job}').status_code == 404
    assert _ask(c, job, 'නැවත කියවන්න').json()['ok'] is False

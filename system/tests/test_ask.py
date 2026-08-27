"""`POST /ask` — the conversation path, tested with both components absent.

That is the condition the endpoint has to survive: Component 3 not running,
Component 4 not running, no network, no API key. With both off the loop must
still work for "read that again", and a question that genuinely needs
generation must come back with a SPEAKABLE sentence rather than a stack trace
or silence.

The pipeline is faked. These are contract tests, not model tests.
"""
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient          # noqa: E402

from app.server import build                       # noqa: E402
from core.schemas import Article, Box, Document, Region   # noqa: E402

BOX = Box(x1=0, y1=0, x2=100, y2=100)
ARTICLE = 'කුරුණෑගල නගර සභාව අද රැස්විය.'


class FakePipeline:
    dev = 'cpu'

    def __init__(self, doc):
        self.doc = doc

    def run(self, paths, **kw):
        return self.doc


def _doc(body=ARTICLE):
    arts = []
    if body:
        arts.append(Article(index=0, box=BOX,
                            regions=[Region(box=BOX, label='text')],
                            body_raw=body, body=body, verdict='ok'))
    return Document(articles=arts, timings={'total': 1.0})


def _jpeg():
    buf = io.BytesIO()
    Image.fromarray(np.full((40, 60, 3), 230, np.uint8)).save(buf, format='JPEG')
    return buf.getvalue()


def _client(doc=None, tmp_path=None):
    return TestClient(build(FakePipeline(doc or _doc()), tmp_path))


def _capture(client):
    r = client.post('/capture',
                    files=[('frames', ('f0.jpg', _jpeg(), 'image/jpeg'))])
    assert r.status_code == 200, r.text
    return r.json()['job']


# ---- the memory ----------------------------------------------------------
def test_capture_puts_the_article_in_session(tmp_path):
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    s = c.get(f'/session/{job}').json()
    assert s['ok'] is True and s['n_articles'] == 1 and s['chars'] > 0


def test_a_capture_that_read_nothing_is_not_remembered(tmp_path):
    """"Read that again" on an empty article must say so, not replay
    silence."""
    c = _client(_doc(body=''), tmp_path)
    r = c.post('/capture',
               files=[('frames', ('f0.jpg', _jpeg(), 'image/jpeg'))])
    job = r.json()['job']
    assert c.get(f'/session/{job}').status_code == 404


def test_unknown_job_is_spoken_not_thrown(tmp_path):
    r = _client(tmp_path=tmp_path).post('/ask',
                                        json={'job': 'nope', 'text': 'මොකක්ද'})
    assert r.status_code == 404
    j = r.json()
    assert j['ok'] is False
    assert j['speakable'], 'a blind user must hear why nothing happened'


# ---- works with both components off -------------------------------------
def test_repeat_is_answered_from_session_with_no_services(tmp_path):
    """The most common follow-up survives a RAG outage during a viva,
    because it never needed RAG."""
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    j = c.post('/ask', json={'job': job, 'text': 'නැවත කියවන්න'}).json()
    assert j['ok'] is True
    assert ARTICLE in j['speakable']
    assert j['route'] == 'LOCAL'


def test_a_real_question_gets_a_speakable_refusal_not_a_fabricated_answer(tmp_path):
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    j = c.post('/ask', json={'job': job,
                             'text': 'මේක සාරාංශ කරන්න'}).json()
    assert j['ok'] is False, 'no answer was generated, so ok must be false'
    assert j['speakable'], 'silence is the one unacceptable outcome'
    assert j['answer_si'] == '', 'nothing may be invented while RAG is off'
    assert any('rag' in w for w in j['warnings'])


def test_stub_routing_is_labelled_in_the_warnings(tmp_path):
    """So no transcript can be read as evidence Component 4 ran."""
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    j = c.post('/ask', json={'job': job, 'text': 'මොකක්ද'}).json()
    assert any('stub' in w for w in j['warnings'])


def test_stop_asks_the_phone_to_act_not_to_speak(tmp_path):
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    j = c.post('/ask', json={'job': job, 'text': 'නවත්වන්න'}).json()
    assert j['intent'] == 'STOP' and j['speakable'] == ''


# ---- the response shape --------------------------------------------------
PHONE_FIELDS = ('ok', 'job', 'route', 'intent', 'speakable', 'answer_si',
                'sources', 'warnings', 'timings')


def test_every_field_the_phone_reads_is_present(tmp_path):
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    j = c.post('/ask', json={'job': job, 'text': 'නැවත'}).json()
    missing = [k for k in PHONE_FIELDS if k not in j]
    assert not missing, f'the phone reads these by name: {missing}'
    assert isinstance(j['warnings'], list) and isinstance(j['sources'], list)


def test_timings_are_recorded(tmp_path):
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    j = c.post('/ask', json={'job': job, 'text': 'නැවත'}).json()
    assert {'voice', 'generate', 'total'} <= set(j['timings'])


# ---- the reading path is untouched --------------------------------------
def test_capture_response_is_unchanged_by_all_of_this(tmp_path):
    """`/ask` must not have altered a single field the phone already reads."""
    c = _client(tmp_path=tmp_path)
    r = c.post('/capture',
               files=[('frames', ('f0.jpg', _jpeg(), 'image/jpeg'))]).json()
    for k in ('ok', 'job', 'title', 'body', 'warnings', 'n_articles',
              'audio_url', 'timings'):
        assert k in r
    assert r['audio_url'] is None


# ---- the debug page's data source ---------------------------------------
def test_document_endpoint_returns_body_raw(tmp_path):
    """web/debug.html needs `body_raw` to show OCR against correction, and
    `/capture` has not carried a `document` key since 21 Aug 2026 — which is
    why the page showed "no articles" for every upload until this existed."""
    c = _client(tmp_path=tmp_path)
    job = _capture(c)
    d = c.get(f'/document/{job}').json()
    assert d['ok'] is True
    art = d['document']['articles'][0]
    assert art['body_raw'] == ARTICLE and art['body'] == ARTICLE
    assert 'timings' in d['document']


def test_document_endpoint_404s_for_an_unknown_job(tmp_path):
    assert _client(tmp_path=tmp_path).get('/document/nope').status_code == 404

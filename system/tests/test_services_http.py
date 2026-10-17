"""The four components talk over HTTP. This is what happens when that fails.

The components have incompatible dependency sets, so they cannot be imported
into one process — they are separate services and the calls between them are
network calls. Every network call has failure modes that do not exist for a
function call: the service is down, it is slow, it answers with HTML, it
answers with a JSON object missing half its fields, or it is the stub and not
the real thing at all.

None of those may reach the user as a stack trace or as silence, and none of
them may be reported as a result. A real HTTP server runs in a thread here —
no mocks — because the mock of a socket is exactly where these bugs hide.

The last two tests are the ones that matter most:

  * a stub answering on the real port must NOT be stamped `component4`;
  * a RAG reply carrying `ok: false` and a failure sentence must NOT come back
    looking like a successful answer, which is what happened before `ok` was
    threaded through — a non-empty string was taken as success.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core import svc
from core.schemas import Article, Box, Document
from layers.l0_voice import voice as l0
from layers.l6_generator import generate as l6gen


# ---- a real server, driven by a per-test script -------------------------
class _Script:
    """What the next request should do. Set by each test."""
    status = 200
    body = '{}'
    delay = 0.0
    content_type = 'application/json'
    seen = None


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(n).decode('utf-8')
        try:
            _Script.seen = json.loads(raw)
        except ValueError:
            _Script.seen = raw
        if _Script.delay:
            time.sleep(_Script.delay)
        payload = _Script.body.encode('utf-8')
        self.send_response(_Script.status)
        self.send_header('Content-Type', _Script.content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):                  # keep pytest output readable
        pass


@pytest.fixture(scope='module')
def base():
    srv = HTTPServer(('127.0.0.1', 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}'
    srv.shutdown()


@pytest.fixture(autouse=True)
def reset():
    _Script.status, _Script.body, _Script.delay = 200, '{}', 0.0
    _Script.content_type, _Script.seen = 'application/json', None


DEAD = 'http://127.0.0.1:9'          # discard port: refuses immediately


def _doc(body='කුරුණෑගල නගර සභාව අද රැස්විය.'):
    box = Box(x1=0, y1=0, x2=10, y2=10)
    return Document(articles=[Article(index=0, box=box, body=body,
                                      title='ශීර්ෂය', verdict='ok')],
                    warnings=[], timings={})


VOICE_OK = {'route': 'GENERATE', 'intent': 'SUMMARY',
            'english_translation': 'what is this about',
            'style_class': 'Detailed', 'prompt_modifier': '',
            'personalization_flags': {}}


# ---- core/svc.py ---------------------------------------------------------
def test_post_json_returns_the_parsed_object(base):
    _Script.body = '{"hello": "\\u0dc4\\u0dd2"}'
    reply, reason = svc.post_json(f'{base}/x', {'a': 1})
    assert reason == '' and reply['hello']
    assert _Script.seen == {'a': 1}, 'the body must arrive as JSON'


def test_an_unreachable_service_is_a_reason_not_an_exception():
    reply, reason = svc.post_json(f'{DEAD}/x', {}, timeout_s=2)
    assert reply is None
    assert reason and 'unreachable' in reason


def test_an_http_error_carries_the_status(base):
    _Script.status, _Script.body = 500, '{"detail": "boom"}'
    reply, reason = svc.post_json(f'{base}/x', {})
    assert reply is None and reason.startswith('HTTP 500')


def test_html_instead_of_json_is_reported_as_such(base):
    """What a proxy or a wrong port actually returns."""
    _Script.body, _Script.content_type = '<html>404</html>', 'text/html'
    reply, reason = svc.post_json(f'{base}/x', {})
    assert reply is None and 'not JSON' in reason


def test_a_json_array_is_refused(base):
    """`reply.get(...)` on a list is an AttributeError two layers away."""
    _Script.body = '[1, 2, 3]'
    reply, reason = svc.post_json(f'{base}/x', {})
    assert reply is None and 'expected an object' in reason


def test_a_slow_service_times_out_rather_than_hanging(base):
    _Script.delay = 1.5
    t0 = time.time()
    reply, reason = svc.post_json(f'{base}/x', {}, timeout_s=0.4)
    assert reply is None and reason
    assert time.time() - t0 < 1.4, 'the timeout was not honoured'


def test_an_unserialisable_body_never_reaches_the_socket():
    reply, reason = svc.post_json('http://127.0.0.1:1/x', {'f': {1, 2}})
    assert reply is None and 'unserialisable' in reason


# ---- Layer 0 over HTTP ---------------------------------------------------
def test_voice_http_uses_component_4_when_it_answers(base, monkeypatch):
    monkeypatch.setattr(l0, 'VOICE_URL', base)
    _Script.body = json.dumps(VOICE_OK)
    out = l0.interpret('මේ ලිපිය ගැන කියන්න', mode='http')
    assert out['source'] == 'component4'
    assert out['route'] == 'GENERATE'
    assert _Script.seen['text']


def test_voice_falls_back_to_the_stub_when_component_4_is_down(monkeypatch):
    """The keyword router still works with nothing running. That is why the
    demo survives a dead service — and the fallback is recorded, so a
    transcript can never be read as evidence Component 4 ran."""
    monkeypatch.setattr(l0, 'VOICE_URL', DEAD)
    out = l0.interpret('නැවත කියවන්න', mode='http')
    assert out['intent'] == 'REPEAT'
    assert out['source'] == 'stub-fallback'
    assert 'unavailable' in out['warning']


def test_a_reply_missing_contract_fields_is_refused_not_half_used(
        base, monkeypatch):
    monkeypatch.setattr(l0, 'VOICE_URL', base)
    _Script.body = json.dumps({'route': 'GENERATE'})     # the rest absent
    out = l0.interpret('නැවත කියවන්න', mode='http')
    assert out['source'] == 'stub-fallback'
    assert 'missing' in out['warning']


def test_the_stub_service_is_not_stamped_as_component_4(base, monkeypatch):
    """tools/stub_services.py answers the real endpoint with a valid shape.
    Without the `stub` flag it passes validation and is indistinguishable
    from the real component — in exactly the setup used for testing."""
    monkeypatch.setattr(l0, 'VOICE_URL', base)
    _Script.body = json.dumps({**VOICE_OK, 'stub': True})
    out = l0.interpret('මේ ලිපිය ගැන කියන්න', mode='http')
    assert out['source'] == 'stub-service'
    assert out['source'] != 'component4'


def test_stub_mode_never_touches_the_network(base, monkeypatch):
    monkeypatch.setattr(l0, 'VOICE_URL', base)
    _Script.body = json.dumps(VOICE_OK)
    out = l0.interpret('නැවත කියවන්න', mode='stub')
    assert out['source'] == 'stub'
    assert _Script.seen is None


# ---- Layer 6 over HTTP ---------------------------------------------------
def test_rag_http_returns_the_generated_answer(base, monkeypatch):
    monkeypatch.setattr(l6gen, 'RAG_URL', base)
    _Script.body = json.dumps({
        'ok': True, 'answer_si': 'නගර සභාව රැස්විය.',
        'speakable_text': 'නගර සභාව රැස්විය.',
        'retrieved_sources': [{'id': 'a1', 'score': 0.8}]})
    out = l6gen.answer(_doc(), dict(VOICE_OK), mode='http')
    assert out['ok'] is True
    assert out['route'] == 'GENERATE'
    assert out['speakable']
    assert out['sources'][0]['id'] == 'a1'


def test_a_rag_failure_sentence_is_not_reported_as_an_answer(base,
                                                             monkeypatch):
    """THE BUG. Component 3 signals failure with ok:false and still fills
    speakable_text so the user hears something. A non-empty string looked
    exactly like success, so a failed question was logged as answered."""
    monkeypatch.setattr(l6gen, 'RAG_URL', base)
    _Script.body = json.dumps({
        'ok': False,
        'speakable_text': 'පිළිතුර ලබාගැනීමට නොහැකි විය.',
        'notes': ['embedding call failed']})
    out = l6gen.answer(_doc(), dict(VOICE_OK), mode='http')
    assert out['ok'] is False, 'ok:false from the service was ignored'
    assert out['speakable'], 'the user must still hear something'
    assert any('failure' in w for w in out['warnings'])
    assert any('embedding' in w for w in out['warnings'])


def test_rag_down_still_speaks(base, monkeypatch):
    monkeypatch.setattr(l6gen, 'RAG_URL', DEAD)
    out = l6gen.answer(_doc(), dict(VOICE_OK), mode='http')
    assert out['ok'] is False
    assert out['speakable'].strip()
    assert any('rag:' in w for w in out['warnings'])


def test_an_empty_rag_answer_is_a_failure_not_silence(base, monkeypatch):
    monkeypatch.setattr(l6gen, 'RAG_URL', base)
    _Script.body = json.dumps({'ok': True, 'answer_si': '   '})
    out = l6gen.answer(_doc(), dict(VOICE_OK), mode='http')
    assert out['ok'] is False
    assert out['speakable'].strip()
    assert any('empty' in w for w in out['warnings'])


def test_a_stub_rag_is_labelled_in_the_warnings(base, monkeypatch):
    monkeypatch.setattr(l6gen, 'RAG_URL', base)
    _Script.body = json.dumps({'ok': True, 'answer_si': 'පිළිතුර',
                               'stub': True})
    out = l6gen.answer(_doc(), dict(VOICE_OK), mode='http')
    assert any('stub' in w for w in out['warnings'])


def test_rag_off_says_so_instead_of_pretending(base):
    out = l6gen.answer(_doc(), dict(VOICE_OK), mode='off')
    assert out['ok'] is False
    assert out['speakable'].strip()
    assert any('disabled' in w for w in out['warnings'])
    assert _Script.seen is None, 'off mode must not call anything'


def test_a_local_intent_never_reaches_the_service(base, monkeypatch):
    """The most common follow-up must survive a RAG outage. It does that by
    not being a RAG question at all."""
    monkeypatch.setattr(l6gen, 'RAG_URL', base)
    out = l6gen.answer(_doc(), {'route': 'LOCAL', 'intent': 'REPEAT',
                                'source': 'stub'}, mode='http')
    assert out['ok'] is True and out['route'] == 'LOCAL'
    assert _Script.seen is None


def test_the_payload_sent_to_rag_carries_no_label_or_confidence(base,
                                                                monkeypatch):
    """The reported result is a corrected-text improvement. Sending a label or
    a confidence would put a number in front of Component 3 that the research
    does not support."""
    monkeypatch.setattr(l6gen, 'RAG_URL', base)
    _Script.body = json.dumps({'ok': True, 'answer_si': 'පිළිතුර'})
    l6gen.answer(_doc(), dict(VOICE_OK), mode='http')
    sent = json.dumps(_Script.seen)
    assert 'confidence' not in sent
    assert '"label"' not in sent
    assert _Script.seen['ocr']['corrected_text']

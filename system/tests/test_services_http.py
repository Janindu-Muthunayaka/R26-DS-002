"""The `http` branch — the one that runs at the viva and has no owner yet.

Both HTTP branches are code that only executes once someone else's component
is running. That is exactly the code that fails on the day. These tests run a
real socket server in a thread and exercise the real `urllib` client, so the
request shape, the reply parsing, and — more importantly — every way a service
can let us down are all covered before either component exists.

A component being down must never become a stack trace. It becomes a warning
and a sentence the user can hear.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.schemas import Article, Box, Document
from layers.l0_voice import voice as l0
from layers.l6_generator import generate as l6gen

BOX = Box(x1=0, y1=0, x2=10, y2=10)
GOOD_VOICE = {
    'route': 'GENERATE', 'intent': 'SUMMARIZE',
    'english_translation': 'summarise this', 'style_class': 'Detailed',
    'prompt_modifier': '', 'personalization_flags': {'detail_level': 'brief'},
}


def _doc(body='පෙළ'):
    return Document(articles=[Article(index=0, box=BOX, body_raw=body,
                                      body=body, verdict='ok')])


class _Server:
    """A one-route JSON server on an ephemeral port."""

    def __init__(self, reply, code=200, raw=None):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get('Content-Length') or 0)
                outer.seen.append(json.loads(self.rfile.read(n) or b'{}'))
                body = (raw if raw is not None
                        else json.dumps(reply, ensure_ascii=False)).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(('127.0.0.1', 0), H)
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


DEAD_URL = 'http://127.0.0.1:9'      # discard port: refuses immediately


# ---- Layer 0, voice ------------------------------------------------------
def test_voice_http_uses_component4_reply(monkeypatch):
    with _Server(GOOD_VOICE) as s:
        monkeypatch.setattr(l0, 'VOICE_URL', s.url)
        out = l0.interpret('සාරාංශ කරන්න', user_id='user_001', mode='http')
    assert out['source'] == 'component4'
    assert out['intent'] == 'SUMMARIZE'
    assert out['personalization_flags'] == {'detail_level': 'brief'}
    assert s.seen[0]['text'] == 'සාරාංශ කරන්න'
    assert s.seen[0]['user_id'] == 'user_001'


def test_voice_http_degrades_when_the_service_is_down(monkeypatch):
    monkeypatch.setattr(l0, 'VOICE_URL', DEAD_URL)
    out = l0.interpret('මොකක්ද', mode='http')
    assert out['source'] == 'stub-fallback'
    assert 'unavailable' in out['warning']
    assert out['route'] == 'GENERATE', 'the question must still be routed'


def test_voice_http_rejects_a_reply_parse_voice_input_would_reject(monkeypatch):
    """Nadee's `parse_voice_input` raises on a missing field. Catching it here
    means the failure is named at the boundary instead of surfacing two layers
    away inside someone else's component."""
    with _Server({'route': 'GENERATE', 'intent': 'SUMMARIZE'}) as s:
        monkeypatch.setattr(l0, 'VOICE_URL', s.url)
        out = l0.interpret('x', mode='http')
    assert out['source'] == 'stub-fallback'
    assert 'missing' in out['warning']


def test_voice_http_survives_a_non_json_reply(monkeypatch):
    with _Server(None, raw='<html>proxy error</html>') as s:
        monkeypatch.setattr(l0, 'VOICE_URL', s.url)
        out = l0.interpret('x', mode='http')
    assert out['source'] == 'stub-fallback'


def test_voice_http_survives_a_500(monkeypatch):
    with _Server({'error': 'ollama not running'}, code=500) as s:
        monkeypatch.setattr(l0, 'VOICE_URL', s.url)
        out = l0.interpret('x', mode='http')
    assert out['source'] == 'stub-fallback' and 'HTTP 500' in out['warning']


# ---- Layer 6, RAG --------------------------------------------------------
RAG_OK = {'intent': 'SUMMARIZE', 'answer_si': 'පිළිතුර',
          'speakable_text': 'පිළිතුර',
          'retrieved_sources': [{'chunk_id': 'chunk_article_2'}]}


def test_rag_http_returns_the_answer(monkeypatch):
    with _Server(RAG_OK) as s:
        monkeypatch.setattr(l6gen, 'RAG_URL', s.url)
        out = l6gen.answer(_doc('ලිපිය'), GOOD_VOICE, mode='http')
    assert out['ok'] is True
    assert out['speakable'] == 'පිළිතුර'
    assert out['sources'][0]['chunk_id'] == 'chunk_article_2'


def test_rag_receives_the_payload_component3_expects(monkeypatch):
    """The shape `adapters.parse_ocr_input` and `parse_voice_input` require,
    asserted at the boundary rather than assumed."""
    with _Server(RAG_OK) as s:
        monkeypatch.setattr(l6gen, 'RAG_URL', s.url)
        l6gen.answer(_doc('ලිපිය'), GOOD_VOICE, mode='http')
    sent = s.seen[0]
    assert 'corrected_text' in sent['ocr'], 'parse_ocr_input requires this'
    assert sent['ocr']['token_source'] == 'diff'
    for k in ('route', 'intent', 'english_translation', 'style_class',
              'prompt_modifier', 'personalization_flags'):
        assert k in sent['voice'], f'parse_voice_input requires {k}'


def test_rag_down_is_spoken_not_raised(monkeypatch):
    monkeypatch.setattr(l6gen, 'RAG_URL', DEAD_URL)
    out = l6gen.answer(_doc(), GOOD_VOICE, mode='http')
    assert out['ok'] is False
    assert out['speakable'] == l6gen.SI_FAILED
    assert out['answer_si'] == '', 'nothing may be invented on failure'
    assert any('rag:' in w for w in out['warnings'])


def test_rag_empty_answer_is_a_failure_not_silence(monkeypatch):
    with _Server({'answer_si': '', 'speakable_text': ''}) as s:
        monkeypatch.setattr(l6gen, 'RAG_URL', s.url)
        out = l6gen.answer(_doc(), GOOD_VOICE, mode='http')
    assert out['ok'] is False and out['speakable'] == l6gen.SI_FAILED
    assert any('empty' in w for w in out['warnings'])


def test_local_intents_never_touch_the_network(monkeypatch):
    """"Read that again" must work with every service down — it is the most
    common follow-up and it never needed retrieval."""
    monkeypatch.setattr(l6gen, 'RAG_URL', DEAD_URL)
    voice = dict(GOOD_VOICE, intent='REPEAT', route='LOCAL')
    out = l6gen.answer(_doc('ලිපියේ පෙළ'), voice, mode='http')
    assert out['ok'] is True and 'ලිපියේ පෙළ' in out['speakable']
    assert not any('rag:' in w for w in out['warnings'])


# ---- a stub service must never pass for the real component ---------------
def test_a_stub_voice_service_is_not_reported_as_component4(monkeypatch):
    """tools/stub_services.py returns a VALID Component 4 shape. Validation
    alone would stamp it 'component4' and the guarantee that nothing routed
    through a stub is a result would quietly stop holding."""
    with _Server(dict(GOOD_VOICE, stub=True)) as s:
        monkeypatch.setattr(l0, 'VOICE_URL', s.url)
        out = l0.interpret('x', mode='http')
    assert out['source'] == 'stub-service'

    out2 = l6gen.answer(_doc(), out, mode='off')
    assert any('stub-service' in w for w in out2['warnings'])


def test_a_stub_rag_service_says_so_in_the_warnings(monkeypatch):
    with _Server(dict(RAG_OK, stub=True)) as s:
        monkeypatch.setattr(l6gen, 'RAG_URL', s.url)
        out = l6gen.answer(_doc(), GOOD_VOICE, mode='http')
    assert out['ok'] is True
    assert any('stub service' in w for w in out['warnings'])


# ---- the stub services' own liveness page --------------------------------
def test_stub_service_get_is_a_status_page_not_a_501():
    """Opening http://127.0.0.1:8101/interpret in a browser sends GET.
    http.server's default answer is `501 Unsupported method ('GET')`, which
    reads as a fault when the service is in fact healthy — the endpoint is
    POST-only. A liveness check that looks like an error is a bad one."""
    import json as _json
    import threading as _th
    from http.server import HTTPServer as _S

    import tools.stub_services as ss

    for role, path in (('voice', '/interpret'), ('rag', '/answer')):
        httpd = _S(('127.0.0.1', 0), ss.make_handler(role, False))
        _th.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = 'http://127.0.0.1:%d%s' % (httpd.server_address[1], path)
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as r:
                assert r.status == 200
                body = _json.loads(r.read().decode())
            assert body['stub'] is True and body['role'] == role
            assert body['endpoint'] == f'POST {path}'
        finally:
            httpd.shutdown(); httpd.server_close()


def test_a_rag_failure_sentence_is_not_reported_as_an_answer(monkeypatch):
    """Component 3 answers a failure with a SENTENCE — a non-empty string the
    user can hear — and `ok: false`. Ignoring that flag made every failure
    look like a successful answer. Caught in a live run, not by a review."""
    fail = {'ok': False, 'intent': 'ASK',
            'answer_si': 'පිළිතුර ලබාගැනීමට නොහැකි විය.',
            'speakable_text': 'පිළිතුර ලබාගැනීමට නොහැකි විය.',
            'retrieved_sources': [], 'notes': ['generation failed: HTTP 401']}
    with _Server(fail) as s:
        monkeypatch.setattr(l6gen, 'RAG_URL', s.url)
        out = l6gen.answer(_doc(), GOOD_VOICE, mode='http')
    assert out['ok'] is False
    assert out['speakable'], 'the user must still hear something'
    assert any('401' in w for w in out['warnings'])


def test_service_notes_reach_the_warnings(monkeypatch):
    ok = dict(RAG_OK, ok=True, notes=['remembered 3 chunk(s) from this article'])
    with _Server(ok) as s:
        monkeypatch.setattr(l6gen, 'RAG_URL', s.url)
        out = l6gen.answer(_doc(), GOOD_VOICE, mode='http')
    assert out['ok'] is True
    assert any('remembered' in w for w in out['warnings'])

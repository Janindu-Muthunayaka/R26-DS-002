"""
Stand-ins for Components 3 and 4, so the `http` path can be exercised before
either component is wrapped.

    python tools/stub_services.py --role voice --port 8101
    python tools/stub_services.py --role rag   --port 8102 --echo

WHAT THIS IS FOR. The `http` branch in layers/l0_voice and layers/l6_generator
is the branch that will run at the viva and the branch that has no way to be
tested until two other people deliver. This lets it be tested today.

WHAT THIS IS NOT. It does not detect intent and it does not retrieve anything.
Every reply is marked `"stub": true` and the reader turns that into a warning.

Stdlib only — no FastAPI — so it starts instantly and cannot drag a dependency
into this system's pinned environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_SYSTEM = Path(__file__).resolve().parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

# ONE source of truth for the placeholder routing. The built-in stub in
# layers/l0_voice already decides route and intent from literal command words;
# duplicating that here would let the two drift apart.
from layers.l0_voice.voice import _stub as _route_like_the_builtin_stub

VOICE_REPLY = {
    'route': 'GENERATE', 'intent': 'SUMMARIZE',
    'english_translation': '[stub] not translated',
    'style_class': 'Detailed', 'prompt_modifier': '',
    'personalization_flags': {}, 'retrieved_chunk_id': None,
    'correction_applied': None, 'stub': True,
}

RAG_ANSWER = 'මෙය පරීක්ෂණ පිළිතුරකි. තොරතුරු ලබාගැනීමේ සේවාව තවම ක්‍රියාත්මක නොවේ.'


def make_handler(role: str, echo: bool):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """A browser address bar sends GET. Without this, opening
            http://127.0.0.1:8101/interpret to check the service is alive
            returns `501 Unsupported method ('GET')` from http.server — which
            looks like a fault and is not one: the endpoint is POST-only.

            A liveness check that reads as an error is a bad liveness check.
            """
            path = '/interpret' if role == 'voice' else '/answer'
            self._send({
                'ok': True, 'stub': True, 'role': role,
                'this_is': 'a STAND-IN for ' + (
                    'Component 4 (voice interaction, Bumal)' if role == 'voice'
                    else 'Component 3 (RAG, Nadee)'),
                'endpoint': f'POST {path}',
                'note': 'Seeing this page means the service is UP. The '
                        'endpoint only accepts POST, so a browser GET is not '
                        'a fault.',
                'test_it_with': ['python tools\\try_ask.py --frames work\\<jobid>',
                                 'open http://127.0.0.1:8000/debug'],
                'nothing_here_is_a_result': 'Every reply carries stub:true, '
                    'and the reader turns that into a warning, so no '
                    'transcript can be read as Component 3 or 4 output.',
            })

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(n).decode('utf-8', 'replace')
            try:
                req = json.loads(raw) if raw else {}
            except ValueError:
                return self._send({'error': 'bad json'}, 400)

            if echo:
                print(json.dumps(req, ensure_ascii=False)[:600], flush=True)

            if role == 'voice' and self.path.rstrip('/') == '/interpret':
                out = dict(VOICE_REPLY)
                out.update(_route_like_the_builtin_stub(req.get('text', '')))
                out['english_translation'] = f"[stub] {req.get('text', '')}"
                out['stub'] = True
                out.pop('source', None)   # the client stamps its own
                return self._send(out)

            if role == 'rag' and self.path.rstrip('/') == '/answer':
                ocr = req.get('ocr') or {}
                chars = len(ocr.get('corrected_text') or '')
                return self._send({
                    'ok': True,
                    'intent': (req.get('voice') or {}).get('intent', ''),
                    'answer_si': RAG_ANSWER,
                    'speakable_text': RAG_ANSWER,
                    'retrieved_sources': [
                        {'source_type': 'ocr_current',
                         'chunk_id': 'chunk_ocr_current', 'chars': chars}],
                    'stub': True,
                })

            self._send({'error': f'no route {self.path}'}, 404)

        def log_message(self, *a):        # quiet unless --echo
            if echo:
                super().log_message(*a)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=('voice', 'rag'), required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--echo', action='store_true',
                    help='print each request body — use this to SEE the '
                         'payload Component 3 will actually receive')
    a = ap.parse_args()
    srv = HTTPServer((a.host, a.port), make_handler(a.role, a.echo))
    path = '/interpret' if a.role == 'voice' else '/answer'
    print(f'stub {a.role} on http://{a.host}:{a.port}{path}  (ctrl-c to stop)')
    print(f'  POST {path} is the real endpoint; opening it in a browser '
          f'(a GET) now\n  returns a status page rather than a 501.')
    srv.serve_forever()


if __name__ == '__main__':
    main()

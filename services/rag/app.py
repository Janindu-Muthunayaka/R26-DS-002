"""
svc-rag — Component 3 as a service.

    python services/rag/app.py --port 8102

Contract (docs/INTEGRATION_CONTRACT.md §3.2):

    POST /answer   {"ocr": <Layer 5 payload>, "voice": <Layer 0 output>}
              ->   {"ok", "intent", "answer_si", "retrieved_sources",
                    "speakable_text", "notes"}

Also: GET / (status page — a browser GET must never look like a fault),
GET /health, GET /stats, POST /ingest, POST /forget.

RUNS IN ITS OWN PROCESS AND ITS OWN VENV. It shares nothing with the reader
except HTTP and two stdlib-only modules (`core.llm`, `core.env`). Its
requirements are fastapi, uvicorn and numpy — no torch, no chroma, no
langchain. See store.py for why.

IT NEVER 500s ON A MISSING KEY. A service that will not answer is still a
service that must say why, in a field the reader can turn into a sentence.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SYSTEM = _REPO / 'system'
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

from fastapi import FastAPI                              # noqa: E402
from fastapi.responses import JSONResponse               # noqa: E402

import answer as A                                       # noqa: E402
from ingest import records_from_folder, records_from_text  # noqa: E402
from store import VectorStore, text_hash                 # noqa: E402
from core import llm                                     # noqa: E402

STORE_DIR = Path(os.getenv('RAG_STORE', str(_HERE / 'store')))
REMEMBER = os.getenv('RAG_REMEMBER_READS', '1') not in ('0', 'false', 'no')


def build(store: VectorStore) -> FastAPI:
    app = FastAPI(title='svc-rag · Component 3')

    @app.get('/')
    def index():
        return {
            'ok': True,
            'service': 'svc-rag (Component 3 — retrieval and generation)',
            'endpoint': 'POST /answer',
            'note': 'A browser GET reaching this page means the service is UP. '
                    '/answer accepts POST only.',
            'store': store.stats(),
            'remember_reads': REMEMBER,
        }

    @app.get('/health')
    def health():
        ok, why = llm.available()
        return {'ok': True, 'llm_configured': ok, 'llm_reason': why,
                'chat_model': llm.chat_model() or '(unset)',
                'embed_model': llm.embed_model() or '(unset)',
                'key': 'set' if llm.key() else 'MISSING',
                'chunks': len(store.chunks)}

    @app.get('/stats')
    def stats():
        return store.stats()

    @app.post('/ingest')
    def ingest(req: dict):
        """Seed the store from a folder of .txt/.jsonl, or from raw texts."""
        records = []
        folder = (req or {}).get('folder')
        if folder:
            records += records_from_folder(Path(folder))
        for i, text in enumerate((req or {}).get('texts') or []):
            records += records_from_text(text, 'article',
                                         text_hash(text) or str(i))
        if not records:
            return {'ok': False, 'added': 0,
                    'error': 'nothing to ingest — pass folder or texts'}
        added, reason = A.index_records(store, records)
        return {'ok': not reason, 'added': added,
                'skipped_as_known': len(records) - added,
                'error': reason or None, 'store': store.stats()}

    @app.post('/forget')
    def forget(req: dict):
        kind = (req or {}).get('source_type') or 'read'
        removed = store.delete_where('source_type', kind)
        store.save()
        return {'ok': True, 'removed': removed, 'store': store.stats()}

    @app.post('/answer')
    def answer(req: dict):
        req = req or {}
        try:
            out = A.run(store, req.get('ocr'), req.get('voice'),
                        remember=REMEMBER)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Still the contract shape. The reader turns speakable_text into
            # something the user hears; a 500 with a stack trace it cannot.
            return JSONResponse({
                'ok': False,
                'intent': (req.get('voice') or {}).get('intent', ''),
                'answer_si': A.FAILED, 'retrieved_sources': [],
                'speakable_text': A.FAILED,
                'notes': [f'{type(e).__name__}: {e}']}, status_code=200)
        for n in out.get('notes') or []:
            print(f'[rag] {n}')
        return out

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8102)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--store', default=None)
    ap.add_argument('--seed', default=None,
                    help='folder of .txt/.jsonl to index at startup')
    a = ap.parse_args()

    store = VectorStore(Path(a.store) if a.store else STORE_DIR)
    ok, why = llm.available()
    print(f'svc-rag on http://{a.host}:{a.port}/answer')
    print(f'  store   : {store.stats()}')
    print(f'  llm     : {"ready" if ok else "NOT CONFIGURED — " + why}')
    print(f'  models  : chat={llm.chat_model() or "(unset)"} '
          f'embed={llm.embed_model() or "(unset)"}')
    if a.seed:
        recs = records_from_folder(Path(a.seed))
        added, reason = A.index_records(store, recs)
        print(f'  seeded  : {added} chunk(s) from {a.seed}'
              + (f' — {reason}' if reason else ''))

    import uvicorn
    uvicorn.run(build(store), host=a.host, port=a.port, log_level='info')


if __name__ == '__main__':
    main()

"""
FastAPI server: the phone endpoint, plus a browser page for debugging.

    python -m app.server --root "E:/RP/corpus/Sinhala_OCR_Correction_v2" --no-layout

THE CONTRACT, and why it changed on 21 Aug 2026
-----------------------------------------------
Until today `POST /capture` returned JSON while `ReaderApi.kt` declared the
response as `ResponseBody` — raw audio bytes to hand to MediaPlayer. The app
worked against `backend/stub_server.py` (which returns `reply.wav`) and would
have broken against this server. Nothing in either codebase said so.

It now returns JSON carrying TEXT, and the phone speaks it with its own TTS:

  * `l6_speech.speak()` returns None. That layer is Bumal's and is a stub, so
    there is no audio to send. Blocking the demo on it was not acceptable.
  * The Android build record section 14 argues for text anyway — on-device TTS
    is lower latency, sends far less data, and speaks better Sinhala than an
    offline server-side engine.

`audio_url` remains in the response and `GET /audio/{job}` remains routed, so
when Layer 6 is delivered the phone can prefer audio without another API
change.

RESPONSE SHAPE (stable — the app depends on these names)

    {
      "ok": true,
      "title": "",              # EMPTY until Layer 4A is delivered
      "body": "…",              # corrected Sinhala, spoken by the phone
      "warnings": ["…"],        # spoken to the user, e.g. "move closer"
      "n_articles": 1,
      "audio_url": null,
      "timings": {...},
      "job": "a1b2c3d4"
    }

On failure: HTTP 4xx/5xx with `{"ok": false, "error": "…"}`. The phone speaks
the error rather than failing silently, so the shape is the same either way.
"""
import argparse
import os
import re
import sys
import uuid
from pathlib import Path

# Run either way. `python -m app.server` puts system/ on sys.path; `python
# app\server.py` puts system/app/ on it instead and every `from core...`
# below fails with ModuleNotFoundError. Every tool in tools/ already does
# this; the server did not, so the two disagreed about how to be started.
_SYSTEM = Path(__file__).resolve().parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import WORK_DIR, HOST, PORT
from core.imaging import imdecode_upright
from layers.l6_speech import speech as l6


def _document_to_reply(doc, job):
    """Flatten a Document into what the phone speaks.

    Articles are joined in reading order. `l5_assemble` has already dropped
    rejected ones and recorded why in `warnings`, which the phone reads out —
    a blind user must be told "article 2 skipped: too far" rather than simply
    receiving less text than they expected.
    """
    titles, bodies = [], []
    for art in doc.articles:
        if art.title.strip():
            titles.append(art.title.strip())
        text = (art.body or art.body_raw or '').strip()
        if text:
            bodies.append(text)
    return {
        'ok': bool(bodies),
        'job': job,
        'title': ' '.join(titles),
        'body': '\n\n'.join(bodies),
        'warnings': list(doc.warnings),
        'n_articles': len(doc.articles),
        'audio_url': None,
        'timings': doc.timings,
    }


def build(pipeline, web_dir: Path):
    app = FastAPI(title='Sinhala Reader')
    app.add_middleware(CORSMiddleware, allow_origins=['*'],
                       allow_methods=['*'], allow_headers=['*'])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    app.mount('/work', StaticFiles(directory=str(WORK_DIR)), name='work')

    def _page(name):
        p = web_dir / name
        return (p.read_text(encoding='utf-8') if p.exists()
                else f'<h1>{name} missing</h1><p>expected at {p}</p>')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return _page('reader.html')

    @app.get('/debug', response_class=HTMLResponse)
    def debug():
        return _page('debug.html')

    @app.get('/health')
    def health():
        return {'ok': True, 'device': getattr(pipeline, 'dev', '?'),
                'root': os.getenv('SINHALA_ROOT', '')}

    # The phone stamps its own glyph estimate and the sharpness at the instant
    # the shutter fired into every filename it uploads:
    #
    #     burst_20260826_142530_g27_s1603_1.jpg
    #
    # Until 26 Aug 2026 this server renamed every upload to f0/f1/f2.jpg and
    # that pairing was destroyed on arrival - the only place the app's estimate
    # could be compared against the metric it is supposed to predict was the
    # phone's own storage, reachable only over USB.
    #
    # It matters because the estimate is NOT the same quantity as glyph_p75.
    # Measured (Android_Capture_Guidance_Calibration.md sec 5): the app read
    # 26-29 on captures that measured 29-34. Carrying the stamp through means
    # every capture ever taken is a calibration point, for free.
    #
    # Kept as a SUFFIX so the f<i> prefix, and therefore frame order, is
    # unchanged. Uploads without a stamp (the debug page, tests) are named
    # exactly as before.
    _STAMP = re.compile(r'_g(\d+)_s(\d+)_', re.I)

    def _stamp_of(filename):
        m = _STAMP.search(filename or '')
        return f'_g{m.group(1)}_s{m.group(2)}' if m else ''

    @app.post('/capture')
    async def capture(frames: list[UploadFile] = File(...)):
        job = uuid.uuid4().hex[:8]
        sess = WORK_DIR / job
        sess.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, f in enumerate(frames):
            data = await f.read()
            # exif_transpose ON ARRIVAL. CameraX writes rotation into EXIF and
            # does not rotate pixels. The file written here is already upright
            # and carries no orientation tag, so nothing downstream repeats it.
            im = imdecode_upright(data)
            if im is None:
                continue
            p = sess / f'f{i}{_stamp_of(f.filename)}.jpg'
            cv2.imwrite(str(p), im)
            paths.append(str(p))

        if not paths:
            return JSONResponse(
                {'ok': False, 'job': job, 'error': 'no decodable frames',
                 'title': '', 'body': '', 'warnings': [], 'n_articles': 0},
                status_code=400)

        try:
            doc = pipeline.run(paths)
            reply = _document_to_reply(doc, job)
            reply['audio_url'] = l6.speak(doc)      # None until Layer 6 lands
            if not reply['ok']:
                reply['error'] = 'nothing could be read from these frames'
            return reply
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                {'ok': False, 'job': job, 'error': f'{type(e).__name__}: {e}',
                 'title': '', 'body': '', 'warnings': [], 'n_articles': 0},
                status_code=500)

    @app.get('/audio/{job}')
    def audio(job: str):
        """Reserved for Layer 6. Routed now so the phone's fallback path is
        exercised rather than discovered on the day it starts returning data."""
        for ext in ('.wav', '.mp3', '.m4a'):
            p = WORK_DIR / job / f'speech{ext}'
            if p.exists():
                return FileResponse(str(p))
        return JSONResponse({'ok': False,
                             'error': 'no audio — Layer 6 (TTS) is a stub'},
                            status_code=404)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--cert', default=None)
    ap.add_argument('--key', default=None)
    ap.add_argument('--no-layout', action='store_true')
    a = ap.parse_args()
    if a.root:
        # Setting the env var is NOT enough: core.config is imported at the top
        # of this module, so PROJECT_ROOT is already frozen and --root was
        # being silently ignored. set_root() recomputes the dependent paths,
        # and must run BEFORE app.pipeline is imported because that module
        # binds YOLO_WEIGHTS/MT5_PLAIN at its own import time.
        from core import config as _cfg
        _cfg.set_root(a.root)
        print(f'root: {_cfg.PROJECT_ROOT}')

    from app.pipeline import Pipeline
    pipe = Pipeline(use_layout=not a.no_layout)
    app = build(pipe, Path(__file__).resolve().parent.parent / 'web')

    print(f'\nphone should POST to  http://<this-machine>:{a.port}/capture')
    print(f'browser test page      http://127.0.0.1:{a.port}/')
    print('for the viva, prefer:  adb reverse tcp:8000 tcp:8000\n')

    import uvicorn
    kw = {'ssl_certfile': a.cert, 'ssl_keyfile': a.key} if a.cert and a.key else {}
    uvicorn.run(app, host=a.host, port=a.port, **kw)


if __name__ == '__main__':
    main()

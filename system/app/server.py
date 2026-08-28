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

  * `l8b_speech.speak()` returns None. That layer is Bumal's and is a stub, so
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
import time
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

from core.config import (WORK_DIR, HOST, PORT, SESSION_MAX, SESSION_TTL_S,
                         VOICE_USER_ID)
from core import quality
from core.imaging import imdecode_upright
from core.schemas import Answer, Question
from core.session import SessionStore
from layers.l0_voice import voice as l0
from layers.l5_assemble.payload import article_text
from layers.l6_generator import generate as l6gen
from layers.l8b_speech import speech as l6


# Spoken to the user when `/ask` cannot proceed, or when no single article
# could be identified. NOT written by a native speaker — have them checked.
SI_NOTHING_READ = 'කිසිවක් තවම කියවා නොමැත. පුවත්පත දෙසට යොමු කරන්න.'
SI_ASK_FAILED = 'පිළිතුර ලබාගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න.'
SI_MOVE_CLOSER = 'ලිපිය හඳුනාගත නොහැකි විය. ටිකක් ළං වී නැවත උත්සාහ කරන්න.'


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
        text = article_text(art)          # one definition, see l5/payload.py
        if text:
            bodies.append(text)

    body = '\n\n'.join(bodies)
    warnings = list(doc.warnings)

    # THE READABILITY GATE. A capture that goes wrong does not fail:
    # Tesseract returns something, mT5 corrects that something, and the phone
    # reads it aloud in the same confident voice it uses for real news. A
    # sighted developer sees garbage on a screen; a blind user cannot.
    #
    # `fatal` separates SHATTERED from merely SHORT. A six-word news brief is
    # a real thing a newspaper prints and is read as-is with a warning; text
    # that is fragments, Latin, or undecodable bytes is replaced by a sentence
    # asking for another photograph.
    verdict, spoken, measures = quality.verdict_for_user(body)
    if body and verdict == 'unreadable':
        warnings.append('unreadable: ' + '; '.join(measures['reasons']))
        body = spoken
    elif body and verdict == 'poor':
        warnings.append('poor quality: ' + '; '.join(measures['reasons']))
        body = spoken + '\n\n' + body
    elif body and verdict == 'short':
        warnings.append('short: ' + '; '.join(measures['reasons']))

    return {
        'ok': bool(bodies),
        'job': job,
        'title': ' '.join(titles),
        'body': body,
        'warnings': warnings,
        'n_articles': len(doc.articles),
        'audio_url': None,
        'timings': doc.timings,
        # Diagnostics. The phone ignores fields it does not read; the debug
        # page shows them, and they are what a bug report needs.
        'quality': measures,
    }


def build(pipeline, web_dir: Path):
    app = FastAPI(title='Sinhala Reader')
    app.add_middleware(CORSMiddleware, allow_origins=['*'],
                       allow_methods=['*'], allow_headers=['*'])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    app.mount('/work', StaticFiles(directory=str(WORK_DIR)), name='work')

    # The system's only memory. See core/session.py for why it is in memory
    # and why that is stated as a limitation rather than papered over.
    sessions = SessionStore(ttl_s=SESSION_TTL_S, max_items=SESSION_MAX)
    app.state.sessions = sessions

    def _page(name):
        p = web_dir / name
        return (p.read_text(encoding='utf-8') if p.exists()
                else f'<h1>{name} missing</h1><p>expected at {p}</p>')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return _page('landing.html')

    @app.get('/reader', response_class=HTMLResponse)
    def reader():
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
                # A frame where no single article could be identified is a
                # DIFFERENT failure from one where nothing was legible, and
                # the user's next action differs too. Speak the instruction
                # rather than a generic "could not read", which sends them
                # nowhere.
                move = next((w for w in reply['warnings']
                             if 'move a little closer' in w), None)
                reply['error'] = move or \
                    'nothing could be read from these frames'
                if move:
                    reply['body'] = SI_MOVE_CLOSER
            else:
                # ONLY on success. A job with nothing in it is not worth
                # remembering, and "read that again" on an empty article
                # should say so rather than replay silence.
                sessions.put(job, doc)
            return reply
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                {'ok': False, 'job': job, 'error': f'{type(e).__name__}: {e}',
                 'title': '', 'body': '', 'warnings': [], 'n_articles': 0},
                status_code=500)

    @app.post('/ask')
    def ask(q: Question):
        """The follow-up question.

        Never 500s because a component is down: an unreachable service becomes
        `ok: false` plus a speakable sentence, because a stack trace cannot be
        spoken and silence tells a blind user nothing.
        """
        t0 = time.time()
        doc = sessions.get(q.job)
        if doc is None:
            # Miss and expiry are the same answer on purpose — the user's next
            # action is identical either way: capture again.
            return JSONResponse(Answer(
                ok=False, job=q.job, route='LOCAL', intent='',
                speakable=SI_NOTHING_READ,
                error='no article in session for this job',
            ).model_dump(), status_code=404)

        try:
            voice = l0.interpret(q.text, user_id=q.user_id or VOICE_USER_ID)
            t_voice = round(time.time() - t0, 2)

            t1 = time.time()
            # The cursor is how far "next" has walked through the article. It
            # lives in the session beside the Document so it expires with it.
            res = l6gen.answer(doc, voice, cursor=sessions.cursor(q.job))
            if res.get('cursor') is not None:
                sessions.set_cursor(q.job, res['cursor'])
            t_gen = round(time.time() - t1, 2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(Answer(
                ok=False, job=q.job, speakable=SI_ASK_FAILED,
                error=f'{type(e).__name__}: {e}',
            ).model_dump(), status_code=500)

        for w in res['warnings']:
            print(f'[ask {q.job}] {w}')

        return Answer(
            ok=res['ok'], job=q.job, route=res['route'], intent=res['intent'],
            speakable=res['speakable'], answer_si=res['answer_si'],
            sources=res['sources'], warnings=res['warnings'],
            timings={'voice': t_voice, 'generate': t_gen,
                     'total': round(time.time() - t0, 2)},
        ).model_dump()

    @app.get('/session/{job}')
    def session(job: str):
        """Diagnostics. Confirms an article is held without re-running OCR."""
        doc = sessions.get(job)
        if doc is None:
            return JSONResponse({'ok': False, 'job': job,
                                 'error': 'not in session'}, status_code=404)
        return {'ok': True, 'job': job,
                'age_s': round(sessions.age_of(job) or 0.0, 1),
                'n_articles': len(doc.articles),
                'chars': sum(len(article_text(a)) for a in doc.articles),
                'warnings': doc.warnings,
                'live_jobs': len(sessions)}

    @app.get('/document/{job}')
    def document(job: str):
        """The FULL Document for a job — every article, `body_raw` included.

        Diagnostics, and the reason the debug page works again. `/capture`
        returns the flat reply the phone reads and has done since 21 Aug 2026;
        web/debug.html was still reading a `document` key that no longer
        existed, so it had been showing "no articles" for every upload since.

        Fixed here rather than by putting the Document back into `/capture`:
        the phone does not need it, and a few extra KB on every capture over
        WiFi is a cost paid by the one client that has no use for it.
        """
        doc = sessions.get(job)
        if doc is None:
            return JSONResponse({'ok': False, 'job': job,
                                 'error': 'not in session'}, status_code=404)
        return {'ok': True, 'job': job, 'document': doc.model_dump()}

    @app.get('/latest')
    def latest():
        """Returns the most recent job data for the dashboard."""
        jobs = sessions.jobs()
        if not jobs:
            return {'ok': False, 'error': 'no active jobs'}
        doc = sessions.get(jobs[0])
        return {'ok': True, 'job': jobs[0], 'document': doc.model_dump()}

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
    uvicorn.run(app, host=a.host, port=a.port, access_log=False, **kw)


if __name__ == '__main__':
    main()

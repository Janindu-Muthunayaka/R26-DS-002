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
from fastapi.concurrency import run_in_threadpool

from core.config import (WORK_DIR, HOST, PORT, SESSION_MAX, SESSION_TTL_S,
                         VOICE_USER_ID)
from core import quality
from core.imaging import imdecode_upright
from core.schemas import Answer, Question, Document
from pydantic import BaseModel


class NewUser(BaseModel):
    base_weights: dict                 # {Simple, Detailed, StepByStep}, % or fractions
    description: str = ''
    name: str = ''
    tag: str = ''


class ActiveUser(BaseModel):
    user_id: str


class Regenerate(BaseModel):
    job: str
    index: int                         # 1-based turn index
    user_id: str
from core.session import SessionStore
from layers.l0_voice import voice as l0
from layers.l5_assemble.payload import article_text
from layers.l6_generator import generate as l6gen
from layers.l6_generator import turns as l6turns
from layers.l4a_title import mat as l4a_mat
from layers.l7_personalization.delivery.planner import (
    format_text_with_tts_pauses, pause_level as _pause_level)
from layers.l8b_speech import speech as l6


# Spoken to the user when `/ask` cannot proceed, or when no single article
# could be identified. NOT written by a native speaker — have them checked.
SI_NOTHING_READ = 'කිසිවක් තවම කියවා නොමැත. පුවත්පත දෙසට යොමු කරන්න.'
SI_ASK_FAILED = 'පිළිතුර ලබාගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න.'
SI_MOVE_CLOSER = 'ලිපිය හඳුනාගත නොහැකි විය. ටිකක් ළං වී නැවත උත්සාහ කරන්න.'

# The user picked in the web UI survives a restart: the next session starts
# with whoever was used last instead of falling back to VOICE_USER_ID.
ACTIVE_USER_FILE = WORK_DIR / 'active_user.txt'


def _load_active_user():
    try:
        uid = ACTIVE_USER_FILE.read_text(encoding='utf-8').strip()
        return uid or VOICE_USER_ID
    except OSError:
        return VOICE_USER_ID


def _save_active_user(uid):
    try:
        ACTIVE_USER_FILE.write_text(uid, encoding='utf-8')
    except OSError:
        pass                                  # still active for this run


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


def _user_meta(user_id):
    """Snapshot of who answered: name, tag, description, base weights.
    Stored on each answer so the compare view still shows the user as they
    were, even if the registry changes later. Best effort, never raises."""
    meta = {'user_id': user_id, 'name': user_id, 'tag': '', 'description': '',
            'base_weights': None}
    try:
        reply, _ = l0.list_users()
        for u in (reply or {}).get('users', []):
            if u.get('user_id') == user_id:
                meta.update({k: u.get(k) for k in ('name', 'tag', 'description', 'base_weights')})
                break
    except Exception:
        pass
    return meta


def _speakable(res, voice):
    """Natural TTS pause formatting, as /ask does."""
    out = res['speakable']
    if out:
        try:
            out = format_text_with_tts_pauses(out, style=voice.get('style_class', 'Detailed'))
        except Exception:
            pass
    return out


def _format_turn(res):
    """(speakable, pause_level) for one policy result. The style gives the
    base pause, the user's pause_scale moves it (TURN_CONDITIONS.xlsx G6)."""
    text = res.get('speakable') or ''
    d = res.get('delivery') or {}
    level = _pause_level(res.get('style_class') or None, d.get('pause_scale', 1.0),
                         base=res.get('pause_base'))
    if not text:
        return '', level
    try:
        return format_text_with_tts_pauses(text, level=level), level
    except Exception:
        return text, level


def _rag_history(generations):
    """Chat history for RAG: generated answers only, each under the question
    that was actually asked. A local turn ("speak fast" -> "හරි, ...") or a
    style command ("සරලව කියන්න") would otherwise be read as a question by
    the query rewriter."""
    out = []
    for g in generations or []:
        if g.get('route') != 'GENERATE' or g.get('kind') == 'error':
            continue
        q = g.get('effective_query') or {}
        question = q.get('si') or g.get('query') or ''
        out.append({'question': question,
                    'standalone_question': g.get('standalone_query') or q.get('en') or question,
                    'answer': g.get('answer') or g.get('answer_si') or ''})
    return out


# The fields of a turn that belong to ONE generated answer. A re-generation
# replaces these; the question and its index stay.
VERSION_FIELDS = ('user_id', 'user_meta', 'answer', 'answer_si', 'route', 'intent',
                  'prompt_modifier', 'style_class', 'style_source', 'user_profile',
                  'learned', 'english_translation', 'correction_applied',
                  'warnings', 'timings')


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
    # The user picked in the web UI. Used whenever a request does not name
    # one: the phone's /capture never does, and /ask may not.
    app.state.active_user = _load_active_user()

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
            # An empty or broken upload is skipped (it was an HTTP 500).
            try:
                im = imdecode_upright(data) if data else None
            except Exception:
                im = None
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
            doc = Document()
            sessions.put(job, doc)
            doc = await run_in_threadpool(pipeline.run, paths, doc=doc)
            reply = _document_to_reply(doc, job)
            
            if reply['ok']:
                # Turn 1 is GENERIC: the same fixed command for everyone (how
                # many articles, and each article's number and title). It is
                # not styled by any user's weights and changes none: no
                # profile is read, nothing is logged or learned, and no
                # weights are stored on the turn.
                scan_uid = app.state.active_user
                first_scan_voice = {
                    'route': 'GENERATE',
                    'intent': 'FIRST_SCAN',
                    'english_translation': 'Tell the number of articles detected,and tell the title of each article along with its article no.',
                    'style_class': '',
                    'prompt_modifier': '',
                    'personalization_flags': {},
                    'sinhala_input': 'හඳුනාගත් ලිපි සංඛ්‍යාව සහ එක් එක් ලිපියේ ශීර්ෂය එහි ලිපි අංකය සමඟ පවසන්න.',
                    'source': 'component4'
                }
                
                t0_l6 = time.time()
                res = l6gen.answer(doc, first_scan_voice)
                t_l6 = round(time.time() - t0_l6, 2)
                
                if res.get('speakable'):
                    reply['body'] = res['speakable']
                elif getattr(doc, 'tts_text', None):
                    reply['body'] = doc.tts_text
                else:
                    tts_text = l6.get_tts_text(doc)
                    if tts_text:
                        reply['body'] = tts_text
                # The phone speaks `title` before `body` (title-first design).
                # Turn 1 already says every title, so the raw OCR headlines
                # left here would be spoken first as a stale prefix.
                reply['title'] = ''

                if not hasattr(doc, 'generations') or doc.generations is None:
                    doc.generations = []
                    
                turn_info = {
                    'index': len(doc.generations) + 1,
                    'query': first_scan_voice['sinhala_input'],
                    'user_id': scan_uid,
                    'user_meta': _user_meta(scan_uid),
                    'style_source': 'fixed',
                    'user_profile': None,
                    'learned': False,
                    'answer': res['speakable'] or res['answer_si'],
                    'answer_si': res['answer_si'],
                    'route': res['route'],
                    'intent': res['intent'],
                    'prompt_modifier': first_scan_voice.get('prompt_modifier', ''),
                    'style_class': first_scan_voice.get('style_class', ''),
                    'english_translation': first_scan_voice.get('english_translation', ''),
                    'correction_applied': None,
                    'warnings': res['warnings'],
                    'timings': {
                        'voice': 0.0,
                        'generate': t_l6,
                        'total': t_l6
                    }
                }
                doc.generations.append(turn_info)
                # The title list is the first thing spoken: "repeat" right
                # after a capture replays it (TURN_CONDITIONS.xlsx R1).
                l6turns.seed_first_scan(sessions.state(job), res.get('speakable') or '',
                                        style=first_scan_voice.get('style_class'))

                ans_dict = Answer(
                    ok=res['ok'], job=job, route=res['route'], intent=res['intent'],
                    speakable=res['speakable'], answer_si=res['answer_si'],
                    sources=res['sources'],
                    english_translation=first_scan_voice.get('english_translation', ''),
                    style_class=first_scan_voice.get('style_class', ''),
                    correction_applied=None,
                    warnings=res['warnings'],
                    timings={'voice': 0.0, 'generate': t_l6, 'total': t_l6},
                ).model_dump()
                
                sessions.set_answer(job, ans_dict)
                sessions.put(job, doc)
            else:
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
                sessions._items.pop(job, None)
            reply['audio_url'] = l6.speak(doc)      # None until Layer 6 lands
            if reply['ok']:
                # Show only: the MAT skeleton drawn on each binarized title,
                # in an idle-priority process, AFTER this reply is built.
                # Returns at once; the scan's time is unchanged.
                try:
                    l4a_mat.schedule(doc, sess)
                except Exception as e:
                    print(f'[title MAT] not scheduled: {e}')
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

        state = sessions.state(q.job)
        try:
            # Component 4 needs to know whether something was just spoken (its
            # REPEAT rule); only this session knows.
            voice = l0.interpret(q.text, user_id=q.user_id or app.state.active_user,
                                 last_spoken_ctx=l6turns.context(state))
            voice.setdefault('sinhala_input', q.text)
            voice.setdefault('session_id', q.job)
            voice['chat_history'] = _rag_history(getattr(doc, 'generations', None))
            t_voice = round(time.time() - t0, 2)

            t1 = time.time()
            # One decision per turn (layers/l6_generator/turns.py,
            # TURN_CONDITIONS.xlsx): only a turn that needs new words reaches
            # the generator. The cursor is how far "next" has walked through
            # the current article; it lives in the session beside the Document.
            res = l6turns.run(doc, voice, state, cursor=sessions.cursor(q.job))
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

        speakable_out, level = _format_turn(res)
        delivery = res.get('delivery') or {}
        rate = float(delivery.get('rate', voice.get('speed', 1.0)))
        learned = bool(voice.get('learned', False)) and res.get('route') == 'GENERATE'

        ans_dict = Answer(
            ok=res['ok'], job=q.job, route=res['route'], intent=res['intent'],
            speakable=speakable_out, answer_si=res['answer_si'],
            sources=res['sources'],
            english_translation=voice.get('english_translation', ''),
            style_class=res.get('style_class') or voice.get('style_class', ''),
            correction_applied=voice.get('correction_applied'),
            prompt_modifier=res.get('prompt_modifier') or voice.get('prompt_modifier', ''),
            style_source=voice.get('style_source'),
            user_profile=voice.get('user_profile'),
            learned=learned,
            speed=rate, speech_rate=rate,
            pause_scale=float(delivery.get('pause_scale', 1.0)),
            pause_level=level,
            chunk_size=int(delivery.get('chunk_size', 3)),
            action=res.get('action', ''),
            warnings=res['warnings'],
            timings={'voice': t_voice, 'generate': t_gen,
                     'total': round(time.time() - t0, 2)},
        ).model_dump()

        if not hasattr(doc, 'generations') or doc.generations is None:
            doc.generations = []
        uid = q.user_id or app.state.active_user
        turn_info = {
            'index': len(doc.generations) + 1,
            'query': q.text,
            'effective_query': res.get('question'),
            'user_id': uid,
            'user_meta': _user_meta(uid),
            'answer': speakable_out or res['answer_si'],
            'answer_si': res['answer_si'],
            'route': res['route'],
            'intent': res['intent'],
            'action': res.get('action', ''),
            'kind': res.get('kind', ''),
            'style_forced': bool(res.get('style_forced')),
            'prompt_modifier': ans_dict['prompt_modifier'],
            'style_class': ans_dict['style_class'],
            'style_source': voice.get('style_source'),
            'user_profile': voice.get('user_profile'),
            'learned': learned,
            'english_translation': voice.get('english_translation', ''),
            'correction_applied': voice.get('correction_applied'),
            'speed': rate,
            'pause_scale': ans_dict['pause_scale'],
            'pause_level': level,
            'warnings': res['warnings'],
            'timings': {
                'voice': t_voice,
                'generate': t_gen,
                'total': round(time.time() - t0, 2)
            }
        }
        doc.generations.append(turn_info)

        sessions.set_answer(q.job, ans_dict)
        sessions.put(q.job, doc)
        return ans_dict

    # -- Users (Component 4 registry, proxied) ------------------------------
    def _proxy(reply_reason):
        reply, reason = reply_reason
        if reply is None:
            code = (403 if 'HTTP 403' in reason else 404 if 'HTTP 404' in reason
                    else 400 if 'HTTP 400' in reason else 503)
            return JSONResponse({'ok': False, 'error': reason}, status_code=code)
        return {'ok': True, **reply}

    @app.get('/users')
    def users_list():
        return _proxy(l0.list_users())

    @app.post('/users')
    def users_add(u: NewUser):
        return _proxy(l0.add_user(u.base_weights, u.description, u.name, u.tag))

    @app.delete('/users/{user_id}')
    def users_delete(user_id: str):
        return _proxy(l0.delete_user(user_id))

    @app.post('/users/{user_id}/reset')
    def users_reset(user_id: str):
        return _proxy(l0.reset_user(user_id))

    @app.get('/active-user')
    def active_user_get():
        return {'ok': True, 'user_id': app.state.active_user}

    @app.post('/active-user')
    def active_user_set(a: ActiveUser):
        app.state.active_user = a.user_id or VOICE_USER_ID
        _save_active_user(app.state.active_user)
        return {'ok': True, 'user_id': app.state.active_user}

    @app.post('/regenerate')
    def regenerate(r: Regenerate):
        """Re-answer ONE earlier question as another user.

        Only that turn changes. Its previous answer is kept in
        `turn['versions']` (the compare view reads it); the new one becomes the
        turn's visible answer. Personalisation runs as a dry run, so the chosen
        user's profile learns nothing from a replay, and the RAG history is
        neither read past this turn nor appended to.
        """
        doc = sessions.get(r.job)
        if doc is None:
            return JSONResponse({'ok': False, 'error': 'not in session'}, status_code=404)
        gens = getattr(doc, 'generations', None) or []
        if not (1 <= r.index <= len(gens)):
            return JSONResponse({'ok': False, 'error': f'no turn {r.index}'}, status_code=404)
        turn = gens[r.index - 1]
        if turn.get('route') != 'GENERATE':
            # A local turn (speed change, repeat, reading, navigation) has no
            # generated answer; re-running it as a question is exactly how
            # "speak fast" reached RAG on 17 Oct 2026.
            return JSONResponse({'ok': False, 'error': 'this turn was answered locally '
                                 '(no generated answer to re-generate)'}, status_code=400)

        t0 = time.time()
        try:
            voice = l0.interpret(turn.get('query') or '', user_id=r.user_id, dry_run=True)
            voice.setdefault('sinhala_input', turn.get('query') or '')
            # A restyled turn ("සරලව කියන්න") re-asks the question it was
            # about, not the command; an explicit style stays explicit.
            eq = turn.get('effective_query') or {}
            if eq.get('si') or eq.get('en'):
                voice['sinhala_input'] = eq.get('si') or voice['sinhala_input']
                voice['english_translation'] = eq.get('en') or eq.get('si')
                voice['intent'] = 'ASK_QUESTION'
            if eq.get('article') is not None:
                voice['article_no'] = eq['article'] + 1
            voice['route'] = 'GENERATE'
            if turn.get('style_forced') and turn.get('style_class'):
                voice['style_class'] = turn['style_class']
                voice['prompt_modifier'] = turn.get('prompt_modifier') or voice.get('prompt_modifier', '')
            voice['session_id'] = r.job
            voice['chat_history'] = _rag_history(gens[:r.index - 1])
            voice['history_override'] = True
            voice['no_history_write'] = True
            t_voice = round(time.time() - t0, 2)
            t1 = time.time()
            res = l6gen.answer(doc, voice, cursor=sessions.cursor(r.job))
            t_gen = round(time.time() - t1, 2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({'ok': False, 'error': f'{type(e).__name__}: {e}'},
                                status_code=500)

        speakable_out = _speakable(res, voice)
        new_version = {
            'user_id': r.user_id,
            'user_meta': _user_meta(r.user_id),
            'answer': speakable_out or res['answer_si'],
            'answer_si': res['answer_si'],
            'route': res['route'],
            'intent': res['intent'],
            'prompt_modifier': voice.get('prompt_modifier', ''),
            'style_class': voice.get('style_class', ''),
            'style_source': voice.get('style_source'),
            'user_profile': voice.get('user_profile'),
            'learned': False,
            'english_translation': voice.get('english_translation', ''),
            'correction_applied': None,
            'warnings': res['warnings'],
            'timings': {'voice': t_voice, 'generate': t_gen,
                        'total': round(time.time() - t0, 2)},
            'regenerated': True,
        }
        versions = turn.setdefault('versions', [])
        if not versions:                         # keep the original answer
            versions.append({k: turn.get(k) for k in VERSION_FIELDS})
        versions.append(new_version)
        turn['active_version'] = len(versions) - 1
        turn.update({k: new_version.get(k) for k in VERSION_FIELDS})
        sessions.put(r.job, doc)
        return {'ok': True, 'job': r.job, 'turn': turn}

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
        jobs = sessions.jobs()
        if not jobs:
            return {'ok': False, 'error': 'no active jobs'}
        job = jobs[0]
        doc = sessions.get(job)
        
        reply = {
            'ok': True,
            'job': job,
            'document': doc.model_dump()
        }
        
        ans = sessions.get_answer(job)
        if ans:
            reply['document']['last_answer'] = ans
            
        return reply

    @app.get('/profile/{user_id}')
    def get_profile(user_id: str):
        return l0.get_profile(user_id)

    @app.get('/profile/{user_id}/history')
    def get_profile_history(user_id: str):
        """Complete weight history since the user was created or reset."""
        return l0.get_profile_history(user_id)

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

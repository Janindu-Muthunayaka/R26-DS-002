"""
End-to-end check of the conversation path, without a phone.

    python tools\try_ask.py --frames work\<jobid>
    python tools\try_ask.py --job <jobid> --ask "මේක සාරාංශ කරන්න"
    python tools\try_ask.py --job <jobid> --session

WHAT "WORKING" LOOKS LIKE with both services off (the defaults):

    වචන කීයද        -> LOCAL    LENGTH    "this article has 118 words"
    ශීර්ෂය මොකක්ද   -> LOCAL    TITLE     the headline, or an honest "not read"
    මොනවද මඟ හැරුණේ -> LOCAL    WARNINGS  what the capture skipped
    මුල සිට කියවන්න -> LOCAL    FIRST     part 1
    ඊළඟ             -> LOCAL    NEXT      the next part, never twice the same
    කලින් එක        -> LOCAL    PREVIOUS  the part before
    නැවත කියවන්න    -> LOCAL    REPEAT    the whole article again
    නවත්වන්න        -> LOCAL    STOP      speakable EMPTY, on purpose
    මේක සාරාංශ...   -> GENERATE ASK       ok=False, "service not available"

Only the last needs a service. The other eight are the correct
implementation, not placeholders: no network, no API key, no teammate.

That last line is not a failure of the loop either. It is the loop working:
nothing was invented, `answer_si` is empty, and the user still hears a
sentence.

Stdlib only - urllib and a hand-rolled multipart body.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

DEFAULT_QUESTIONS = [
    ('වචන කීයද', 'how many words'),
    ('ශීර්ෂය මොකක්ද', "what's the headline"),
    ('මොනවද මඟ හැරුණේ', 'what did I miss'),
    ('මුල සිට කියවන්න', 'read from the start'),
    ('ඊළඟ', 'the next part'),
    ('ඊළඟ', 'next again — must NOT repeat itself'),
    ('කලින් එක', 'the part before'),
    ('නැවත කියවන්න', 'the whole article again'),
    ('නවත්වන්න', 'stop'),
    ('මේක සාරාංශ කරන්න', 'summarise — the only one needing Component 3'),
]


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode('utf-8')), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode('utf-8')), e.code
        except Exception:
            return {}, e.code
    except Exception as e:
        print(f'  cannot reach {url}: {e}')
        sys.exit(2)


def _post_json(url, body):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method='POST',
        headers={'Content-Type': 'application/json; charset=utf-8'})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            return json.loads(r.read().decode('utf-8')), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode('utf-8')), e.code
        except Exception:
            return {}, e.code


def _post_files(url, paths):
    """Multipart by hand — every file under the field name `frames`, which is
    what FastAPI receives as a list and what the phone sends."""
    boundary = '----' + uuid.uuid4().hex
    body = bytearray()
    for p in paths:
        ctype = mimetypes.guess_type(p.name)[0] or 'image/jpeg'
        body += f'--{boundary}\r\n'.encode()
        body += (f'Content-Disposition: form-data; name="frames"; '
                 f'filename="{p.name}"\r\n').encode()
        body += f'Content-Type: {ctype}\r\n\r\n'.encode()
        body += p.read_bytes() + b'\r\n'
    body += f'--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        url, data=bytes(body), method='POST',
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode('utf-8')), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode('utf-8')), e.code
        except Exception:
            return {}, e.code


def _clip(s, n=110):
    s = (s or '').replace('\n', ' ⏎ ')
    return s if len(s) <= n else s[:n] + f' …(+{len(s) - n} chars)'


def show_ask(question, gloss, reply, code):
    print(f'\n  Q  {question}   ({gloss})' if gloss else f'\n  Q  {question}')
    if code == 404:
        print(f'     HTTP 404 — {reply.get("error", "")}')
        print(f'     speakable: {_clip(reply.get("speakable"))}')
        return
    print(f'     route={reply.get("route", "?"):9} '
          f'intent={reply.get("intent", "?"):10} ok={reply.get("ok")}')
    sp = reply.get('speakable') or ''
    if sp:
        print(f'     speakable: {_clip(sp)}')
    else:
        print('     speakable: (empty — correct for STOP: the phone acts)')
    if reply.get('answer_si'):
        print(f'     answer_si: {_clip(reply["answer_si"])}')
    for w in reply.get('warnings') or []:
        print(f'     ! {w}')
    for s in reply.get('sources') or []:
        print(f'     source: {json.dumps(s, ensure_ascii=False)}')
    if reply.get('timings'):
        print(f'     timings: {json.dumps(reply["timings"])}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--frames', nargs='*',
                    help='a folder of frames, or the frame files themselves')
    ap.add_argument('--job', help='ask against an article already in session')
    ap.add_argument('--ask', action='append',
                    help='a question; repeatable. Omit for the standard set')
    ap.add_argument('--session', action='store_true',
                    help='report what the server is holding, then stop')
    a = ap.parse_args()

    base = a.url.rstrip('/')
    health, code = _get(f'{base}/health')
    if code != 200:
        print(f'server not healthy at {base} (HTTP {code})')
        sys.exit(2)
    print(f'server {base}  device={health.get("device", "?")}')

    job = a.job

    if a.frames:
        paths = []
        for f in a.frames:
            p = Path(f)
            if p.is_dir():
                paths += sorted(q for q in p.iterdir()
                                if q.suffix.lower() in IMG_EXTS)
            elif p.exists():
                paths.append(p)
            else:
                print(f'  no such path: {f}')
        if not paths:
            print('  no frames found')
            sys.exit(2)
        print(f'\nPOST /capture  ({len(paths)} frames: '
              f'{", ".join(p.name for p in paths)})')
        reply, code = _post_files(f'{base}/capture', paths)
        job = reply.get('job') or job
        print(f'  HTTP {code}  ok={reply.get("ok")}  job={job}')
        print(f'  title: {_clip(reply.get("title")) or "(empty — Layer 4A is off)"}')
        print(f'  body : {_clip(reply.get("body"))}')
        for w in reply.get('warnings') or []:
            print(f'  ! {w}')
        if reply.get('error'):
            print(f'  error: {reply["error"]}')
        if not reply.get('ok'):
            print('\n  nothing was read, so nothing was stored.')
            sys.exit(1)

    if not job:
        print('\nNothing to ask about. Pass --frames or --job.')
        sys.exit(2)

    sess, code = _get(f'{base}/session/{job}')
    print(f'\nGET /session/{job}  HTTP {code}')
    if code == 200:
        print(f'  {sess["n_articles"]} article(s), {sess["chars"]} chars, '
              f'{sess["age_s"]}s old, {sess["live_jobs"]} job(s) live')
    else:
        print(f'  {sess.get("error")}  '
              f'(expired, or the server restarted, or nothing was read)')
        sys.exit(1)
    if a.session:
        return

    print('\nPOST /ask')
    questions = ([(q, '') for q in a.ask] if a.ask else DEFAULT_QUESTIONS)
    for q, gloss in questions:
        reply, code = _post_json(f'{base}/ask', {'job': job, 'text': q})
        show_ask(q, gloss, reply, code)

    print('\nReminder: a `voice routing: stub` warning means Component 4 did '
          'not run.\nNothing measured through it is a result.')


if __name__ == '__main__':
    main()

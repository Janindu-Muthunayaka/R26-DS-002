"""
Full-system benchmark: real phone captures through the RUNNING system.

For every distinct capture in system/work (each phone burst once, each
distinct single upload once) it

  1. POSTs the frames to /capture, exactly as the phone does, and records
     whether a readable document came back, the article count, the quality
     verdict and the per-stage timings;
  2. runs a fixed 8-turn spoken conversation through /ask and checks each
     turn against the action the turn policy must take
     (layers/l6_generator/turns.py, TURN_CONDITIONS.xlsx).

A turn PASSES when the reply is ok, has something to say, and takes the
expected action (plus a turn-specific check: REPEAT replays the previous
content, SLOWER lowers the rate, SIMPLER yields the Simple style).

Needs the server (:8000), the voice service and the RAG service running.
/capture and /ask write to the RAG store, chat_sessions and the
personalisation logs: back them up first and restore after.

    python system/tools/bench/bench_e2e.py [--limit N]
Output: system/docs/eval/e2e.json and e2e_turns.csv
"""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from layers.l6_generator.turns import SI_AT_SLOWEST  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WORK = os.path.join(ROOT, 'work')
OUT = os.path.join(ROOT, 'docs', 'eval')
SERVER = os.getenv('BENCH_SERVER', 'http://127.0.0.1:8000')
USER = 'user_001'

# (name, Sinhala command, accepted actions, extra check)
TURNS = [
    ('read_article_1', 'පළමු ලිපිය කියවන්න', {'READ'}, None),
    ('next_part', 'ඊළඟ කොටස කියවන්න', {'NAV', 'READ'}, None),
    ('repeat', 'නැවත කියන්න', {'REPLAY', 'ASK'}, 'replays_content'),
    ('slower', 'ටිකක් හෙමින් කියවන්න', {'CONFIRM', 'CONFIRM + REPLAY'}, 'rate_down'),
    ('summarize', 'මේ ලිපිය සාරාංශ කරන්න', {'REGEN'}, 'generated'),
    ('question', 'මේ ලිපිය කුමක් ගැනද?', {'REGEN'}, 'generated'),
    ('simpler', 'සරලව කියන්න', {'REGEN'}, 'simple_style'),
    ('stop', 'නවත්වන්න', {'STOP'}, 'may_be_silent'),
]


def captures():
    """Each phone burst, and each distinct single image, once (oldest first)."""
    seen, out = set(), []
    for d in sorted(os.listdir(WORK), key=lambda d: os.path.getmtime(os.path.join(WORK, d))):
        p = os.path.join(WORK, d)
        if not os.path.isdir(p) or d == 'eval':
            continue
        frames = sorted(f for f in os.listdir(p) if re.match(r'f\d+.*\.jpg$', f))
        if not frames:
            continue
        h = hashlib.md5(open(os.path.join(p, frames[0]), 'rb').read()).hexdigest()
        if h in seen or _too_small(os.path.join(p, frames[0])):
            continue
        seen.add(h)
        out.append({'source_job': d, 'frames': [os.path.join(p, f) for f in frames],
                    'phone_burst': any(re.search(r'_g\d+_s\d+', f) for f in frames)})
    return out


def _too_small(path, min_side=300):
    """Unit-test fixtures (40x60 px) are not captures."""
    try:
        from PIL import Image
        return min(Image.open(path).size) < min_side
    except Exception:
        return True


def post_frames(paths):
    b = uuid.uuid4().hex
    body = b''
    for p in paths:
        body += (f'--{b}\r\nContent-Disposition: form-data; name="frames"; '
                 f'filename="{os.path.basename(p)}"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
        body += open(p, 'rb').read() + b'\r\n'
    body += f'--{b}--\r\n'.encode()
    req = urllib.request.Request(f'{SERVER}/capture', data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    return _call(req, 600)


def ask(job, text):
    body = json.dumps({'job': job, 'text': text, 'user_id': USER}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(f'{SERVER}/ask', data=body,
                                 headers={'Content-Type': 'application/json; charset=utf-8'})
    return _call(req, 300)


def _call(req, timeout):
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            out = json.loads(e.read().decode('utf-8'))
        except Exception:
            out = {'ok': False, 'error': f'HTTP {e.code}'}
    except Exception as e:
        out = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    return out, time.perf_counter() - t


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 2) if xs else None


def lat(xs):
    return {'n': len(xs), 'median': pct(xs, 0.5), 'p95': pct(xs, 0.95),
            'mean': round(statistics.mean(xs), 2) if xs else None}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', help='one source job id from system/work')
    ap.add_argument('--rescore', action='store_true',
                    help='re-judge the saved e2e_turns.csv instead of running')
    args = ap.parse_args()
    if args.rescore:
        return rescore()
    caps = captures()
    if args.only:
        caps = [c for c in caps if c['source_job'] == args.only]
    if args.limit:
        caps = caps[-args.limit:]
    print(f'{len(caps)} distinct captures')

    turn_rows, cap_rows = [], []
    for i, c in enumerate(caps):
        rep, secs = post_frames(c['frames'])
        q = rep.get('quality') or {}
        cap = {'source_job': c['source_job'], 'phone_burst': c['phone_burst'],
               'n_frames': len(c['frames']), 'job': rep.get('job'), 'ok': bool(rep.get('ok')),
               'n_articles': rep.get('n_articles', 0), 'verdict': q.get('verdict'),
               'secs': round(secs, 2), 'timings': rep.get('timings') or {},
               'error': rep.get('error'), 'warnings': rep.get('warnings', [])}
        cap['readable'] = cap['ok'] and cap['n_articles'] > 0 and cap['verdict'] not in ('unreadable',)
        cap_rows.append(cap)
        print(f"[{i + 1}/{len(caps)}] {c['source_job']} ok={cap['ok']} articles={cap['n_articles']} "
              f"verdict={cap['verdict']} {cap['secs']}s", flush=True)
        if not cap['ok']:
            continue

        history = []
        for name, text, actions, check in TURNS:
            r, s = ask(cap['job'], text)
            row = {'source_job': c['source_job'], 'job': cap['job'], 'turn': name,
                   'text': text, 'expected': '|'.join(sorted(actions)),
                   'action': r.get('action') or '', 'intent': r.get('intent'),
                   'route': r.get('route'), 'ok': bool(r.get('ok')),
                   'style_class': r.get('style_class') or '', 'rate': r.get('speech_rate'),
                   'secs': round(s, 2),
                   'voice_secs': (r.get('timings') or {}).get('voice'),
                   'generate_secs': (r.get('timings') or {}).get('generate'),
                   'spoken': (r.get('speakable') or '').strip()[:200]}
            row['passed'], row['why'] = judge(row, actions, check, history)
            history.append(row)
            turn_rows.append(row)
            print(f"    {name:<15} {'PASS' if row['passed'] else 'FAIL'} {row['action']:<8} "
                  f"{s:5.1f}s {row['why']}", flush=True)

    write(summarise(cap_rows, turn_rows), turn_rows)


def judge(row, actions, check, history):
    """(passed, why) for one turn. `history` is the session's earlier turns."""
    spoken, action = _norm(row['spoken']), row['action']
    if not row['ok'] or action not in actions or (not spoken and check != 'may_be_silent'):
        return False, f"action={action} ok={row['ok']} spoken={bool(spoken)}"
    prev = history[-1] if history else None
    if check == 'replays_content':
        # R6: right after the system ASKED something, "repeat" re-asks it.
        if action == 'ASK':
            ok = prev is not None and spoken == _norm(prev['spoken'])
            return ok, '' if ok else 'ASK is not the pending question'
        # G3: otherwise it replays the last CONTENT, never a status line.
        ok = len(spoken) > 30 and spoken in [_norm(h['spoken']) for h in history]
        return ok, '' if ok else 'replay is not earlier content'
    if check == 'rate_down':
        # The user's rate persists across sessions; at the floor the system
        # must say so instead of going lower (SI_AT_SLOWEST).
        if spoken.startswith(_norm(SI_AT_SLOWEST)):
            return True, ''
        before = next((h['rate'] for h in reversed(history) if h['rate'] not in (None, '')), None)
        ok = before is None or row['rate'] in (None, '') or float(row['rate']) < float(before)
        return ok, '' if ok else f"rate {before} -> {row['rate']}"
    if check == 'generated':
        ok = row['route'] == 'GENERATE'
        return ok, '' if ok else f"route={row['route']}"
    if check == 'simple_style':
        ok = row['style_class'] == 'Simple'
        return ok, '' if ok else f"style={row['style_class']}"
    return True, ''


def rescore():
    """Re-judge a saved run (e2e_turns.csv + e2e.json) with the current rules."""
    with open(os.path.join(OUT, 'e2e_turns.csv'), encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    cap_rows = json.load(open(os.path.join(OUT, 'e2e.json'), encoding='utf-8'))['capture_rows']
    spec = {t[0]: t for t in TURNS}
    by_job = {}
    for r in rows:
        r['ok'] = r['ok'] == 'True'
        r['secs'] = float(r['secs'])
        _, _, actions, check = spec[r['turn']]
        hist = by_job.setdefault(r['job'], [])
        old = (r['passed'] == 'True', r['why'])
        legacy = 'rate' not in r                      # runs before 18 Oct 10:50 lack rate/style
        r.setdefault('rate', None)
        r.setdefault('style_class', '')
        r['passed'], r['why'] = judge(r, actions, check, hist)
        # Without the saved field the new rule cannot decide: keep the verdict
        # the live run reached, except where the reply itself settles it.
        if legacy and (check == 'simple_style'
                       or (check == 'rate_down' and not _norm(r['spoken']).startswith(_norm(SI_AT_SLOWEST)))):
            r['passed'], r['why'] = old
        hist.append(r)
    write(summarise(cap_rows, rows), rows)


def summarise(cap_rows, turn_rows):
    n = len(cap_rows)
    readable = [c for c in cap_rows if c['readable']]
    res = {'run_at': time.strftime('%Y-%m-%d %H:%M'), 'server': SERVER,
           'turn_script': [t[1] for t in TURNS],
           'captures': {
               'n': n,
               'phone_bursts': sum(c['phone_burst'] for c in cap_rows),
               'pipeline_ok': sum(c['ok'] for c in cap_rows),
               'readable_document': len(readable),
               'readable_rate': round(len(readable) / n, 4) if n else None,
               'verdicts': {v: sum(c['verdict'] == v for c in cap_rows)
                            for v in sorted({str(c['verdict']) for c in cap_rows})},
               'articles_per_capture_mean': round(statistics.mean(
                   [c['n_articles'] for c in cap_rows if c['ok']] or [0]), 2),
               'capture_latency_sec': lat([c['secs'] for c in cap_rows]),
           }}
    stages = {}
    for c in cap_rows:
        for k, v in (c['timings'] or {}).items():
            if isinstance(v, (int, float)):
                stages.setdefault(k, []).append(v)
    res['captures']['stage_latency_sec'] = {k: lat(v) for k, v in stages.items()}

    res['turns'] = {'n': len(turn_rows), 'passed': sum(t['passed'] for t in turn_rows)}
    res['turns']['task_success_rate'] = round(res['turns']['passed'] / len(turn_rows), 4) if turn_rows else None
    res['turns']['by_turn'] = {
        name: {'n': len(ts), 'passed': sum(t['passed'] for t in ts),
               'rate': round(sum(t['passed'] for t in ts) / len(ts), 4),
               'latency_sec': lat([t['secs'] for t in ts])}
        for name in [t[0] for t in TURNS]
        for ts in [[t for t in turn_rows if t['turn'] == name]] if ts}
    res['turns']['latency_sec'] = {
        'local': lat([t['secs'] for t in turn_rows if t['route'] == 'LOCAL']),
        'generate': lat([t['secs'] for t in turn_rows if t['route'] == 'GENERATE'])}
    # Whole sessions: capture readable AND all 8 turns passed.
    by_job = {}
    for t in turn_rows:
        by_job.setdefault(t['job'], []).append(t['passed'])
    res['sessions_all_turns_passed'] = sum(all(v) and len(v) == len(TURNS) for v in by_job.values())
    res['capture_rows'] = cap_rows
    return res


def write(res, turn_rows):
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'e2e.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'e2e_turns.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(turn_rows[0]) if turn_rows else ['none'])
        w.writeheader()
        w.writerows(turn_rows)
    print(json.dumps({k: v for k, v in res.items() if k != 'capture_rows'}, ensure_ascii=False, indent=2))


def _norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[​-‍﻿]', '', s)).strip()


if __name__ == '__main__':
    main()

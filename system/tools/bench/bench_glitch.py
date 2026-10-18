"""
Whole-system glitch hunt against the RUNNING services.

Part A — every server endpoint, including bad input. A glitch is an HTTP 500,
a non-JSON reply, or a reply a blind user cannot act on.

Part B — real phone captures through /capture, then a long spoken
conversation per page that exercises every turn type (read, navigate, repeat,
speed, pauses, volume, style, rephrase, summary, questions, an off-topic
question, junk input, stop). Each turn is checked against the action the turn
policy must take AND for generic glitches:

    HTTP error / ok=false            silent turn (nothing to say, not STOP)
    a control command sent to RAG    "not enough information" for a command
    English / code text spoken       ("None", "{", "Article 1:", "[RAG", mostly Latin)
    too slow                         LOCAL > 15 s, GENERATE > 90 s
    speech rate / pause not applied

Runs as a throwaway user created at the start and deleted at the end, so no
real user's history or style weights change.

    python system/tools/bench/bench_glitch.py [--captures 3]
Output: system/docs/eval/glitch.json and glitch_turns.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_e2e as e2e                                               # noqa: E402

SERVER = e2e.SERVER
OUT = e2e.OUT
NO_INFO = 'කියවන ලද පාඨයේ මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත'

# (name, Sinhala, accepted actions, check)  — None = any action is fine
SCRIPT = [
    ('read_article_2', 'දෙවන ලිපිය කියවන්න', {'READ'}, 'starts:ලිපිය 2'),
    ('next_one', 'ඊළඟ එකට යන්න', {'NAV', 'READ', 'ASK'}, 'local'),
    ('go_back', 'පස්සට යන්න', {'NAV', 'READ', 'ASK'}, 'local'),
    ('repeat', 'නැවත කියන්න', {'REPLAY', 'ASK'}, 'local'),
    ('faster', 'වේගෙන් කියන්න', {'CONFIRM', 'CONFIRM + REPLAY'}, 'rate_up'),
    ('slower', 'හෙමින් කියන්න', {'CONFIRM', 'CONFIRM + REPLAY'}, 'rate_down'),
    ('more_pauses', 'නවත්තලා නවත්තලා කියන්න', {'CONFIRM', 'CONFIRM + REPLAY'}, 'pause_up'),
    ('noise_repeat', 'සද්දෙ නිසා මට ඇහුනෙ නෑ', {'REPLAY', 'ASK'}, 'local'),
    ('volume', 'ශබ්දය වැඩි කරන්න', {'CONFIRM'}, 'local'),
    ('read_article_1', 'පළමු ලිපිය කියවන්න', {'READ'}, 'starts:ලිපිය 1'),
    ('summary_art1', 'ලිපිය එකේ සාරාංශය කියන්න', {'REGEN'}, 'answer'),
    ('question_art1', 'ලිපිය එකේ ප්‍රධාන කරුණ මොකක්ද', {'REGEN'}, 'answer'),
    ('question_inflected', 'ලිපියේ එකෙහි ප්‍රධාන කරුණ මොකක්ද', {'REGEN'}, 'answer'),
    ('simpler', 'සරලව කියන්න', {'REGEN'}, 'style:Simple'),
    ('step_by_step', 'පියවරෙන් පියවර කියන්න', {'REGEN'}, 'style:StepByStep'),
    ('rephrase', 'වෙනත් විදිහකට කියන්න', {'REGEN'}, 'answer'),
    ('off_topic', 'අද කාලගුණය කොහොමද', {'REGEN', 'MSG'}, 'abstain_ok'),
    ('junk', 'ම්ම් අහ්', None, 'no_rag'),
    ('previous_page', 'පෙර පිටුව', {'NAV', 'READ', 'ASK'}, 'local'),
    ('stop', 'නවත්වන්න', {'STOP'}, 'silent_ok'),
]
CODEY = re.compile(r'(\bNone\b|\bnull\b|[{}\[\]<>]|Traceback|Article \d|\[RAG)')


def call(method, path, body=None, raw=None, ctype='application/json', timeout=300):
    data = raw if raw is not None else (
        json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None)
    req = urllib.request.Request(f'{SERVER}{path}', data=data, method=method,
                                 headers={'Content-Type': ctype} if data is not None else {})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, text = r.status, r.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        code, text = e.code, e.read().decode('utf-8', 'replace')
    except Exception as e:
        code, text = 0, f'{type(e).__name__}: {e}'
    secs = time.perf_counter() - t
    try:
        js = json.loads(text)
    except Exception:
        js = None
    return code, js, text, secs


def latin_share(s):
    letters = [c for c in s if c.isalpha()]
    return sum('a' <= c.lower() <= 'z' for c in letters) / len(letters) if letters else 0.0


# ── Part A ────────────────────────────────────────────────────────────────
def endpoints(job, uid):
    checks = [
        ('GET /', 'GET', '/', None, 200, 'html'),
        ('GET /reader', 'GET', '/reader', None, 200, 'html'),
        ('GET /debug', 'GET', '/debug', None, 200, 'html'),
        ('GET /health', 'GET', '/health', None, 200, 'json'),
        ('GET /users', 'GET', '/users', None, 200, 'json'),
        ('GET /active-user', 'GET', '/active-user', None, 200, 'json'),
        ('GET /latest', 'GET', '/latest', None, 200, 'json'),
        ('GET /session/job', 'GET', f'/session/{job}', None, 200, 'json'),
        ('GET /document/job', 'GET', f'/document/{job}', None, 200, 'json'),
        ('GET /profile/user', 'GET', f'/profile/{uid}', None, 200, 'json'),
        ('GET /audio/job', 'GET', f'/audio/{job}', None, (200, 404), 'any'),
        ('GET /session/unknown', 'GET', '/session/nosuchjob', None, (200, 404), 'json'),
        ('GET /document/unknown', 'GET', '/document/nosuchjob', None, (200, 404), 'json'),
        ('POST /ask unknown job', 'POST', '/ask', {'job': 'nosuchjob', 'text': 'නැවත කියන්න'}, 404, 'speakable'),
        ('POST /ask empty text', 'POST', '/ask', {'job': job, 'text': '', 'user_id': uid}, (200, 400, 422), 'json'),
        ('POST /ask no job', 'POST', '/ask', {'text': 'x'}, 422, 'json'),
        ('POST /regenerate bad index', 'POST', '/regenerate', {'job': job, 'index': 999, 'user_id': uid},
         (200, 400, 404), 'json'),
        ('POST /users bad body', 'POST', '/users', {'name': 'x'}, 422, 'json'),
        ('POST /active-user unknown', 'POST', '/active-user', {'user_id': 'no_such_user'},
         (200, 400, 404), 'json'),
    ]
    rows = []
    for name, m, path, body, want, kind in checks:
        code, js, text, secs = call(m, path, body)
        want = want if isinstance(want, tuple) else (want,)
        problems = []
        if code == 500 or code == 0:
            problems.append(f'HTTP {code}')
        elif code not in want:
            problems.append(f'HTTP {code}, expected {want}')
        if kind == 'json' and js is None:
            problems.append('not JSON')
        if kind == 'html' and not re.search(r'<!doctype html|<html', text, re.I):
            problems.append('not HTML')
        if kind == 'speakable' and not ((js or {}).get('speakable') or '').strip():
            problems.append('nothing to say')
        rows.append({'check': name, 'http': code, 'secs': round(secs, 2),
                     'ok': not problems, 'problems': '; '.join(problems)})
    # Bad uploads must be refused cleanly, never a 500.
    for name, raw in (('POST /capture not an image', b'not an image'),
                      ('POST /capture empty file', b'')):
        b = 'glitchbound'
        body = (f'--{b}\r\nContent-Disposition: form-data; name="frames"; filename="x.jpg"\r\n'
                f'Content-Type: image/jpeg\r\n\r\n').encode() + raw + f'\r\n--{b}--\r\n'.encode()
        code, js, text, secs = call('POST', '/capture', raw=body, ctype=f'multipart/form-data; boundary={b}')
        problems = [] if code == 400 and js else [f'HTTP {code}']
        rows.append({'check': name, 'http': code, 'secs': round(secs, 2),
                     'ok': not problems, 'problems': '; '.join(problems)})
    return rows


# ── Part B ────────────────────────────────────────────────────────────────
def judge(name, actions, check, r, prev, code, secs):
    spoken = (r.get('speakable') or '').strip()
    action, route = r.get('action') or '', r.get('route') or ''
    p = []
    if code != 200:
        p.append(f'HTTP {code}')
    if not r.get('ok') and check != 'abstain_ok':
        p.append('ok=false')
    if actions is not None and action not in actions:
        p.append(f'action {action or "-"}')
    if not spoken and check != 'silent_ok':
        p.append('silent')
    if check in ('local', 'no_rag') and route == 'GENERATE':
        p.append('command sent to RAG')
    if NO_INFO in spoken and check != 'abstain_ok':
        p.append('"not enough information"')
    if spoken and (CODEY.search(spoken) or latin_share(spoken) > 0.3):
        p.append('code / English in speech')
    if (route == 'LOCAL' and secs > 15) or secs > 90:
        p.append(f'slow {secs:.0f}s')
    if check.startswith('starts:') and not spoken.startswith(check[7:]):
        p.append(f'does not start with {check[7:]!r}')
    if check.startswith('style:') and r.get('style_class') != check[6:]:
        p.append(f"style {r.get('style_class')}")
    rate, prate = r.get('speech_rate'), (prev or {}).get('speech_rate')
    slowest, fastest = 'මෙය අඩුම වේගයයි', 'මෙය උපරිම වේගයයි'
    if check == 'rate_up' and prate is not None and not (rate > prate or fastest in spoken or rate >= 1.3):
        p.append(f'rate {prate} -> {rate}')
    if check == 'rate_down' and prate is not None and not (rate < prate or slowest in spoken or rate <= 0.8):
        p.append(f'rate {prate} -> {rate}')
    ps, pps = r.get('pause_scale'), (prev or {}).get('pause_scale')
    if check == 'pause_up' and pps is not None and ps is not None and not ps >= pps:
        p.append(f'pause {pps} -> {ps}')
    return p


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', type=int, default=3)
    ap.add_argument('--skip', type=int, default=0, help='skip the N richest pages (test new ones)')
    args = ap.parse_args()

    code, js, _, _ = call('POST', '/users', {'base_weights': {'Simple': 0.34, 'Detailed': 0.33, 'StepByStep': 0.33},
                                             'name': 'Glitch test', 'tag': 'test',
                                             'description': 'temporary user for bench_glitch.py'})
    uid = (js or {}).get('user_id') or ((js or {}).get('user') or {}).get('user_id')
    if not uid:
        sys.exit(f'could not create a test user: HTTP {code} {js}')
    print('test user', uid)

    caps = [c for c in e2e.captures() if c['phone_burst']]
    # The richest pages first: more articles = more to navigate.
    prior = {c['source_job']: c for c in json.load(open(os.path.join(OUT, 'e2e.json'), encoding='utf-8'))
             ['capture_rows']} if os.path.exists(os.path.join(OUT, 'e2e.json')) else {}
    caps.sort(key=lambda c: -(prior.get(c['source_job'], {}).get('n_articles') or 0))
    caps = caps[args.skip:args.skip + args.captures]

    turn_rows, endpoint_rows, cap_rows = [], [], []
    try:
        for i, c in enumerate(caps):
            rep, secs = e2e.post_frames(c['frames'])
            cap_rows.append({'source_job': c['source_job'], 'job': rep.get('job'), 'ok': rep.get('ok'),
                             'n_articles': rep.get('n_articles'), 'secs': round(secs, 1),
                             'verdict': (rep.get('quality') or {}).get('verdict')})
            print(f"[{i + 1}/{len(caps)}] {c['source_job']} -> {rep.get('job')} ok={rep.get('ok')} "
                  f"articles={rep.get('n_articles')} {secs:.0f}s", flush=True)
            if not rep.get('ok'):
                continue
            if not endpoint_rows:
                endpoint_rows = endpoints(rep['job'], uid)
                for e in endpoint_rows:
                    print(f"    {'ok ' if e['ok'] else 'BAD'} {e['check']:<30} {e['http']} {e['problems']}")
            prev = None
            for name, text, actions, check in SCRIPT:
                code, r, _, s = call('POST', '/ask', {'job': rep['job'], 'text': text, 'user_id': uid})
                r = r or {}
                problems = judge(name, actions, check, r, prev, code, s)
                turn_rows.append({'source_job': c['source_job'], 'job': rep['job'], 'turn': name, 'text': text,
                                  'http': code, 'action': r.get('action'), 'intent': r.get('intent'),
                                  'route': r.get('route'), 'style': r.get('style_class'),
                                  'rate': r.get('speech_rate'), 'pause': r.get('pause_scale'),
                                  'secs': round(s, 2), 'glitch': bool(problems),
                                  'problems': '; '.join(problems),
                                  'spoken': (r.get('speakable') or '').strip()[:240]})
                print(f"    {'GLITCH' if problems else 'ok    '} {name:<19} {r.get('action') or '-':<17} "
                      f"{s:5.1f}s {'; '.join(problems)}", flush=True)
                prev = r
    finally:
        code, _, _, _ = call('DELETE', f'/users/{uid}')
        print('test user deleted' if code == 200 else f'could not delete test user {uid}: HTTP {code}')

    by = {}
    for t in turn_rows:
        b = by.setdefault(t['turn'], {'n': 0, 'glitches': 0})
        b['n'] += 1
        b['glitches'] += t['glitch']
    kinds = {}
    for t in turn_rows:
        for p in filter(None, t['problems'].split('; ')):
            k = re.sub(r'[\d.]+', '#', p)
            kinds[k] = kinds.get(k, 0) + 1
    res = {'run_at': time.strftime('%Y-%m-%d %H:%M'), 'captures': cap_rows,
           'endpoints': {'n': len(endpoint_rows), 'bad': [e for e in endpoint_rows if not e['ok']]},
           'turns': {'n': len(turn_rows), 'glitches': sum(t['glitch'] for t in turn_rows),
                     'by_turn': by, 'by_problem': dict(sorted(kinds.items(), key=lambda x: -x[1]))}}
    os.makedirs(OUT, exist_ok=True)
    json.dump({**res, 'endpoint_rows': endpoint_rows}, open(os.path.join(OUT, 'glitch.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    if turn_rows:
        with open(os.path.join(OUT, 'glitch_turns.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(turn_rows[0]))
            w.writeheader()
            w.writerows(turn_rows)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

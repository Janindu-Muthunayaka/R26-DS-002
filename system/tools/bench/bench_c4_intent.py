"""
C4 benchmark: intent detection through the DEPLOYED voice service.

Posts every test command to POST {VOICE_URL}/interpret (the same call the
server makes for each spoken turn) and scores the returned intent and
modifiers.

Test sets
  gen500  Work/Bumal/approach1_generated_results.csv  (500 generated commands,
          17 thesis labels mapped onto the production labels; GO_TO_START and
          GO_TO_END have no production label and are left out)
  p2b45 / multi / fresh   layers/l7_personalization/data/test_samples*.py
          (production labels; multi and fresh also carry expected modifiers;
          fresh is the held-out set)

/interpret LOGS the turn into the personalisation data (db.json,
interaction_logs.csv). Back those files up before running and restore them
after, or the benchmark becomes part of a user's history.

    python system/tools/bench/bench_c4_intent.py [--limit N]
Output: system/docs/eval/c4_intent.json and c4_intent_rows.csv
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SYS = os.path.join(ROOT, 'system')
OUT = os.path.join(SYS, 'docs', 'eval')
VOICE_URL = os.getenv('SINHALA_VOICE_URL', 'http://127.0.0.1:8101')
USER = 'user_001'

THESIS_TO_PROD = {
    'NEXT_PAGE': 'NEXT', 'NEXT_ARTICLE': 'NEXT',
    'PREVIOUS_PAGE': 'PREVIOUS', 'PREVIOUS_ARTICLE': 'PREVIOUS',
    'NAVIGATE_SECTION': 'GO_TO_SECTION', 'NAVIGATE_PAGE_NUMBER': 'GO_TO_PAGE',
    'READ_ARTICLE': 'READ_ALOUD',
    'GO_TO_START': None, 'GO_TO_END': None,
}


def load_sets():
    rows = []
    with open(os.path.join(ROOT, 'Work', 'Bumal', 'approach1_generated_results.csv'),
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            exp = THESIS_TO_PROD.get(r['Expected'], r['Expected'])
            if exp:
                rows.append({'set': 'gen500', 'id': r['ID'], 'text': r['Input'],
                             'expected': exp, 'thesis_label': r['Expected']})
    sys.path.insert(0, os.path.join(SYS, 'layers', 'l7_personalization'))
    from data.test_samples import test_samples
    from data.test_samples_multi import multi_samples
    from data.test_samples_fresh import fresh_samples
    for name, s in (('p2b45', test_samples), ('multi', multi_samples), ('fresh', fresh_samples)):
        for r in s:
            rows.append({'set': name, 'id': r['id'], 'text': r['stt_output'],
                         'expected': r['expected_intent'],
                         'expected_modifiers': r.get('expected_modifiers')})
    return rows


def interpret(text):
    body = json.dumps({'text': text, 'user_id': USER}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(f'{VOICE_URL}/interpret', data=body,
                                 headers={'Content-Type': 'application/json; charset=utf-8'})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode('utf-8'))
    return out, time.perf_counter() - t


def prf(rows):
    labels = sorted({r['expected'] for r in rows})
    per, f1s = {}, []
    for l in labels:
        tp = sum(r['expected'] == l == r['pred'] for r in rows)
        fp = sum(r['pred'] == l != r['expected'] for r in rows)
        fn = sum(r['expected'] == l != r['pred'] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        per[l] = {'n': tp + fn, 'precision': round(p, 3), 'recall': round(rc, 3), 'f1': round(f, 3)}
        f1s.append((f, tp + fn))
    n = len(rows)
    return {
        'n': n,
        'accuracy': round(sum(r['expected'] == r['pred'] for r in rows) / n, 4),
        'macro_f1': round(sum(f for f, _ in f1s) / len(f1s), 4),
        'weighted_f1': round(sum(f * s for f, s in f1s) / n, 4),
        'per_intent': per,
        'top_confusions': [{'true': a, 'pred': b, 'count': c} for (a, b), c in Counter(
            (r['expected'], r['pred']) for r in rows if r['expected'] != r['pred']).most_common(8)],
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    rows = load_sets()
    if args.limit:
        rows = rows[:args.limit]
    interpret('ආයුබෝවන්')                               # warm-up, not scored
    for i, r in enumerate(rows):
        try:
            out, secs = interpret(r['text'])
            r['pred'] = out.get('intent') or 'UNKNOWN'
            r['route'] = out.get('route')
            r['secs'] = round(secs, 3)
            r['english'] = out.get('english_translation')
            r['modifiers'] = (out.get('frame') or {}).get('modifiers') or {}
        except Exception as e:                           # a failed call is a wrong answer
            r['pred'], r['secs'], r['error'] = 'ERROR', None, str(e)
        if i % 25 == 0:
            print(f"{i}/{len(rows)} {r['expected']} -> {r['pred']} ({r.get('secs')}s)", flush=True)

    res = {'service': f'{VOICE_URL}/interpret', 'user': USER,
           'run_at': time.strftime('%Y-%m-%d %H:%M'), 'sets': {}}
    for name in ('gen500', 'p2b45', 'multi', 'fresh'):
        s = [r for r in rows if r['set'] == name]
        if s:
            res['sets'][name] = prf(s)
    # Modifier accuracy where the set defines expected modifiers.
    mods = [r for r in rows if r.get('expected_modifiers') and r['pred'] != 'ERROR']
    if mods:
        res['modifiers'] = {}
        for k in ('repeat', 'speed', 'style'):
            ok = sum((r['modifiers'].get(k) or None) == (r['expected_modifiers'].get(k) or None)
                     for r in mods)
            res['modifiers'][k] = {'n': len(mods), 'accuracy': round(ok / len(mods), 4)}
        res['modifiers']['all_three'] = round(sum(all(
            (r['modifiers'].get(k) or None) == (r['expected_modifiers'].get(k) or None)
            for k in ('repeat', 'speed', 'style')) for r in mods) / len(mods), 4)
    t = sorted(r['secs'] for r in rows if r.get('secs') is not None)
    res['latency_sec'] = {'n': len(t), 'mean': round(statistics.mean(t), 3),
                          'median': round(statistics.median(t), 3),
                          'p95': round(t[int(0.95 * (len(t) - 1))], 3), 'max': round(t[-1], 3)}
    res['errors'] = sum(r['pred'] == 'ERROR' for r in rows)

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'c4_intent.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'c4_intent_rows.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['set', 'id', 'text', 'expected', 'pred', 'correct', 'route', 'secs', 'english'])
        for r in rows:
            w.writerow([r['set'], r['id'], r['text'], r['expected'], r['pred'],
                        r['expected'] == r['pred'], r.get('route'), r.get('secs'), r.get('english')])
    print(json.dumps({k: (v if k != 'sets' else {s: {x: v[s][x] for x in ('n', 'accuracy', 'macro_f1')}
                                                  for s in v}) for k, v in res.items()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

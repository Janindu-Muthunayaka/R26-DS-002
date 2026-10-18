"""
L7 -> turn-policy replay: what the USER experiences for each voice command.

bench_c4_intent.py scores Component 4's label. A wrong label only matters if
it changes what happens, so this replays every recorded /interpret result
(docs/eval/c4_intent_rows.csv) through the real turn policy
(layers/l6_generator/turns.py) with a fake RAG, in a fresh session where an
article has just been read, and checks the ACTION against what the command
asked for:

    NEXT / PREVIOUS          navigate (NAV / READ / ASK "read next?"), never RAG
    STOP                     STOP
    REPEAT                   REPLAY (or re-ASK the pending question)
    READ_ALOUD               READ
    READ_HEADLINES           read the titles locally, never RAG
    GO_TO_PAGE / SECTION     handled locally, never RAG
    CHANGE_SPEED             CONFIRM, never RAG
    content requests         answered by RAG (REGEN)

`--rules` re-labels each command with the Sinhala rule layer first
(layers/l7_personalization/intent_detection/sinhala_rules.py), exactly as the
service does, so the before / after effect of the rules is measured on the
same recorded outputs.

    system/.venv/Scripts/python.exe system/tools/bench/replay_turns.py [--rules]
Output: docs/eval/replay_turns[_rules].json and .csv
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

SYS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, SYS)
OUT = os.path.join(SYS, 'docs', 'eval')

from core.schemas import Article, Box, Document                       # noqa: E402
from layers.l6_generator import generate as l6gen                      # noqa: E402
from layers.l6_generator import turns                                  # noqa: E402
from layers.l7_personalization.intent_detection.modifiers import build_frame  # noqa: E402

CONTENT = {'SUMMARIZE', 'EXPLAIN', 'ELABORATE', 'SIMPLIFY', 'STEP_BY_STEP', 'REPHRASE',
           'IDENTIFY_CONTENT', 'ASK_QUESTION'}
EXPECT = {
    'NEXT': {'NAV', 'READ', 'ASK'}, 'PREVIOUS': {'NAV', 'READ', 'ASK'},
    'STOP': {'STOP'}, 'REPEAT': {'REPLAY', 'ASK'}, 'READ_ALOUD': {'READ'},
    'READ_HEADLINES': {'READ', 'MSG'}, 'GO_TO_PAGE': {'READ', 'MSG', 'NAV', 'ASK'},
    'GO_TO_SECTION': {'READ', 'MSG', 'NAV', 'ASK'},
    'CHANGE_SPEED': {'CONFIRM', 'CONFIRM + REPLAY', 'ASK'},
}
BOX = Box(x1=0, y1=0, x2=10, y2=10)


def _doc():
    long = ' '.join(f'පළමු ලිපියේ {i} වන වාක්‍යයයි එය තරමක් දිගු වේ.' for i in range(30))
    return Document(articles=[
        Article(index=0, box=BOX, body_raw=long, body=long, title='බෝක්කුව කැඩිලා', verdict='ok'),
        Article(index=1, box=BOX, body_raw='දෙවන ලිපියේ පෙළ මෙයයි.', body='දෙවන ලිපියේ පෙළ මෙයයි.',
                title='විකිරණ යන්ත්‍රය ක්‍රියා විරහිතයි', verdict='ok'),
        Article(index=2, box=BOX, body_raw='තුන්වන ලිපියයි.', body='තුන්වන ලිපියයි.',
                title='තුන්වන මාතෘකාව', verdict='ok')])


def install_fake_rag():
    real = l6gen.answer
    asked = []

    def fake(doc, voice, mode=None, cursor=0):
        if str(voice.get('route')).upper() == 'GENERATE' \
                and str(voice.get('intent')).upper() not in l6gen.LOCAL_INTENTS:
            asked.append(voice)
            text = f"[RAG] {voice.get('english_translation')}"
            return l6gen._out(True, 'GENERATE', voice['intent'], text, answer_si=text, cursor=cursor)
        return real(doc, voice, mode='off', cursor=cursor)

    l6gen.answer = fake
    return real, asked


def voice_for(text, english, intent):
    frame = build_frame(text, english, intent)
    local = intent in EXPECT and intent not in ('READ_HEADLINES',)
    return {'route': 'LOCAL' if local else 'GENERATE', 'intent': intent,
            'english_translation': english or text, 'sinhala_input': text,
            'frame': frame, 'style_class': 'Detailed', 'prompt_modifier': '',
            'source': 'replay'}


def run_one(real, asked, text, english, intent):
    doc, state = _doc(), {}
    first = real(doc, {'intent': 'FIRST_SCAN', 'route': 'GENERATE'}, mode='off')
    turns.seed_first_scan(state, first['speakable'])
    # An article has just been read: "next", "repeat", "previous" have context.
    r = turns.run(doc, voice_for('පළමු ලිපිය කියවන්න', 'Read the first article', 'READ_ALOUD'),
                  state, cursor=0)
    asked.clear()
    r = turns.run(doc, voice_for(text, english, intent), state, cursor=r['cursor'])
    return r, bool(asked)


def judge(expected, action, to_rag):
    if expected in CONTENT:
        return to_rag and action == 'REGEN'
    return action in EXPECT.get(expected, set()) and not to_rag


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', action='store_true')
    args = ap.parse_args()
    rules = None
    if args.rules:
        from layers.l7_personalization.intent_detection.sinhala_rules import rule_intent
        rules = rule_intent
    real, asked = install_fake_rag()
    rows = list(csv.DictReader(open(os.path.join(OUT, 'c4_intent_rows.csv'), encoding='utf-8-sig')))
    out = []
    for row in rows:
        pred = row['pred']
        if rules:
            pred = rules(row['text']) or pred
        r, to_rag = run_one(real, asked, row['text'], row['english'], pred)
        ok = judge(row['expected'], r['action'], to_rag)
        out.append({'set': row['set'], 'id': row['id'], 'text': row['text'],
                    'expected': row['expected'], 'label': pred,
                    'label_correct': pred == row['expected'], 'action': r['action'],
                    'to_rag': to_rag, 'experience_ok': ok})
    l6gen.answer = real

    def summ(rs):
        ctrl = [r for r in rs if r['expected'] not in CONTENT]
        return {'n': len(rs),
                'label_accuracy': round(sum(r['label_correct'] for r in rs) / len(rs), 4),
                'experience_ok': round(sum(r['experience_ok'] for r in rs) / len(rs), 4),
                'control_commands': len(ctrl),
                'control_sent_to_rag': sum(r['to_rag'] for r in ctrl),
                'content_not_answered': sum(not r['to_rag'] for r in rs if r['expected'] in CONTENT)}
    res = {'rules': bool(rules), 'all': summ(out),
           'by_set': {s: summ([r for r in out if r['set'] == s]) for s in ('gen500', 'p2b45', 'multi', 'fresh')},
           'failures_by_expected': dict(Counter(r['expected'] for r in out if not r['experience_ok']).most_common())}
    tag = '_rules' if rules else ''
    json.dump(res, open(os.path.join(OUT, f'replay_turns{tag}.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, f'replay_turns{tag}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

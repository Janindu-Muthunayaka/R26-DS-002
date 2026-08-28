"""
Where core/quality.py's thresholds come from.

    python tools/calibrate_quality.py
    python tools/calibrate_quality.py --self-test

Scores this project's own OCR outputs — the target-glyph sweep in
`tools/out/sweep/`, the framing comparison in `tools/out/cer/`, and the hand
ground truth in `Work/Ishara/article_truth*.txt` — and prints each measure so
the separation can be seen rather than asserted.

`--self-test` first, as every tool in this folder does where a scorer is
involved: a scorer that cannot reproduce a known answer cannot be trusted on
an unknown one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SYSTEM = Path(__file__).resolve().parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

from core import quality      # noqa: E402


def self_test() -> bool:
    """Constructed inputs whose answers are known by inspection."""
    ok = True
    cases = [
        ('clean Sinhala', 'ඉංග්‍රීසි ජාතික ආක්‍රමණිකයන් හා ලක්දිව එකල පැවති '
                          'එකම නිදහස් රට වූ සිංහලේ රදළ වරුන් අතර ඇති කර ගත් '
                          'අවබෝධතා ගිවිසුමකි.', 'good'),
        ('all Latin', 'with ikon One ush kinni high the report said today '
                      'again and again for sure', 'unreadable'),
        ('fragments', 'ක ය ම න ද ව ග ර ත ප ල ස හ ක ය ම න ද ව', 'unreadable'),
        ('too short', 'කුරුණෑගල', 'unreadable'),
        ('decode damage', ('කුරුණෑගල ��� චාන්දනී �� '
                           'දිසානායක � විවිධ සංවර්ධන �� '
                           'ව්‍යාපෘති කඩ කාමර �� බෙදා දීම'), 'unreadable'),
        ('empty', '', 'unreadable'),
    ]
    for name, text, want in cases:
        got = quality.score(text)['verdict']
        flag = 'ok ' if got == want else 'FAIL'
        if got != want:
            ok = False
        print(f'  [{flag}] {name:16} -> {got:10} (want {want})')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()

    print('self-test — constructed inputs with known answers')
    passed = self_test()
    print(f'  {"PASSED" if passed else "FAILED"}\n')
    if a.self_test:
        sys.exit(0 if passed else 1)
    if not passed:
        print('scorer failed its own test; not scoring real files')
        sys.exit(1)

    roots = [_SYSTEM / 'tools' / 'out' / 'sweep',
             _SYSTEM / 'tools' / 'out' / 'cer',
             _SYSTEM.parent / 'Work' / 'Ishara']
    files = []
    for r in roots:
        if r.is_dir():
            files += sorted(p for p in r.glob('*.txt'))
    if not files:
        print('no OCR output found — run the pipeline first')
        return

    print(f'{"file":46} {"words":>6} {"sinhala":>8} {"short":>7} '
          f'{"bad/1k":>7}  verdict')
    print('-' * 92)
    for p in files:
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f'{p.name:46} unreadable: {e}')
            continue
        m = quality.score(text)
        print(f'{p.name[:46]:46} {m["n_words"]:6d} {m["sinhala_ratio"]:8.3f} '
              f'{m["short_ratio"]:7.3f} {m["replacement_per_1k"]:7.2f}  '
              f'{m["verdict"]}'
              + (f'  ({m["reasons"][0]})' if m['reasons'] else ''))

    print('\nGround truth files should score "good". Anything from the 11px '
          'end of the\nsweep scoring "good" too means the thresholds are too '
          'loose to be useful.')


if __name__ == '__main__':
    main()

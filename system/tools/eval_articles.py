#!/usr/bin/env python3
"""
eval_articles.py — end-to-end reading accuracy over SEVERAL articles.

WHY THIS EXISTS
---------------
Every CER figure this project can currently quote for the deployed path comes
from ONE article of 684 characters. n = 1. The correction result (0.1197 ->
0.0757 on 217 sentences) is properly powered; the *system* result is not, and
an examiner will ask.

This runs whole captures through the real read path, scores each against a hand
transcription, and reports a mean with a spread instead of a single number.

IT ALSO SETTLES psm 3 vs psm 6
------------------------------
The deployed path OCRs the whole multi-column crop in one pass with psm 3.
Measured once, on one crop: psm 6 on a SINGLE column returned 693 characters
against psm 3's 561, and psm 3 returned ZERO on one 277x445 crop. That was one
observation and it was never turned into a decision.

So every article is read twice:

    page    whole crop, psm 3            what ships today
    cols    each whole column, psm 6,    what Large_Articles_Design.md step B
            concatenated left to right   proposes

Same frames, same scaling rule, same mT5, same transcription. Two numbers per
article, and the difference is paired, so the comparison does not depend on the
articles being equally hard.

LAYOUT
------
    work/eval/
        article1/  f0_g23_s2969.jpg  f1_...  f2_...  truth.txt
        article2/  ...

One capture per article. `truth.txt` is a hand transcription of the WHOLE
article as printed.

    python tools\\eval_articles.py --root E:\\RP\\corpus\\Sinhala_OCR_Correction_v2 work\\eval

TRANSCRIBE FROM THE NEWSPAPER, NOT FROM THE OCR
-----------------------------------------------
Correcting the OCR output into a transcription biases CER DOWNWARDS: the errors
you fail to notice are precisely the OCR's errors, and they then score as
correct. Type it from the printed page. Afterwards, `--diff` shows you where
your text and the OCR disagree, which is a good way to catch YOUR typing slips
without adopting the machine's.

Scoring is imported from verify_model.py, the same function the research
harness uses, so these numbers and the Chapter 4 numbers are the same metric by
construction.
"""
import argparse
import os
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
TRUTH = 'truth.txt'


def flat(s):
    """Collapse whitespace. Line breaks are an OCR artefact, not content."""
    return re.sub(r'\s+', ' ', s).strip()


def articles(root):
    """-> [(name, [frame paths], truth text or None)] for each subfolder."""
    root = Path(root)
    if not root.exists():
        sys.exit(f'not found: {root}')
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = [f for f in sorted(d.iterdir()) if f.suffix.lower() in EXT]
        if not frames:
            continue
        t = d / TRUTH
        out.append((d.name, frames,
                    t.read_text(encoding='utf-8') if t.exists() else None))
    return out


def self_test():
    """A scorer that cannot reproduce a hand-computable number cannot be
    trusted on an unknown one. Two scoring bugs in this project have already
    produced plausible wrong results."""
    from verify_model import cer, levenshtein
    cases = [
        ('abcde', 'abcde', 0.0,  'identical'),
        ('abcde', 'abcdX', 0.2,  'one substitution in five'),
        ('abcde', 'abcd',  0.2,  'one deletion in five'),
        ('abcde', 'abcdef', 0.2, 'one insertion in five'),
        ('abcde', '',      1.0,  'nothing read at all'),
    ]
    bad = 0
    for g, h, want, why in cases:
        got = cer(g, h)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f'  {"ok  " if ok else "FAIL"} cer({g!r:8}, {h!r:8}) = {got:.4f}'
              f'   expected {want:.4f}   {why}')
    assert levenshtein(list('abc'), list('abd')) == 1
    print('  ok   levenshtein on token lists')
    # A longer real-shaped case, computed by hand: 3 edits in 20 characters.
    g = 'a' * 20
    h = 'b' * 3 + 'a' * 17
    assert abs(cer(g, h) - 0.15) < 1e-9, cer(g, h)
    print('  ok   3 edits in 20 characters -> 0.1500')
    if bad:
        sys.exit('self-test FAILED — do not trust this harness')
    print('\nself-test passed')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root_dir', nargs='?',
                    help='folder of per-article subfolders (e.g. work\\eval)')
    ap.add_argument('--root', default=os.getenv('SINHALA_ROOT'),
                    help='model root, e.g. E:\\RP\\corpus\\Sinhala_OCR_Correction_v2')
    ap.add_argument('--modes', default='page,cols',
                    help='which read paths to score (default both)')
    ap.add_argument('--min-p75', type=float, default=14.0, dest='min_p75')
    ap.add_argument('--save-text', metavar='DIR', default=None,
                    help='write the OCR and corrected text per article')
    ap.add_argument('--diff', action='store_true',
                    help='after scoring, show where the transcription and the '
                         'OCR disagree — to catch YOUR typos, not to adopt '
                         'the machine\'s')
    ap.add_argument('--dry-run', action='store_true',
                    help='report layout and coverage only, no OCR, no model')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()

    if a.self_test:
        sys.path.insert(0, str(HERE))
        from core import config                                  # noqa: F401
        return self_test()
    if not a.root_dir:
        ap.error('give the folder of article subfolders, or --self-test')

    from core import config
    if a.root:
        config.set_root(a.root)
    from core.imaging import imread_upright, glyph_p75, closeup_scale
    from core.textutils import norm, strong_dedup, vote_lines, sentences
    from layers.l3_segment import layout as L

    arts = articles(a.root_dir)
    if not arts:
        sys.exit(f'no article subfolders with images under {a.root_dir}')

    missing = [n for n, _f, t in arts if not (t and t.strip())]
    print(f'{len(arts)} articles, {len(arts) - len(missing)} with a '
          f'transcription')
    if missing:
        print(f'  no {TRUTH} in: {", ".join(missing)}')

    # ---- stage 1: layout, crops, coverage --------------------------------
    plans = []
    for name, frames, truth in arts:
        crops, cols, an0 = [], [], None
        for f in frames:
            img = imread_upright(str(f))
            if img is None:
                continue
            an = L.analyse(img, min_p75=a.min_p75)
            if not an['applicable']:
                print(f'  {name}/{f.name}: REFUSED — {an["reason"]}')
                continue
            x1, y1, x2, y2 = an['crop']
            up = an['upright']
            crops.append(up[y1:y2, x1:x2])
            clipped = set(map(tuple, an['clipped_columns']))
            whole = [b for b in an['columns'] if tuple(b) not in clipped]
            cols.append([up[y1:y2, b[0]:b[1]] for b in sorted(whole)])
            an0 = an0 or an
        if not crops:
            print(f'{name}: no usable frames')
            continue
        plans.append(dict(name=name, truth=truth, crops=crops, cols=cols,
                          an=an0, p75=glyph_p75(crops[0]) or 0.0))

    print(f'\n{"article":14} {"p75":>5} {"cols":>5} {"clip":>5} {"lines":>6} '
          f'{"frames":>7} {"truth ch":>9}')
    print('-' * 60)
    for p in plans:
        an = p['an']
        clip = ''.join(c for c, v in zip('LR', (an['left_open'],
                                                an['right_open'])) if v) or '-'
        print(f'{p["name"][:14]:14} {p["p75"]:5.1f} {an["n_columns"]:5d} '
              f'{clip:>5} {an["lines_in_block"]:6.1f} {len(p["crops"]):7d} '
              f'{len(flat(p["truth"] or "")):9d}')

    clipped_any = [p['name'] for p in plans
                   if p['an']['left_open'] or p['an']['right_open']]
    if clipped_any:
        print(f'\n*** {len(clipped_any)} article(s) have a column off the '
              f'frame edge: {", ".join(clipped_any)}.\n    Their CER measures '
              f'COVERAGE, not accuracy — the text is not in the photo.\n    '
              f'Re-capture them or report them separately. ***')

    if a.dry_run:
        print('\n--dry-run: stopping before OCR.')
        return 0

    # ---- stage 2: OCR + mT5 ----------------------------------------------
    import pytesseract
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    from layers.l4b_body.body import BodyReader
    from verify_model import cer, levenshtein

    if getattr(config, 'TESSERACT_EXE', None):
        pytesseract.pytesseract.tesseract_cmd = str(config.TESSERACT_EXE)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nloading mT5 from {config.MT5_PLAIN}  (device {dev})')
    tok = AutoTokenizer.from_pretrained(str(config.MT5_PLAIN))
    mdl = MT5ForConditionalGeneration.from_pretrained(
        str(config.MT5_PLAIN)).to(dev).eval()
    body = BodyReader(tok, mdl, dev, pytesseract)

    def wer(g, h):
        return levenshtein(g.split(), h.split()) / max(1, len(g.split()))

    def ocr(img, cfg):
        """One image -> normalised text, at the deployed adaptive scale."""
        s = closeup_scale(glyph_p75(img) or 0.0)
        r = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) \
            if abs(s - 1.0) > 1e-3 else img
        t = pytesseract.image_to_string(cv2.cvtColor(r, cv2.COLOR_BGR2RGB),
                                        lang=config.TESS_LANG, config=cfg)
        return '\n'.join(norm(x) for x in t.split('\n') if x.strip())

    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    out_dir = Path(a.save_text) if a.save_text else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in plans:
        gt = flat(p['truth'] or '')
        for mode in modes:
            per_frame = []
            for i in range(len(p['crops'])):
                if mode == 'page':
                    t = ocr(p['crops'][i], config.TESS_CONFIG_PAGE)
                else:
                    # Each whole column on its own, psm 6, left to right.
                    parts = [ocr(c, config.TESS_CONFIG) for c in p['cols'][i]]
                    t = '\n'.join(x for x in parts if x.strip())
                if t.strip():
                    per_frame.append(t)
            raw = strong_dedup(vote_lines(per_frame) if len(per_frame) > 1
                               else (per_frame[0] if per_frame else ''))
            cor = strong_dedup(' '.join(body.correct_lines(sentences(raw)))) \
                if raw.strip() else ''
            rf, cf = flat(raw), flat(cor)
            rows.append(dict(
                name=p['name'], mode=mode, n=len(cf),
                raw_cer=cer(gt, rf) if gt else float('nan'),
                raw_wer=wer(gt, rf) if gt else float('nan'),
                cor_cer=cer(gt, cf) if gt else float('nan'),
                cor_wer=wer(gt, cf) if gt else float('nan'),
                raw=raw, cor=cor))
            print(f'  {p["name"][:14]:14} {mode:5}  raw {len(rf):5d} ch  '
                  f'corrected {len(cf):5d} ch')
            if out_dir:
                (out_dir / f'{p["name"]}_{mode}_raw.txt').write_text(
                    raw, encoding='utf-8')
                (out_dir / f'{p["name"]}_{mode}_corrected.txt').write_text(
                    cor, encoding='utf-8')

    scored = [r for r in rows if not np.isnan(r['cor_cer'])]
    if not scored:
        print(f'\nNothing scored — no {TRUTH} files. Text written to '
              f'{out_dir}.' if out_dir else
              f'\nNothing scored — no {TRUTH} files. Add --save-text to keep '
              f'the OCR output.')
        return 0

    print('\n' + '=' * 78)
    print('END-TO-END, against hand transcriptions   (lower is better)')
    print('=' * 78)
    print(f'{"article":14} {"mode":6} {"chars":>7} {"OCR CER":>9} '
          f'{"OCR WER":>9} {"mT5 CER":>9} {"mT5 WER":>9}')
    print('-' * 78)
    for r in scored:
        print(f'{r["name"][:14]:14} {r["mode"]:6} {r["n"]:7d} '
              f'{r["raw_cer"]:9.4f} {r["raw_wer"]:9.4f} '
              f'{r["cor_cer"]:9.4f} {r["cor_wer"]:9.4f}')

    def summarise(vals):
        if len(vals) < 2:
            return f'{vals[0]:.4f}' if vals else '-'
        return (f'{statistics.mean(vals):.4f} +- '
                f'{statistics.stdev(vals):.4f}')

    print('\n' + '-' * 78)
    for mode in modes:
        v = [r for r in scored if r['mode'] == mode]
        if not v:
            continue
        print(f'{mode:6} n={len(v)}   OCR CER {summarise([x["raw_cer"] for x in v]):>18}'
              f'   mT5 CER {summarise([x["cor_cer"] for x in v]):>18}')

    # ---- the paired comparison, which is the point of two modes ----------
    if len(modes) == 2:
        m1, m2 = modes
        pairs = [(r1, r2) for r1 in scored if r1['mode'] == m1
                 for r2 in scored if r2['mode'] == m2 and r2['name'] == r1['name']]
        if len(pairs) >= 2:
            d = [r1['cor_cer'] - r2['cor_cer'] for r1, r2 in pairs]
            md, sd = statistics.mean(d), statistics.stdev(d)
            se = sd / (len(d) ** 0.5)
            ci = 1.96 * se
            better = m2 if md > 0 else m1
            print(f'\nPAIRED, same articles: mT5 CER({m1}) - mT5 CER({m2}) = '
                  f'{md:+.4f}  95% CI [{md - ci:+.4f}, {md + ci:+.4f}]  n={len(d)}')
            if abs(md) <= ci:
                print(f'  The interval contains zero. On this evidence the two '
                      f'read paths are\n  indistinguishable — keep the simpler '
                      f'one ({m1}) and say so.')
            else:
                print(f'  Zero is outside the interval: {better} is better by '
                      f'{abs(md):.4f} CER.\n  That is a decision, not an '
                      f'impression.')
        else:
            print(f'\nNeed at least 2 articles scored in both modes for the '
                  f'paired comparison.')

    if a.diff:
        print('\n' + '=' * 78)
        print('WHERE YOUR TRANSCRIPTION AND THE OCR DISAGREE')
        print('This is to catch YOUR typing slips. Do NOT copy the OCR into')
        print('the transcription — that biases CER downwards by construction.')
        print('=' * 78)
        import difflib
        for p in plans:
            r = next((x for x in scored if x['name'] == p['name']), None)
            if not r:
                continue
            gt, hy = flat(p['truth']).split(), flat(r['cor']).split()
            sm = difflib.SequenceMatcher(None, gt, hy)
            print(f'\n{p["name"]}:')
            shown = 0
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal' or shown >= 15:
                    continue
                print(f'  truth: {" ".join(gt[i1:i2]) or "-"}')
                print(f'  read : {" ".join(hy[j1:j2]) or "-"}')
                shown += 1

    print(f"""
READING IT

  * mT5 CER should be BELOW OCR CER on most articles. Where it is not, the
    corrector is hurting that article — count how often, and say so.
  * A CER above ~0.5 means the crop or the transcription is wrong, not that
    the system is bad. Open the --save-text output before believing it.
  * Report the MEAN AND THE SPREAD, never the best article.
""")
    return 0


if __name__ == '__main__':
    sys.exit(main())

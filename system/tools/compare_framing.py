#!/usr/bin/env python3
"""
compare_framing.py — does reading the WHOLE article cost accuracy?

THE QUESTION
------------
The distance test showed the whole article fits at glyph_p75 25 and not at 29
(see Work/Ishara/Distance_Test_Result.md). 25 is exactly CAPTURE_MIN_GLYPH_P75,
the measured resolution floor. What is not known is whether OCR at 25 is as
good as OCR at the 33-38 you normally shoot at.

This measures that, on ONE COLUMN of the same article, against one hand
transcription.

WHY ONE COLUMN AND NOT THE WHOLE ARTICLE
----------------------------------------
Scoring whole crops would compare two different amounts of text: the close
framing physically lacks a column and a half, so it would lose on coverage no
matter how sharp it is, and the number would say nothing about resolution.
Coverage is already answered — by the layout tool, without any CER.

So both frames are cropped to the SAME single column and scored against a
transcription of that column. Same words, different pixel density: exactly the
question. It is also about twenty minutes of typing instead of two hours.

    python tools\\compare_framing.py --root E:\\RP\\corpus\\Sinhala_OCR_Correction_v2 ^
        --gt Work\\Ishara\\article_truth.txt ^
        work\\71a97929 work\\14f7798c

THE CONFOUND IT AVOIDS
----------------------
CLOSEUP_OCR_SCALE = 0.40 is FIXED and was chosen by eye on a frame at
glyph_p75 33, giving an effective glyph of about 13 px after downscaling.
Applied unchanged to a glyph_p75 25 frame it gives about 10 px, below the
~11 px at which diacritics disappear in this project's own measurements. A
comparison using the fixed scale would show the wider framing is terrible for
a reason that has nothing to do with framing.

So every frame is read TWICE:

    fixed   scale = CLOSEUP_OCR_SCALE                     (what ships today)
    auto    scale = CLOSEUP_TARGET_GLYPH / crop's p75      (same effective
                                                            glyph at any
                                                            distance)

Four numbers, not two. If `auto` beats `fixed` at the wider framing, that is
the answer to a second question you also have not measured.

WHAT THIS DOES NOT CHANGE
-------------------------
Nothing in core/ or layers/. The deployed path is untouched until this
measurement says what to change it to. Everything here is computed in this
file, deliberately, so that running the experiment cannot alter the system it
is measuring.

The mT5 correction is NOT reimplemented — it calls BodyReader.correct_lines()
so generation settings are identical to the research path by construction.
CER is imported from verify_model.py for the same reason.

Needs cv2, numpy, pytesseract, torch, transformers, and --root pointing at the
models. --dry-run needs none of those.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# The one number this file introduces, and it is INHERITED, not measured:
# CLOSEUP_OCR_SCALE 0.40 was chosen by eye on a frame whose glyph_p75 was 33.
# 0.40 x 33 = 13.2. Holding that product constant is what "auto" means. The
# point of this experiment is to find out whether 13 is the right target at
# all, so do not promote this to core/config.py on the strength of its
# appearance here.
CLOSEUP_TARGET_GLYPH = 13.2


def expand(patterns):
    """Group frames by the folder they came from — one folder per capture."""
    groups = []
    for p in patterns:
        q = Path(p)
        if q.is_dir():
            fs = [f for f in sorted(q.iterdir()) if f.suffix.lower() in EXT]
            if fs:
                groups.append((q.name, fs))
        elif q.suffix.lower() in EXT:
            groups.append((q.stem, [q]))
    return groups


def flat(s):
    """Collapse all whitespace. Line breaks are an OCR artefact, not content,
    and a transcription cannot be expected to reproduce them."""
    return re.sub(r'\s+', ' ', s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames', nargs='+',
                    help='one folder per capture (work\\<job>), or files')
    ap.add_argument('--gt', default=None,
                    help='UTF-8 text file: the article body, corrected by '
                         'hand. Omit it to run OCR and write the outputs '
                         'without scoring — that is how you produce a '
                         'starting point to correct.')
    ap.add_argument('--root', default=os.getenv('SINHALA_ROOT'),
                    help='where the models live (the corpus folder)')
    ap.add_argument('--min-p75', type=float, default=14.0, dest='min_p75',
                    help='close-up gate override for MEASUREMENT. Default 14 '
                         'so the wider framings are not refused — they are '
                         'the whole point here.')
    ap.add_argument('--dry-run', action='store_true',
                    help='crops and scales only. No Tesseract, no mT5, no '
                         'models. Run this first: it is instant and it shows '
                         'whether the crops are right before you spend ten '
                         'minutes on the real pass.')
    ap.add_argument('--lines', type=int, default=0, dest='n_lines',
                    help='read only the first N text lines of the column, '
                         'counted in line pitches from the top of the block. '
                         'THIS IS HOW THE COMPARISON IS MADE FAIR: a closer '
                         'frame holds fewer lines of the same column, so '
                         'without it the close capture is punished for text '
                         'it physically cannot see. Trim the ground truth to '
                         'the same N lines.')
    ap.add_argument('--column', type=int, default=1,
                    help='which WHOLE column to read, counting from the left, '
                         '1-based. Default 1. Use 0 for the entire crop, but '
                         'see the module docstring: a whole-crop comparison '
                         'measures coverage, not resolution.')
    ap.add_argument('--save-text', metavar='DIR', default=None,
                    help='write every OCR and corrected output to files, so '
                         'a bad CER can be read rather than guessed at')
    a = ap.parse_args()

    from core import config
    if a.root:
        config.set_root(a.root)
    from core.imaging import imread_upright, glyph_p75, sharpness
    from core.textutils import norm, strong_dedup, vote_lines, sentences
    from layers.l3_segment import layout as L

    gt = flat(Path(a.gt).read_text(encoding='utf-8')) if a.gt else ''
    if a.gt and not gt:
        print(f'{a.gt} is empty')
        return 2
    if gt:
        print(f'ground truth: {len(gt)} chars, {len(gt.split())} words  '
              f'({a.gt})\n')
    else:
        print('NO --gt: running OCR and writing the text, but scoring '
              'nothing.\n')
        if not a.save_text and not a.dry_run:
            print('  ...which is pointless without --save-text. Add it.')
            return 2

    groups = expand(a.frames)
    if not groups:
        print('no frames matched')
        return 2

    # ---- stage 1: crop every frame, decide both scales -------------------
    plans = []
    for name, files in groups:
        crops = []
        info = None
        for f in files:
            img = imread_upright(str(f))
            if img is None:
                continue
            an = L.analyse(img, min_p75=a.min_p75)
            if not an['applicable']:
                print(f'  {name}/{f.name}: REFUSED — {an["reason"]}')
                continue
            x1, y1, x2, y2 = an['crop']
            if a.column:
                clipped = set(map(tuple, an['clipped_columns']))
                whole = [b for b in an['columns'] if tuple(b) not in clipped]
                if len(whole) < a.column:
                    print(f'  {name}/{f.name}: only {len(whole)} whole '
                          f'columns, cannot take #{a.column}')
                    continue
                b = whole[a.column - 1]
                x1, x2 = b[0], b[1]
            if a.n_lines:
                # +0.35 pitch so the Nth line's descenders are not sliced off
                y2 = min(y2, y1 + int((a.n_lines + 0.35) * an['pitch']))
            c = an['upright'][y1:y2, x1:x2]
            if c.size:
                crops.append(c)
                info = info or an
        if not crops:
            print(f'{name}: no usable frames')
            continue
        p75 = glyph_p75(crops[0]) or 0.0
        auto = float(np.clip(CLOSEUP_TARGET_GLYPH / p75, config.OCR_SCALE_MIN,
                             config.OCR_SCALE_MAX)) if p75 else 1.0
        # Sharpness measured AFTER scaling to the same effective glyph
        # height. Comparing it on the raw crops would just re-measure
        # distance; at a matched glyph size it measures focus, which is the
        # thing that can silently ruin this experiment.
        sh = float(np.mean([sharpness(cv2.resize(c, None, fx=auto, fy=auto,
                                                 interpolation=cv2.INTER_AREA)
                                      if abs(auto - 1) > 1e-3 else c)
                            for c in crops]))
        plans.append(dict(name=name, crops=crops, an=info, p75=p75, sharp=sh,
                          fixed=config.CLOSEUP_OCR_SCALE, auto=round(auto, 3)))

    print(f'\nreading column #{a.column} of each capture'
          if a.column else '\nreading the whole crop of each capture')
    print(f'{"capture":12} {"p75":>5} {"cols":>4} {"clip":>5} {"blines":>7} '
          f'{"crop":>12} {"fixed":>6} {"->px":>5} {"auto":>6} {"->px":>5} '
          f'{"sharp@px":>9}')
    print('-' * 92)
    for p in plans:
        an = p['an']
        h, w = p['crops'][0].shape[:2]
        clip = ''.join(c for c, v in zip('LR', (an['left_open'],
                                                an['right_open'])) if v) or '-'
        print(f'{p["name"][:12]:12} {p["p75"]:5.1f} {an["n_columns"]:4d} '
              f'{clip:>5} {an["lines_in_block"]:7.1f} {f"{w}x{h}":>12} '
              f'{p["fixed"]:6.2f} {p["p75"]*p["fixed"]:5.1f} '
              f'{p["auto"]:6.2f} {p["p75"]*p["auto"]:5.1f} {p["sharp"]:9.0f}')
    print('\n"->px" is the glyph height Tesseract actually sees. Diacritics '
          'die below\nabout 11 px — that is the whole reason `auto` exists.')
    if len(plans) > 1:
        sh = [p['sharp'] for p in plans]
        if max(sh) > 3.0 * max(min(sh), 1e-9):
            worst = min(plans, key=lambda p: p['sharp'])
            print(f'\n*** WARNING: focus differs by {max(sh)/max(min(sh),1e-9):.1f}x '
                  f'at a matched glyph size\n    (worst: {worst["name"]} at '
                  f'{worst["sharp"]:.0f}). CER would then be measuring FOCUS,\n'
                  '    not framing. Re-capture the soft one before believing '
                  'anything below. ***')

    if not a.n_lines and len(plans) > 1:
        bl = [p['an']['lines_in_block'] for p in plans]
        if max(bl) - min(bl) > 1.0:
            print(f'\n*** WARNING: the captures hold different amounts of the '
                  f'column\n    ({min(bl):.1f} to {max(bl):.1f} lines). CER '
                  'would then be measuring COVERAGE, not\n    accuracy. Use '
                  f'--lines {int(min(bl)) - 1} and trim the ground truth to '
                  'match. ***')

    if a.save_text:
        # Save the crop that was actually read -- in dry-run too, because
        # this is exactly when you want to look at it. If two captures
        # disagree wildly, the first thing to check is whether they are the
        # same physical column, and that is something you look at.
        d = Path(a.save_text)
        d.mkdir(parents=True, exist_ok=True)
        for p in plans:
            cv2.imwrite(str(d / f'{p["name"]}_col{a.column}.jpg'),
                        p['crops'][0], [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'\ncrops written to {d} — OPEN THEM and check every capture '
              'shows the SAME\ncolumn of the SAME article. Nothing below is '
              'meaningful if they do not.')

    if a.dry_run:
        print('\n--dry-run: stopping before OCR. Check the crops in '
              'tools\\out\\distance first,\nthen run again without it.')
        return 0

    # ---- stage 2: OCR + mT5 ---------------------------------------------
    import pytesseract
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    from layers.l4b_body.body import BodyReader
    from verify_model import cer, levenshtein     # the SAME metric as the
    #                                               research harness, imported
    #                                               rather than reimplemented

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

    # psm 6 for a single column, psm 3 only for a multi-column crop -- which
    # is what core/config.py already says these two are for. Getting this
    # wrong is not a small quality difference: psm 3 does automatic page
    # segmentation and, on the 277x445 single-column crop of 14f7798c at
    # scale 0.49, it returned ZERO characters where psm 6 returned 784. It
    # also under-read the sharp capture, 561 characters against 693.
    tess_cfg = config.TESS_CONFIG_PAGE if not a.column else config.TESS_CONFIG
    print(f'tesseract: {tess_cfg}')

    out_dir = Path(a.save_text) if a.save_text else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in plans:
        for mode in ('fixed', 'auto'):
            s = p[mode]
            per_frame = []
            for c in p['crops']:
                r = cv2.resize(c, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA) \
                    if abs(s - 1.0) > 1e-3 else c
                t = pytesseract.image_to_string(
                    cv2.cvtColor(r, cv2.COLOR_BGR2RGB),
                    lang=config.TESS_LANG, config=tess_cfg)
                t = '\n'.join(norm(x) for x in t.split('\n') if x.strip())
                if t:
                    per_frame.append(t)
            raw = strong_dedup(vote_lines(per_frame) if len(per_frame) > 1
                               else (per_frame[0] if per_frame else ''))
            corrected = strong_dedup(' '.join(
                body.correct_lines(sentences(raw)))) if raw.strip() else ''
            r_flat, c_flat = flat(raw), flat(corrected)
            rows.append(dict(
                name=p['name'], mode=mode, scale=s, px=p['p75'] * s,
                raw_cer=cer(gt, r_flat) if gt else float('nan'),
                raw_wer=wer(gt, r_flat) if gt else float('nan'),
                cor_cer=cer(gt, c_flat) if gt else float('nan'),
                cor_wer=wer(gt, c_flat) if gt else float('nan'),
                raw_n=len(r_flat), cor_n=len(c_flat)))
            print(f'  {p["name"]:12} {mode:5}  scale {s:.2f}  '
                  f'raw {len(r_flat):5d} ch  corrected {len(c_flat):5d} ch')
            if out_dir:
                (out_dir / f'{p["name"]}_{mode}_raw.txt').write_text(
                    raw, encoding='utf-8')
                (out_dir / f'{p["name"]}_{mode}_corrected.txt').write_text(
                    corrected, encoding='utf-8')

    # ---- the table -------------------------------------------------------
    if not gt:
        print(f'\nText written to {out_dir}. Correct the best one against '
              'the newspaper,\nsave it as your ground truth, and run again '
              'with --gt.')
        return 0
    print('\n' + '=' * 88)
    print('CER / WER against the hand transcription   (lower is better)')
    print('=' * 88)
    print(f'{"capture":12} {"mode":6} {"scale":>6} {"->px":>5} {"chars":>7} '
          f'{"OCR CER":>8} {"OCR WER":>8} {"mT5 CER":>8} {"mT5 WER":>8}')
    print('-' * 88)
    for r in rows:
        print(f'{r["name"][:12]:12} {r["mode"]:6} {r["scale"]:6.2f} '
              f'{r["px"]:5.1f} {r["cor_n"]:7d} '
              f'{r["raw_cer"]:8.4f} {r["raw_wer"]:8.4f} '
              f'{r["cor_cer"]:8.4f} {r["cor_wer"]:8.4f}')

    print("""
HOW TO READ IT

  Compare the BEST row of each capture, not fixed against fixed. The question
  is what each framing can achieve, not what today's constant does to it.

  * A wide framing whose best mT5 CER is close to the close framing's
    -> the whole article can be read at no real cost. Lower CLOSEUP_MIN_P75
       to 25 and aim the guidance at 25-28.
  * A wide framing that is clearly worse at BOTH scales
    -> the conflict is real and now measured. Leave the guidance alone and
       report it; panning is the answer, after the chapters.
  * `auto` much better than `fixed` anywhere
    -> the fixed 0.40 is costing accuracy on its own, independently of this
       whole question, and should become a target glyph height.

  A CER above ~0.5 usually means the crop was wrong or the transcription does
  not match the columns that were read. Read the files from --save-text before
  believing any such number.""")
    return 0


if __name__ == '__main__':
    sys.exit(main())

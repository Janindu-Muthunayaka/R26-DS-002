"""
C1 benchmark: headline (title) OCR on rendered news headlines with EXACT
ground truth.

Real Dinamina headlines (Work/Nadee/corpus/articles_fixed.jsonl) are rendered
as bold newspaper headlines in four Unicode Sinhala fonts, clean, degraded and
hard (same degradations as bench_c2_ocr). Each headline is read three ways:

  baseline   stock Tesseract 'sin' (TESS_LANG), psm 6, on the unprocessed crop
  deployed   Layer 4A as shipped: CRAFT line detection + stitching +
             binarize_strip -> Tesseract sin_raw, psm 13   (title.py)
  mat        the same binarized strip, skeletonised (mat.skeleton_on_binarized)
             -> Janindu's sin_custom model, psm 13   (research variant; the
             deployed system only DISPLAYS the skeleton)

Reports CER / WER per method, per font and per condition, and exact-match
rate (the whole headline right).

    system/.venv/Scripts/python.exe system/tools/bench/bench_c1_title.py [--n 60]
Output: system/docs/eval/c1_title.json and c1_title_rows.csv
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import bench_c2_ocr as c2                      # noqa: E402  (corpus, degradations, scoring)

OUT = c2.OUT
FONTS = {
    'Nirmala UI Bold': (r'C:\Windows\Fonts\NirmalaB.ttf', 0),
    'Iskoola Pota': (c2.FONTS['Iskoola Pota'], 1),          # stroke 1 = newspaper bold
    'Dinamina Unicode': (c2.FONTS['Dinamina Unicode'], 1),
    'Malithi Web': (c2.FONTS['Malithi Web'], 1),
}
HEAD_PX = 58
HEAD_W = 1100


def headlines(n):
    rows = [json.loads(l) for l in open(c2.CORPUS, encoding='utf-8')]
    random.Random(c2.SEED + 1).shuffle(rows)
    out = []
    for r in rows:
        h = c2.clean(r.get('headline') or '')
        if 12 <= len(h) <= 90 and sum('\u0d80' <= ch <= '\u0dff' for ch in h) > 0.6 * len(h):
            out.append({'article_id': r['article_id'], 'text': h})
        if len(out) == n:
            break
    return out


def render_headline(text, font_path, stroke):
    font = ImageFont.truetype(font_path, HEAD_PX, layout_engine=ImageFont.Layout.RAQM)
    probe = ImageDraw.Draw(Image.new('L', (10, 10)))
    lines = c2.wrap(probe, text, font, HEAD_W)
    lh = int(HEAD_PX * 1.5)
    img = Image.new('L', (HEAD_W + 60, lh * len(lines) + 50), 255)
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((30, 25 + i * lh), l, font=font, fill=0, stroke_width=stroke, stroke_fill=0)
    return img


def read_all(pil):
    import pytesseract
    from core.config import TESS_LANG
    from layers.l4a_title.mat import skeleton_on_binarized
    from layers.l4a_title.title import _ocr_strip
    from layers.l4a_title.title_extractor import binarize_strip, extract_and_stitch_title
    img = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    out, secs = {}, {}

    t = time.perf_counter()
    out['baseline'] = pytesseract.image_to_string(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lang=TESS_LANG, config='--oem 1 --psm 6')
    secs['baseline'] = time.perf_counter() - t

    t = time.perf_counter()
    res = extract_and_stitch_title(img)
    strip = res[0] if isinstance(res, tuple) else res
    if strip is None:
        out['deployed'] = out['mat'] = ''
        secs['deployed'] = time.perf_counter() - t
        secs['mat'] = 0.0
        return out, secs
    _, binarized = binarize_strip(strip, is_dark_bg=False)
    out['deployed'], _ = _ocr_strip(binarized, 'sin_raw', '13')
    secs['deployed'] = time.perf_counter() - t

    t = time.perf_counter()
    out['mat'], _ = _ocr_strip(skeleton_on_binarized(binarized), 'sin_custom', '13')
    secs['mat'] = time.perf_counter() - t
    return out, secs


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    args = ap.parse_args()
    rng = random.Random(c2.SEED)
    fonts = list(FONTS)
    methods = ('baseline', 'deployed', 'mat')
    rows = []
    for i, h in enumerate(headlines(args.n)):
        font = fonts[i % len(fonts)]
        base = render_headline(h['text'], *FONTS[font])
        for cond, img in (('clean', base), ('degraded', c2.degrade(base, rng)),
                          ('hard', c2.degrade(base, rng, 'hard'))):
            texts, secs = read_all(img)
            ref = c2.score_norm(h['text'])
            row = {'article_id': h['article_id'], 'font': font, 'condition': cond,
                   'ref_chars': len(ref), 'ref_words': len(ref.split()), 'ref': h['text']}
            for m in methods:
                hyp = c2.score_norm(texts[m])
                row[f'cer_{m}'] = round(min(1.0, c2.cer(hyp, ref)), 4)
                row[f'wer_{m}'] = round(min(1.0, c2.wer(hyp, ref)), 4)
                row[f'exact_{m}'] = hyp == ref
                row[f'sec_{m}'] = round(secs[m], 2)
                row[f'text_{m}'] = texts[m].strip()
            rows.append(row)
            print(f"{i + 1:>2} {font:<17} {cond:<8} CER base {row['cer_baseline']:.3f}  "
                  f"deployed {row['cer_deployed']:.3f}  mat {row['cer_mat']:.3f}", flush=True)

    def agg(rs):
        chars = sum(r['ref_chars'] for r in rs)
        words = sum(r['ref_words'] for r in rs)
        return {'n': len(rs), **{m: {
            'cer': round(sum(r[f'cer_{m}'] * r['ref_chars'] for r in rs) / chars, 4),
            'wer': round(sum(r[f'wer_{m}'] * r['ref_words'] for r in rs) / words, 4),
            'exact_match': round(sum(r[f'exact_{m}'] for r in rs) / len(rs), 4),
            'median_sec': round(statistics.median(r[f'sec_{m}'] for r in rs), 2),
        } for m in methods}}

    res = {'run_at': time.strftime('%Y-%m-%d %H:%M'), 'headlines': args.n, 'head_px': HEAD_PX,
           'fonts': fonts, 'overall': agg(rows),
           'by_condition': {c: agg([r for r in rows if r['condition'] == c])
                            for c in ('clean', 'degraded', 'hard')},
           'by_font': {f: agg([r for r in rows if r['font'] == f])
                       for f in fonts if any(r['font'] == f for r in rows)}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'c1_title.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'c1_title_rows.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: res[k] for k in ('overall', 'by_condition')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

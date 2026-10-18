"""
C2 benchmark: body OCR + mT5 correction on rendered news text with EXACT
ground truth.

Real Sinhala news (Work/Nadee/corpus/articles_fixed.jsonl, Dinamina) is
rendered as a newspaper-style column in four Unicode Sinhala fonts, clean and
"camera-degraded" (slight rotation, blur, sensor noise, JPEG). Each image goes
through the DEPLOYED reading path of Layer 4B:

    rescale_to_optimum -> Tesseract (TESS_LANG, TESS_CONFIG) -> norm
    -> strong_dedup -> sentences -> mT5 correct_lines -> strong_dedup

and is scored against the text that was rendered.

Reports CER / WER for raw OCR and after mT5, per font and per condition, and
a word-level CORRECTION F1:
    TP  a word OCR got wrong that mT5 made right
    FP  a word OCR got right that mT5 made wrong
    FN  a word OCR got wrong that mT5 left wrong
(words aligned to the ground truth by edit distance).

This is synthetic print, not a phone photo: it isolates OCR + correction
from capture and segmentation. Those are measured end to end in bench_e2e.

    system/.venv/Scripts/python.exe system/tools/bench/bench_c2_ocr.py [--n 40]
Output: system/docs/eval/c2_ocr.json and c2_ocr_rows.csv
"""
import argparse
import csv
import io
import json
import os
import random
import re
import statistics
import sys
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SYS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, SYS)
ROOT = os.path.dirname(SYS)
OUT = os.path.join(SYS, 'docs', 'eval')
CORPUS = os.path.join(ROOT, 'Work', 'Nadee', 'corpus', 'articles_fixed.jsonl')
FONT_DIR = r'E:\Sliit\Research\Dataset\01 Font pack full\2000 Sinhala Fonts pack - unicodesinhala.com'
FONTS = {
    'Iskoola Pota': os.path.join(FONT_DIR, 'Iskoola Pota.ttf'),
    'Dinamina Unicode': os.path.join(FONT_DIR, 'Dinamina Unicode for Web applications.ttf'),
    'Malithi Web': os.path.join(FONT_DIR, 'Malithi Web.ttf'),
    'Nirmala UI': r'C:\Windows\Fonts\Nirmala.ttf',
}
FONT_PX = 34          # body-text size, close to a phone capture of newsprint
COL_W = 900           # one newspaper column
SEED = 7


# ---- ground truth ----------------------------------------------------------
def clean(s):
    s = s.replace('\xad', '')                       # soft hyphens from the web page
    s = re.sub(r'\s+', ' ', s).strip()
    return unicodedata.normalize('NFC', s)


def passages(n, chars=320):
    rows = [json.loads(l) for l in open(CORPUS, encoding='utf-8')]
    random.Random(SEED).shuffle(rows)
    out = []
    for r in rows:
        body = clean(r.get('raw_body') or '')
        body = re.sub(r'^[\d\s]+', '', body)         # leading page numbers
        if len(body) < chars:
            continue
        cut = body.find('.', chars)                  # end on a full sentence
        text = body[:cut + 1] if 0 < cut < chars + 200 else body[:chars]
        if sum('\u0d80' <= c <= '\u0dff' for c in text) < 0.6 * len(text):
            continue
        out.append({'article_id': r['article_id'], 'text': text})
        if len(out) == n:
            break
    return out


# ---- rendering -------------------------------------------------------------
def wrap(draw, text, font, width):
    lines, cur = [], ''
    for w in text.split(' '):
        t = (cur + ' ' + w).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(text, font_path):
    font = ImageFont.truetype(font_path, FONT_PX, layout_engine=ImageFont.Layout.RAQM)
    probe = ImageDraw.Draw(Image.new('L', (10, 10)))
    lines = wrap(probe, text, font, COL_W)
    lh = int(FONT_PX * 1.75)
    img = Image.new('L', (COL_W + 80, lh * len(lines) + 80), 255)
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((40, 40 + i * lh), l, font=font, fill=0)
    return img


LEVELS = {                     # blur sigma, noise sigma, JPEG quality, downscale
    'degraded': (1.1, 9, 60, 1.0),
    'hard': (1.5, 14, 45, 0.70),
}


def degrade(img, rng, level='degraded'):
    """A phone photo of newsprint, roughly: paper tone, tilt, defocus,
    sensor noise and JPEG. 'hard' also shrinks the text to about 24 px."""
    blur, noise, q, scale = LEVELS[level]
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    a = np.array(img).astype(np.float32)
    a = a * 0.80 + 30                                 # grey paper, grey ink
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    im = im.rotate(rng.uniform(-1.0, 1.0), resample=Image.BICUBIC, expand=True, fillcolor=225)
    im = im.filter(ImageFilter.GaussianBlur(blur))
    a = np.array(im).astype(np.float32) + np.random.default_rng(rng.randint(0, 9999)).normal(0, noise, im.size[::-1])
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert('L')


# ---- scoring ---------------------------------------------------------------
def score_norm(s):
    s = unicodedata.normalize('NFC', s or '')
    s = re.sub(r'[\u200b-\u200d\ufeff]', '', s)      # ZWJ / ZWNJ are not glyphs
    s = re.sub(r'[^\w\s]', ' ', s)                    # punctuation is not scored
    return re.sub(r'\s+', ' ', s).strip()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(h, r):
    return lev(h, r) / max(1, len(r))


def wer(h, r):
    return lev(h.split(), r.split()) / max(1, len(r.split()))


def align_words(hyp, ref):
    """For each reference word: True if the hypothesis has it at the aligned
    position (edit-distance backtrace over words)."""
    h, r = hyp.split(), ref.split()
    D = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h) + 1):
        D[i][0] = i
    for j in range(len(r) + 1):
        D[0][j] = j
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (h[i - 1] != r[j - 1]))
    ok = [False] * len(r)
    i, j = len(h), len(r)
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (h[i - 1] != r[j - 1]):
            ok[j - 1] = h[i - 1] == r[j - 1]
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return ok


# ---- the deployed read path ------------------------------------------------
class Reader:
    def __init__(self):
        import pytesseract
        import torch
        from transformers import AutoTokenizer, MT5ForConditionalGeneration
        from core.config import MT5_PLAIN
        from layers.l4b_body.body import BodyReader
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        tok = AutoTokenizer.from_pretrained(str(MT5_PLAIN))
        mdl = MT5ForConditionalGeneration.from_pretrained(str(MT5_PLAIN)).to(dev).eval()
        self.body = BodyReader(tok, mdl, dev, pytesseract)
        self.dev = dev

    def read(self, pil):
        from core.schemas import Article, Box, Region
        from core.textutils import strong_dedup
        img = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        box = Box(x1=0, y1=0, x2=w, y2=h)
        art = Article(index=0, box=box, regions=[Region(label='text', box=box)])
        t0 = time.perf_counter()
        art = self.body.read([img], art)
        t1 = time.perf_counter()
        art = self.body.correct(art)
        t2 = time.perf_counter()
        return strong_dedup(art.body_raw), art.body, t1 - t0, t2 - t1


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40, help='passages (each read clean, degraded and hard)')
    args = ap.parse_args()
    rng = random.Random(SEED)
    reader = Reader()
    fonts = list(FONTS)
    rows = []
    for i, p in enumerate(passages(args.n)):
        font = fonts[i % len(fonts)]
        base = render(p['text'], FONTS[font])
        for cond, img in (('clean', base), ('degraded', degrade(base, rng)),
                          ('hard', degrade(base, rng, 'hard'))):
            raw, cor, t_ocr, t_mt5 = reader.read(img)
            ref, r_raw, r_cor = score_norm(p['text']), score_norm(raw), score_norm(cor)
            a_raw, a_cor = align_words(r_raw, ref), align_words(r_cor, ref)
            rows.append({
                'article_id': p['article_id'], 'font': font, 'condition': cond,
                'ref_chars': len(ref), 'ref_words': len(ref.split()),
                'cer_raw': round(cer(r_raw, ref), 4), 'cer_mt5': round(cer(r_cor, ref), 4),
                'wer_raw': round(wer(r_raw, ref), 4), 'wer_mt5': round(wer(r_cor, ref), 4),
                'tp': sum((not x) and y for x, y in zip(a_raw, a_cor)),
                'fp': sum(x and (not y) for x, y in zip(a_raw, a_cor)),
                'fn': sum((not x) and (not y) for x, y in zip(a_raw, a_cor)),
                'ocr_sec': round(t_ocr, 2), 'mt5_sec': round(t_mt5, 2),
                'ref': p['text'], 'raw': raw, 'mt5': cor,
            })
            r = rows[-1]
            print(f"{i + 1:>2} {font:<17} {cond:<8} CER {r['cer_raw']:.3f} -> {r['cer_mt5']:.3f}  "
                  f"WER {r['wer_raw']:.3f} -> {r['wer_mt5']:.3f}", flush=True)

    def agg(rs):
        tp, fp, fn = (sum(r[k] for r in rs) for k in ('tp', 'fp', 'fn'))
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        # Corpus-level CER/WER: total edits / total reference length.
        chars = sum(r['ref_chars'] for r in rs)
        words = sum(r['ref_words'] for r in rs)
        return {
            'n': len(rs),
            'cer_raw': round(sum(r['cer_raw'] * r['ref_chars'] for r in rs) / chars, 4),
            'cer_mt5': round(sum(r['cer_mt5'] * r['ref_chars'] for r in rs) / chars, 4),
            'wer_raw': round(sum(r['wer_raw'] * r['ref_words'] for r in rs) / words, 4),
            'wer_mt5': round(sum(r['wer_mt5'] * r['ref_words'] for r in rs) / words, 4),
            'cer_mt5_better_in': sum(r['cer_mt5'] < r['cer_raw'] for r in rs),
            'cer_mt5_worse_in': sum(r['cer_mt5'] > r['cer_raw'] for r in rs),
            'correction': {'tp_fixed': tp, 'fp_broken': fp, 'fn_missed': fn,
                           'precision': round(p, 4), 'recall': round(rc, 4),
                           'f1': round(2 * p * rc / (p + rc), 4) if p + rc else 0.0},
        }

    res = {'run_at': time.strftime('%Y-%m-%d %H:%M'), 'device': reader.dev,
           'font_px': FONT_PX, 'passages': args.n, 'fonts': fonts,
           'overall': agg(rows),
           'by_condition': {c: agg([r for r in rows if r['condition'] == c])
                            for c in ('clean', 'degraded', 'hard')},
           'by_font': {f: agg([r for r in rows if r['font'] == f])
                       for f in fonts if any(r['font'] == f for r in rows)},
           'latency_sec': {'ocr_median': round(statistics.median(r['ocr_sec'] for r in rows), 2),
                           'mt5_median': round(statistics.median(r['mt5_sec'] for r in rows), 2)}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'c2_ocr.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'c2_ocr_rows.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

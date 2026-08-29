#!/usr/bin/env python3
"""
setup_project.py — create the Sinhala Reader structure inside an EXISTING repo.

SAFE BY DEFAULT:
  * never overwrites a file that already exists
  * never touches folders it did not create (Work/, README.md, .git/ ...)
  * --dry-run shows exactly what would happen and writes nothing

Usage from the repo root (the folder containing README.md):

    python setup_project.py --dry-run        # look first
    python setup_project.py                  # create
    python setup_project.py --force          # overwrite scaffold files only

By default the system is created in  ./system/ , leaving your research files
(Work/, README.md, notebooks) untouched. Use --into . to place it at the repo
root instead, or --into Work/system to nest it under Work.
"""
import argparse, sys, textwrap
from pathlib import Path

FILES = {}


def f(path, body):
    FILES[path] = textwrap.dedent(body).lstrip('\n')


# ============================================================ core/schemas.py
f('core/schemas.py', '''
    """
    THE CONTRACT between layers. Change nothing here without telling the team —
    every layer depends on these shapes.

    Flow:
        Frame[]  -> L2 select  -> Frame[]        (best frames only)
                 -> L3 segment -> Article[]      (boxes + region crops)
                 -> L4A titles -> Article.title  (OTHER MEMBER)
                 -> L4B body   -> Article.body   (Component 2, correction)
                 -> L5 assemble-> Document
                 -> L6 RAG/TTS -> audio          (OTHER MEMBER)
    """
    from __future__ import annotations
    from typing import Optional, List, Literal
    from pydantic import BaseModel, Field


    class Box(BaseModel):
        """Pixel box on a frame, top-left origin."""
        x1: float; y1: float; x2: float; y2: float

        @property
        def w(self) -> float: return self.x2 - self.x1
        @property
        def h(self) -> float: return self.y2 - self.y1


    class Frame(BaseModel):
        """One captured photo plus the quality measurements from Layer 2."""
        path: str
        width: int
        height: int
        sharpness: Optional[float] = None
        glyph_p90: Optional[float] = Field(
            None, description='p90 connected-component height, px')
        verdict: Literal['ok', 'warn', 'reject', 'unknown'] = 'unknown'
        note: str = ''


    class Region(BaseModel):
        """A labelled sub-area inside an article (from layout detection)."""
        box: Box
        label: Literal['title', 'text', 'image', 'other'] = 'text'


    class Article(BaseModel):
        """One article. Layers fill in different fields; none overwrite another's."""
        index: int                                  # reading order, 0-based
        box: Box
        regions: List[Region] = []

        # ---- Layer 4A writes these (OTHER MEMBER) ----
        title_raw: str = ''
        title: str = ''

        # ---- Layer 4B writes these (Component 2 — correction) ----
        body_raw: str = ''                          # OCR output, uncorrected
        body: str = ''                              # after mT5 correction

        # ---- diagnostics, any layer may set ----
        glyph_p90: Optional[float] = None
        ocr_scale: Optional[float] = None
        verdict: Literal['ok', 'warn', 'reject', 'unknown'] = 'unknown'
        note: str = ''


    class Document(BaseModel):
        """Layer 5 output — the whole page, ready for RAG/TTS."""
        source_frames: List[str] = []
        articles: List[Article] = []
        timings: dict = {}
        warnings: List[str] = []


    class CaptureResponse(BaseModel):
        """What the phone app receives."""
        document: Document
        audio_url: Optional[str] = None
        error: Optional[str] = None
    ''')

# ============================================================ core/config.py
f('core/config.py', '''
    """
    All tunable numbers in ONE place. Layers import from here — never hardcode.
    """
    import os
    from pathlib import Path

    PROJECT_ROOT = Path(os.getenv(
        'SINHALA_ROOT', r'D:/Sinhala_OCR_Correction_v2')).expanduser()

    # ---- models ----
    YOLO_WEIGHTS = [
        PROJECT_ROOT/'layout'/'runs'/'articles_full'/'weights'/'best.pt',
        PROJECT_ROOT/'layout'/'article_model_v1.pt',
    ]
    MT5_PLAIN = PROJECT_ROOT/'models'/'mt5_plain'

    # ---- capture quality (MEASURED, do not guess) --------------------------
    # Base glyphs read best at 22-30 px. Below ~22 px the dependent vowel signs
    # (pilla) fall under ~11 px and become unrecoverable, so no downstream
    # correction can help. Source: capture-resolution sweep.
    TARGET_GLYPH   = 24.0
    MIN_BASE_GLYPH = 22.0
    REJECT_BELOW   = MIN_BASE_GLYPH * 0.75     # 16.5 px

    # ---- OCR ----
    # Scale is CAPPED AT 1.0. Upscaling cannot restore detail that was never
    # captured: the sweep measured 2.0x -> CER 0.336 and 3.0x -> 0.659 against
    # 0.175 at the optimum.
    OCR_SCALE_MIN, OCR_SCALE_MAX = 0.15, 1.0
    TESS_CONFIG = '--oem 1 --psm 6'
    TESS_LANG   = 'sin'

    # ---- correction ----
    # no_repeat_ngram_size=6 measured 0.0847 -> 0.0515 CER on identical inputs
    # by suppressing seq2seq generation runaway on long lines.
    MT5_NUM_BEAMS = 4
    MT5_NO_REPEAT_NGRAM = 6
    MT5_MAX_LENGTH = 128

    # ---- frame selection (Layer 2) ----
    BURST_KEEP      = 3
    MIN_SHARPNESS   = 45.0
    SHARP_MIN_RATIO = 0.30     # drop frames far below the best

    # ---- detection (Layer 3) ----
    YOLO_CONF  = 0.40
    YOLO_IMGSZ = 1024
    MAX_ARTICLES = 8

    # ---- server ----
    HOST, PORT = '0.0.0.0', 8000
    WORK_DIR = Path(os.getenv('SINHALA_WORK', './work')).resolve()
    ''')

# ============================================================ core/imaging.py
f('core/imaging.py', '''
    """Shared image measurements. Every layer that needs glyph height uses THIS,
    so the number means the same thing everywhere."""
    import cv2, numpy as np
    from .config import TARGET_GLYPH, MIN_BASE_GLYPH, REJECT_BELOW
    from .config import OCR_SCALE_MIN, OCR_SCALE_MAX


    def to_gray(img):
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def glyph_p90(img):
        """p90 connected-component height in px — the base-glyph proxy."""
        g = to_gray(img)
        if g.size == 0:
            return None
        bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        _, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        h = stats[1:, cv2.CC_STAT_HEIGHT]
        h = h[(h > 2) & (h < 200)]
        return float(np.percentile(h, 90)) if len(h) else None


    def sharpness(img):
        return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())


    def scale_for_target(p90, target=TARGET_GLYPH):
        """Capped at 1.0 — never upscale. See config for the measurement."""
        if not p90 or p90 <= 0:
            return 1.0
        return float(min(OCR_SCALE_MAX, max(OCR_SCALE_MIN, target / p90)))


    def capture_verdict(p90):
        if not p90:
            return 'unknown', 'no text found'
        if p90 < REJECT_BELOW:
            return 'reject', f'glyph {p90:.0f}px (need >={MIN_BASE_GLYPH:.0f}) — much closer'
        if p90 < MIN_BASE_GLYPH:
            return 'warn', f'glyph {p90:.0f}px (want >={MIN_BASE_GLYPH:.0f}) — closer'
        return 'ok', ''


    def rescale_to_optimum(img):
        """Returns (resized, p90, scale). The single place rescaling happens."""
        p = glyph_p90(img)
        s = scale_for_target(p)
        if abs(s - 1.0) > 1e-3:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
        return img, p, s
    ''')

# ============================================================ core/textutils.py
f('core/textutils.py', '''
    """Text tidying shared by every OCR-consuming layer. Ported from Pipeline v9."""
    import difflib, re, unicodedata


    def norm(s: str) -> str:
        return re.sub(r'\\s+', ' ', unicodedata.normalize('NFC', s)).strip()


    def collapse_repeats(line, sim=0.80):
        w = line.split(); n = len(w)
        if n < 3:
            return line
        for p in range(1, n // 2 + 1):
            if n % p == 0:
                blocks = [' '.join(w[k*p:(k+1)*p]) for k in range(n // p)]
                if all(difflib.SequenceMatcher(None, blocks[0], b).ratio() >= sim
                       for b in blocks[1:]):
                    w = blocks[0].split(); n = len(w); break
        h = len(w) // 2
        if h > 0 and difflib.SequenceMatcher(
                None, ' '.join(w[:h]), ' '.join(w[h:2*h])).ratio() >= sim:
            w = w[:h] + w[2*h:]
        out = []
        for word in w:
            if out and word == out[-1]:
                continue
            out.append(word)
        return ' '.join(out)


    def _pass(words, sim, mx):
        i, out = 0, []
        while i < len(words):
            hit = False
            for L in range(min(mx, (len(words) - i) // 2), 1, -1):
                a, b = words[i:i+L], words[i+L:i+2*L]
                if difflib.SequenceMatcher(None, ' '.join(a), ' '.join(b)).ratio() >= sim:
                    out += a; i += 2*L; hit = True; break
            if not hit:
                out.append(words[i]); i += 1
        return out


    def strong_dedup(text, sim=0.78, mx=8):
        stream = ' '.join(l for l in text.split('\\n') if l.strip())
        prev = None
        while prev != stream:
            prev = stream
            stream = ' '.join(_pass(stream.split(), sim, mx))
        return collapse_repeats(stream)


    def vote_lines(texts):
        """Medoid per line across frames — multi-frame consensus."""
        seqs = [t.split('\\n') for t in texts if t.strip()]
        if not seqs:
            return ''
        out = []
        for k in range(max(len(s) for s in seqs)):
            cands = [s[k] for s in seqs if k < len(s) and s[k].strip()]
            if not cands:
                continue
            best, bs = cands[0], -1.0
            for c in cands:
                sc = sum(difflib.SequenceMatcher(None, c, d).ratio() for d in cands)
                if sc > bs:
                    bs, best = sc, c
            out.append(best)
        return '\\n'.join(out)
    ''')

# ============================================================ layers
f('layers/l2_select/select.py', '''
    """
    LAYER 2 — burst frame selection.
    OWNER: Ishara

    IN : list of image paths
    OUT: List[Frame], best first, weak frames dropped

    Two gates, not one:
      * sharpness  — rejects motion blur and focus hunt
      * glyph_p90  — rejects photos taken too far away, where the pilla fall
                     below the recoverable threshold

    The second gate is the one that matters. A perfectly sharp photo of a whole
    page from arm's length is still unreadable.
    """
    import cv2
    from core.schemas import Frame
    from core.imaging import sharpness, glyph_p90, capture_verdict
    from core.config import BURST_KEEP, MIN_SHARPNESS, SHARP_MIN_RATIO


    def select(paths, keep=BURST_KEEP):
        frames = []
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            s = sharpness(im)
            g = glyph_p90(im)
            v, note = capture_verdict(g)
            frames.append(Frame(path=str(p), width=w, height=h,
                                sharpness=s, glyph_p90=g, verdict=v, note=note))
        if not frames:
            return []

        frames.sort(key=lambda f: -(f.sharpness or 0))
        best = frames[0].sharpness or 1.0

        # Keep only frames of COMPARABLE quality. Voting helps when views fail
        # independently; a frame much worse than the rest drags the majority
        # toward its own errors (measured: mixed-quality views made consensus
        # 3.7% WORSE, comparable-quality views made it 7.9% better).
        out = [f for f in frames[:keep]
               if (f.sharpness or 0) >= best * SHARP_MIN_RATIO
               and (f.sharpness or 0) >= MIN_SHARPNESS]
        return out or frames[:1]
    ''')

f('layers/l3_segment/segment.py', '''
    """
    LAYER 3 — article segmentation and layout.
    OWNER: Ishara (engineering support for Component 1)

    IN : Frame (the reference frame)
    OUT: List[Article] with box + regions, in reading order

    YOLO finds article boxes; PP-DocLayout-L labels title/text regions inside
    them. If PaddleOCR is unavailable it falls back to a glyph-height heuristic,
    which is weaker but keeps the system running.
    """
    import cv2, numpy as np
    from core.schemas import Article, Box, Region
    from core.imaging import glyph_p90, capture_verdict
    from core.config import YOLO_CONF, YOLO_IMGSZ, MAX_ARTICLES

    from .geometry import (border_filter, deoverlap, page_reading_order,
                           assign_by_containment, order_columns)

    TITLE_LABELS = {'title', 'paragraph_title', 'doc_title',
                    'figure_title', 'chart_title'}
    IMAGE_LABELS = {'image', 'figure', 'table', 'chart'}
    DROP_LABELS  = {'header', 'footer', 'number', 'page_number'}


    class Segmenter:
        def __init__(self, yolo, layout=None):
            self.yolo = yolo
            self.layout = layout

        def run(self, img, conf=YOLO_CONF, max_articles=MAX_ARTICLES):
            H, W = img.shape[:2]
            r = self.yolo.predict(img, imgsz=YOLO_IMGSZ, conf=conf, verbose=False)[0]
            boxes = [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]
            boxes = border_filter(boxes, W, H)
            boxes = [boxes[i] for i in page_reading_order(boxes)]
            boxes = deoverlap(boxes)
            if not boxes:
                boxes = [[0, 0, W, H]]
            boxes = boxes[:max_articles]

            body_by_art, title_by_art = self._regions(img, boxes)

            arts = []
            for i, b in enumerate(boxes):
                crop = img[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])]
                p = glyph_p90(crop) if crop.size else None
                v, note = capture_verdict(p)
                regs = ([Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]),
                                label='title') for r_ in title_by_art.get(i, [])] +
                        [Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]),
                                label='text')
                         for r_ in order_columns(body_by_art.get(i, []))])
                arts.append(Article(index=i,
                                    box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                                    regions=regs, glyph_p90=p,
                                    verdict=v, note=note))
            return arts

        def _regions(self, img, boxes):
            if self.layout is None:
                return self._fallback(img, boxes)
            res = list(self.layout.predict(img, batch_size=1, layout_nms=True))[0]
            regs = [{'label': b['label'],
                     'box': [float(v) for v in b['coordinate']]}
                    for b in res['boxes']]
            body = [x['box'] for x in regs
                    if x['label'] not in TITLE_LABELS | IMAGE_LABELS | DROP_LABELS]
            title = [x['box'] for x in regs if x['label'] in TITLE_LABELS]
            return (assign_by_containment(boxes, body),
                    assign_by_containment(boxes, title))

        def _fallback(self, img, boxes):
            """Glyph-height heuristic when PP-DocLayout is unavailable."""
            body, title = {}, {}
            for i, a in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in a]
                crop = img[max(0, y1):y2, max(0, x1):x2]
                body[i], title[i] = [], []
                if crop.size == 0:
                    continue
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                bw = cv2.threshold(g, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                ker = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (max(15, crop.shape[1] // 12), 3))
                lines = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)
                cs, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                bs = [cv2.boundingRect(c) for c in cs]
                bs = [b for b in bs if b[3] > 4 and b[2] > crop.shape[1] * 0.2]
                if not bs:
                    continue
                base = np.percentile([b[3] for b in bs], 35)
                bs.sort(key=lambda b: b[1])
                for k, (bx, by, bw_, bh) in enumerate(bs):
                    box = [x1 + bx, y1 + by, x1 + bx + bw_, y1 + by + bh]
                    (title[i] if (bh >= base * 1.55 and k <= 1) else body[i]).append(box)
            return body, title
    ''')

f('layers/l4a_title/README.md', '''
    # Layer 4A — Title extraction

    **OWNER: other team member. Do not edit this folder.**

    ## Contract

    Input : `Article` with `regions` where `label == 'title'`, plus the frame image
    Output: set `article.title_raw` (OCR output) and `article.title` (final text)

    Implement `extract(img, article) -> article` in `title.py`. Touch nothing
    outside this folder.

    ## Notes for whoever builds this

    - Region boxes are in **full-frame coordinates**, not article-relative.
    - `core.imaging.rescale_to_optimum()` is available and is what the body path
      uses. Titles are usually larger than body text, so the optimum may differ —
      measure rather than assume.
    - If Medial Axis Transform is used, verify it against plain Tesseract at the
      optimal scale first. Sinhala dependent vowel signs (*pilla*) are only a few
      pixels wide and skeletonisation can destroy them.
    - Consider running the title through Component 2 correction as well. Titles
      matter more to a listener than body text — they decide whether to keep
      listening.

    ## Stub behaviour

    `title.py` currently returns the article unchanged so the rest of the system
    runs end to end while this layer is under development.
    ''')

f('layers/l4a_title/title.py', '''
    """
    LAYER 4A — title extraction.  OWNER: other team member.

    STUB — returns the article unchanged so the pipeline runs end to end.
    Replace the body of extract() only. Do not change the signature.
    """
    from core.schemas import Article


    def extract(img, article: Article) -> Article:
        # TODO (other member): MAT + Tesseract over article.regions
        #                      where label == 'title'
        return article
    ''')

f('layers/l4b_body/body.py', '''
    """
    LAYER 4B — body OCR and correction.  COMPONENT 2. OWNER: Ishara.

    IN : Article with text regions + the frame images
    OUT: article.body_raw (OCR) and article.body (after mT5)

    Two measured behaviours are enforced here and must not be silently changed:

      1. Every crop is rescaled to the optimal glyph size before OCR.
         Native-scale OCR measured CER 0.2205 against 0.1754 at 0.40x.
         Scaling is capped at 1.0 — upscaling made things dramatically worse.

      2. Generation uses no_repeat_ngram_size=6, which moved CER from
         0.0847 to 0.0515 on identical inputs by suppressing runaway.
    """
    import cv2, torch
    from core.schemas import Article
    from core.imaging import rescale_to_optimum
    from core.textutils import norm, strong_dedup, vote_lines
    from core.config import (TESS_CONFIG, TESS_LANG, MT5_NUM_BEAMS,
                             MT5_NO_REPEAT_NGRAM, MT5_MAX_LENGTH)


    class BodyReader:
        def __init__(self, tok, mdl, device, pytesseract):
            self.tok, self.mdl, self.dev = tok, mdl, device
            self.pt = pytesseract

        def ocr_region(self, img, box, pad=4):
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            if crop.size == 0:
                return '', None, None
            crop, p90, scale = rescale_to_optimum(crop)
            txt = self.pt.image_to_string(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                          lang=TESS_LANG, config=TESS_CONFIG)
            return norm(txt), p90, scale

        def read(self, frames_imgs, article: Article) -> Article:
            """frames_imgs: list of images (the selected frames). Multi-frame
            consensus is applied when more than one is supplied."""
            per_frame = []
            p90 = scale = None
            for img in frames_imgs:
                parts = []
                for r in article.regions:
                    if r.label != 'text':
                        continue
                    t, p, s = self.ocr_region(img, r.box)
                    if p90 is None:
                        p90, scale = p, s
                    if t:
                        parts.append(t)
                if parts:
                    per_frame.append('\\n'.join(parts))

            raw = vote_lines(per_frame) if len(per_frame) > 1 else \\
                  (per_frame[0] if per_frame else '')
            article.body_raw = strong_dedup(raw)
            article.glyph_p90 = article.glyph_p90 or p90
            article.ocr_scale = scale
            return article

        @torch.no_grad()
        def correct(self, article: Article) -> Article:
            if not article.body_raw.strip():
                return article
            out = []
            for line in [l for l in article.body_raw.split('\\n') if l.strip()]:
                e = self.tok(line, return_tensors='pt', truncation=True,
                             max_length=MT5_MAX_LENGTH).to(self.dev)
                g = self.mdl.generate(**e, max_length=MT5_MAX_LENGTH,
                                      num_beams=MT5_NUM_BEAMS,
                                      no_repeat_ngram_size=MT5_NO_REPEAT_NGRAM,
                                      early_stopping=True)
                out.append(self.tok.decode(g[0], skip_special_tokens=True))
            article.body = strong_dedup('\\n'.join(out))
            return article
    ''')

f('layers/l5_assemble/assemble.py', '''
    """
    LAYER 5 — document assembly.  OWNER: shared.

    Collects finished Articles into a Document in reading order, drops rejected
    ones, and records warnings the phone app can speak.
    """
    from core.schemas import Article, Document


    def assemble(articles, source_frames, timings=None) -> Document:
        warnings = []
        keep = []
        for a in sorted(articles, key=lambda x: x.index):
            if a.verdict == 'reject':
                warnings.append(f'Article {a.index + 1} skipped: {a.note}')
                continue
            if a.verdict == 'warn' and a.note:
                warnings.append(f'Article {a.index + 1}: {a.note}')
            if not (a.title.strip() or a.body.strip()):
                continue
            keep.append(a)
        for i, a in enumerate(keep):
            a.index = i
        return Document(source_frames=source_frames, articles=keep,
                        timings=timings or {}, warnings=warnings)
    ''')

f('layers/l6_speech/README.md', '''
    # Layer 6 — RAG and TTS

    **OWNER: Bumal (Components 3 and 4). Do not edit this folder.**

    ## Contract

    Input : `Document` (see `core/schemas.py`)
    Output: audio file path, or a URL the phone can fetch

    Implement `speak(document) -> str` in `speech.py`.

    ## Latency note

    Return the **title audio first** if you can. A Sinhala headline takes three
    to four seconds to speak, which covers the body correction time, so the user
    perceives almost no wait. This is why the pipeline keeps title and body
    separate.
    ''')

f('layers/l6_speech/speech.py', '''
    """
    LAYER 6 — RAG + TTS.  OWNER: Bumal.

    STUB — returns None so the pipeline runs end to end without audio.
    Replace the body of speak() only.
    """
    from core.schemas import Document
    from typing import Optional


    def speak(document: Document) -> Optional[str]:
        # TODO (Bumal): RAG over document.articles, then Sinhala TTS.
        return None
    ''')

# ============================================================ orchestrator
f('app/pipeline.py', '''
    """
    The orchestrator. Wires the layers together and owns model loading.

    Nothing here implements algorithm logic — that belongs in the layers.
    """
    import time, cv2
    from pathlib import Path

    from core.config import (YOLO_WEIGHTS, MT5_PLAIN, MAX_ARTICLES, YOLO_CONF)
    from core.schemas import Document
    from layers.l2_select.select import select
    from layers.l3_segment.segment import Segmenter
    from layers.l4a_title import title as l4a
    from layers.l4b_body.body import BodyReader
    from layers.l5_assemble.assemble import assemble
    from layers.l6_speech import speech as l6


    class Pipeline:
        def __init__(self, use_layout=True, verbose=True):
            import torch, pytesseract
            from ultralytics import YOLO
            from transformers import MT5ForConditionalGeneration, AutoTokenizer

            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'

            w = next((p for p in YOLO_WEIGHTS if p.exists()), None)
            if w is None:
                raise SystemExit('YOLO weights not found — check core/config.py')
            yolo = YOLO(str(w))

            layout = None
            if use_layout:
                try:
                    from paddleocr import LayoutDetection
                    layout = LayoutDetection(model_name='PP-DocLayout-L',
                                             device='cpu', enable_mkldnn=False)
                except Exception as e:
                    if verbose:
                        print('PP-DocLayout unavailable, using fallback:', e)

            self.seg = Segmenter(yolo, layout)

            tok = AutoTokenizer.from_pretrained(str(MT5_PLAIN))
            mdl = MT5ForConditionalGeneration.from_pretrained(
                str(MT5_PLAIN)).to(self.dev).eval()
            self.body = BodyReader(tok, mdl, self.dev, pytesseract)

            if verbose:
                print(f'pipeline ready on {self.dev} '
                      f'(layout={"on" if layout else "fallback"})')

        def run(self, image_paths, conf=YOLO_CONF,
                max_articles=MAX_ARTICLES, correct=True) -> Document:
            t = {}

            t0 = time.time()
            frames = select(image_paths)
            t['select'] = round(time.time() - t0, 2)
            if not frames:
                return Document(warnings=['no usable frames'])

            imgs = [cv2.imread(f.path) for f in frames]
            imgs = [i for i in imgs if i is not None]
            ref = imgs[0]

            t0 = time.time()
            arts = self.seg.run(ref, conf=conf, max_articles=max_articles)
            t['segment'] = round(time.time() - t0, 2)

            t0 = time.time()
            arts = [l4a.extract(ref, a) for a in arts]
            t['title'] = round(time.time() - t0, 2)

            t0 = time.time()
            arts = [self.body.read(imgs, a) for a in arts]
            t['ocr'] = round(time.time() - t0, 2)

            if correct:
                t0 = time.time()
                arts = [self.body.correct(a) for a in arts]
                t['correct'] = round(time.time() - t0, 2)
            else:
                for a in arts:
                    a.body = a.body_raw

            doc = assemble(arts, [f.path for f in frames], t)
            doc.timings['total'] = round(sum(t.values()), 2)
            return doc
    ''')

f('app/server.py', '''
    """
    FastAPI server: phone endpoint + debug UI.

        python -m app.server --root "D:/Sinhala_OCR_Correction_v2"
    """
    import argparse, os, shutil, uuid
    from pathlib import Path

    import numpy as np, cv2
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from core.config import WORK_DIR, HOST, PORT
    from core.schemas import CaptureResponse
    from layers.l6_speech import speech as l6


    def build(pipeline, web_dir: Path):
        app = FastAPI(title='Sinhala Reader')
        app.add_middleware(CORSMiddleware, allow_origins=['*'],
                           allow_methods=['*'], allow_headers=['*'])
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        app.mount('/work', StaticFiles(directory=str(WORK_DIR)), name='work')

        @app.get('/', response_class=HTMLResponse)
        def index():
            p = web_dir / 'reader.html'
            return p.read_text(encoding='utf-8') if p.exists() else '<h1>reader.html missing</h1>'

        @app.get('/debug', response_class=HTMLResponse)
        def debug():
            p = web_dir / 'debug.html'
            return p.read_text(encoding='utf-8') if p.exists() else '<h1>debug.html missing</h1>'

        @app.get('/health')
        def health():
            return {'ok': True, 'device': pipeline.dev}

        @app.post('/capture')
        async def capture(frames: list[UploadFile] = File(...)):
            sess = WORK_DIR / uuid.uuid4().hex[:8]
            sess.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, f in enumerate(frames):
                data = await f.read()
                arr = np.frombuffer(data, np.uint8)
                im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if im is None:
                    continue
                p = sess / f'f{i}.jpg'
                cv2.imwrite(str(p), im)
                paths.append(str(p))
            if not paths:
                return JSONResponse(
                    {'document': None, 'error': 'no decodable frames'},
                    status_code=400)
            try:
                doc = pipeline.run(paths)
                audio = l6.speak(doc)
                return CaptureResponse(document=doc, audio_url=audio).model_dump()
            except Exception as e:
                import traceback; traceback.print_exc()
                return JSONResponse({'error': str(e)}, status_code=500)

        return app


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument('--root', default=None)
        ap.add_argument('--host', default=HOST)
        ap.add_argument('--port', type=int, default=PORT)
        ap.add_argument('--cert', default=None)
        ap.add_argument('--key', default=None)
        ap.add_argument('--no-layout', action='store_true')
        a = ap.parse_args()
        if a.root:
            os.environ['SINHALA_ROOT'] = a.root

        from app.pipeline import Pipeline
        pipe = Pipeline(use_layout=not a.no_layout)
        app = build(pipe, Path(__file__).resolve().parent.parent / 'web')

        import uvicorn
        kw = {'ssl_certfile': a.cert, 'ssl_keyfile': a.key} if a.cert and a.key else {}
        uvicorn.run(app, host=a.host, port=a.port, **kw)


    if __name__ == '__main__':
        main()
    ''')

# ============================================================ tests
f('tests/test_contracts.py', '''
    """Contract tests. If these fail, layers will not fit together."""
    from core.schemas import Box, Frame, Region, Article, Document


    def test_box_dims():
        b = Box(x1=10, y1=20, x2=110, y2=220)
        assert b.w == 100 and b.h == 200


    def test_article_fields_independent():
        """Layer 4A and 4B must write DIFFERENT fields — neither may clobber
        the other. This is the whole reason the split exists."""
        a = Article(index=0, box=Box(x1=0, y1=0, x2=10, y2=10))
        a.title = 'T'
        a.body = 'B'
        assert a.title == 'T' and a.body == 'B'


    def test_document_roundtrip():
        d = Document(articles=[Article(index=0,
                                       box=Box(x1=0, y1=0, x2=1, y2=1))])
        assert Document(**d.model_dump()).articles[0].index == 0


    def test_assemble_drops_rejected():
        from layers.l5_assemble.assemble import assemble
        good = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1),
                       body='text', verdict='ok')
        bad = Article(index=1, box=Box(x1=0, y1=0, x2=1, y2=1),
                      verdict='reject', note='too far')
        doc = assemble([good, bad], ['f.jpg'])
        assert len(doc.articles) == 1
        assert any('too far' in w for w in doc.warnings)


    def test_stubs_do_not_break_flow():
        from layers.l4a_title import title as l4a
        from layers.l6_speech import speech as l6
        a = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1))
        assert l4a.extract(None, a) is a
        assert l6.speak(Document()) is None
    ''')

f('tests/test_imaging.py', '''
    """The measured constants must stay enforced. If someone changes these,
    the numbers in the report stop matching the system."""
    import numpy as np, cv2
    from core.imaging import glyph_p90, scale_for_target, capture_verdict


    def _page(glyph_h, W=1200):
        H = int(W * .75)
        img = np.full((H, W), 245, np.uint8)
        y = glyph_h
        while y < H - glyph_h * 2:
            x = glyph_h
            while x < W - glyph_h * 2:
                img[y:y + glyph_h, x:x + max(3, glyph_h // 2)] = 30
                x += max(3, glyph_h // 2) + max(2, glyph_h // 4)
            y += int(glyph_h * 2)
        return img


    def test_glyph_measurement():
        for h in (12, 24, 48):
            assert abs(glyph_p90(_page(h)) - h) <= 3


    def test_scale_never_upscales():
        """Upscaling measured CER 0.336 at 2x and 0.659 at 3x, against 0.175 at
        the optimum. The cap is not a preference."""
        assert scale_for_target(19) == 1.0
        assert scale_for_target(10) == 1.0
        assert abs(scale_for_target(60) - 0.40) < .01


    def test_verdict_bands():
        assert capture_verdict(60)[0] == 'ok'
        assert capture_verdict(19)[0] == 'warn'
        assert capture_verdict(15)[0] == 'reject'
        assert capture_verdict(None)[0] == 'unknown'
    ''')

f('tests/test_text.py', '''
    from core.textutils import strong_dedup, collapse_repeats, vote_lines, norm


    def test_dedup():
        assert strong_dedup('abc def abc def ghi') == 'abc def ghi'
        assert collapse_repeats('aa bb aa bb') == 'aa bb'


    def test_vote_picks_majority():
        assert vote_lines(['x y', 'x y', 'z w']) == 'x y'
        assert vote_lines([]) == ''


    def test_norm_nfc():
        import unicodedata
        s = unicodedata.normalize('NFD', '\\u0dda')
        assert norm(s) == '\\u0dda'
    ''')

# ============================================================ misc
f('requirements.txt', '''
    fastapi
    uvicorn[standard]
    python-multipart
    pydantic>=2
    numpy
    opencv-python
    pillow
    torch
    transformers
    sentencepiece
    ultralytics
    pytesseract
    pytest
    # optional, better title/text labelling:
    # paddlepaddle==3.2.0
    # paddleocr>=3.1.0
    ''')

f('.gitignore', '''
    __pycache__/
    *.pyc
    work/
    *.pem
    .venv/
    venv/
    # models live outside the repo - they are large and belong on Drive
    models/
    *.pt
    *.safetensors
    ''')

f('README.md', '''
    # Sinhala Reader — integrated system

    Phone captures a newspaper page, the laptop reads it, audio comes back.

    ## Layers and ownership

    | Layer | Does | Owner |
    |---|---|---|
    | L1 phone app | capture, guidance, upload | Ishara |
    | L2 select | pick usable frames (sharpness + glyph height) | Ishara |
    | L3 segment | YOLO articles + layout regions | Ishara |
    | **L4A title** | title OCR | **other member** |
    | **L4B body** | body OCR + mT5 correction (Component 2) | **Ishara** |
    | L5 assemble | order, drop rejects, collect warnings | shared |
    | **L6 speech** | RAG + Sinhala TTS | **Bumal** |

    ## The rule that keeps this working

    **`core/schemas.py` is the contract.** L4A writes `title`, L4B writes `body`,
    and neither touches the other's fields. Anyone can develop and test their
    layer alone against the schema — no waiting.

    Change `core/schemas.py` only after telling the team.

    ## Run

        pip install -r requirements.txt
        pytest -q                                    # contracts + measured constants
        python -m app.server --root "D:/Sinhala_OCR_Correction_v2"

    Phone camera needs HTTPS:

        openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \\
                -days 90 -subj "/CN=localhost"
        python -m app.server --root "..." --cert cert.pem --key key.pem

    Then `https://<laptop-ip>:8000/` on the phone, `/debug` on the laptop.

    ## Models

    Models are **not** in this repo. Set `SINHALA_ROOT` (or pass `--root`) to the
    project folder containing:

        models/mt5_plain/               config.json + model.safetensors + tokenizer
        layout/.../best.pt              YOLO article detector

    ## Constants you must not casually change

    `core/config.py` holds measured values, not preferences:

    - `TARGET_GLYPH = 24`, `MIN_BASE_GLYPH = 22` — below this the Sinhala vowel
      signs fall under ~11 px and become unrecoverable
    - `OCR_SCALE_MAX = 1.0` — upscaling measured CER 0.336 at 2x and 0.659 at 3x
      against 0.175 at the optimum
    - `MT5_NO_REPEAT_NGRAM = 6` — moved CER from 0.0847 to 0.0515

    `tests/test_imaging.py` enforces these. If a test fails, the system has
    drifted from the reported research.

    ## Cloud later

    Nothing here assumes localhost. Moving to cloud means changing the URL the
    phone posts to and running the same `app.server` on a GPU instance. Do it
    only after the local version works end to end.
    ''')

f('web/debug.html', '''
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Pipeline debug</title>
    <style>
    body{font:14px/1.5 system-ui;margin:0;background:#0f1113;color:#e6edf3}
    header{padding:12px 16px;border-bottom:1px solid #21262d}
    .pad{padding:16px}
    .art{border:1px solid #21262d;border-radius:10px;padding:12px;margin-top:12px}
    .k{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
    pre{white-space:pre-wrap;background:#0d1117;padding:10px;border-radius:8px}
    .warn{color:#d29922}.bad{color:#da3633}
    table{border-collapse:collapse;margin-top:8px}
    td,th{border:1px solid #21262d;padding:5px 9px;text-align:left}
    </style></head><body>
    <header><b>Pipeline debug</b> — upload frames, inspect every stage</header>
    <div class="pad">
      <input type="file" id="f" multiple accept="image/*">
      <button onclick="go()">Run</button>
      <div id="t"></div><div id="o"></div>
    </div>
    <script>
    async function go(){
      const fs=document.getElementById('f').files;
      if(!fs.length){alert('choose images');return}
      const fd=new FormData();
      for(const x of fs) fd.append('frames',x);
      document.getElementById('o').innerHTML='running…';
      const r=await fetch('/capture',{method:'POST',body:fd});
      const j=await r.json();
      const d=j.document||{};
      let t='<table><tr><th>stage</th><th>seconds</th></tr>';
      for(const k in (d.timings||{})) t+=`<tr><td>${k}</td><td>${d.timings[k]}</td></tr>`;
      document.getElementById('t').innerHTML=t+'</table>';
      let h='';
      (d.warnings||[]).forEach(w=>h+=`<p class="warn">⚠ ${w}</p>`);
      (d.articles||[]).forEach((a,i)=>{
        h+=`<div class="art"><div class="k">article ${i+1} ·
            glyph ${a.glyph_p90?a.glyph_p90.toFixed(0)+'px':'?'} ·
            scale ${a.ocr_scale?a.ocr_scale.toFixed(2)+'×':'?'} ·
            ${a.verdict}</div>
            <div class="k">title</div><pre>${a.title||'—'}</pre>
            <div class="k">body — raw OCR</div><pre>${a.body_raw||'—'}</pre>
            <div class="k">body — after mT5</div><pre>${a.body||'—'}</pre></div>`;
      });
      document.getElementById('o').innerHTML=h||'<p class="bad">no articles</p>';
    }
    </script></body></html>
    ''')

f('layers/l3_segment/geometry.py', '''
    """Geometry helpers for Layer 3 (from Pipeline v9)."""
    import numpy as np
    
    EDGE, FRAC = 8, 0.45
    
    def inter(a,b):
        ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
        return ix*iy
    
    def iou(a,b):
        i=inter(a,b); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
        return i/u if u>0 else 0.0
    
    def ctr(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
    
    def border_filter(bx,W,H):
        if not bx: return []
        med=np.median([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in bx]); keep=[]
        for b in bx:
            x1,y1,x2,y2=b; a=(x2-x1)*(y2-y1)
            if (x1<=EDGE or y1<=EDGE or x2>=W-EDGE or y2>=H-EDGE) and a<FRAC*med: continue
            keep.append(b)
        return keep
    
    def deoverlap(A):
        A=[list(b) for b in A]
        for i in range(len(A)):
            for j in range(len(A)):
                if i==j or inter(A[i],A[j])<=0: continue
                a,b=A[i],A[j]
                if a[0]<b[0]:
                    oxw=min(a[2],b[2])-max(a[0],b[0]); oyh=min(a[3],b[3])-max(a[1],b[1])
                    if oxw<oyh:
                        m=(a[2]+b[0])/2; a[2]=min(a[2],m); b[0]=max(b[0],m)
        return A
    
    def page_reading_order(bx):
        if not bx: return []
        idx=sorted(range(len(bx)),key=lambda i:bx[i][1])
        band=0.12*max(b[3]-b[1] for b in bx)+0.03*max(b[3] for b in bx); rows=[]
        for i in idx:
            y=bx[i][1]
            if rows and y-rows[-1][0]<=band: rows[-1][1].append(i)
            else: rows.append([y,[i]])
        o=[]
        for _,r in rows: o+=sorted(r,key=lambda i:bx[i][0])
        return o
    
    def assign_by_containment(articles,regs,thr=0.15):
        out={i:[] for i in range(len(articles))}
        for r in regs:
            ra=(r[2]-r[0])*(r[3]-r[1]); best=-1; bs=0.0
            for i,art in enumerate(articles):
                s=inter(art,r)/ra if ra>0 else 0
                if s>bs: bs,best=s,i
            if best>=0 and bs>thr: out[best].append(r)
            else:
                cx,cy=ctr(r); d=1e18; bi=0
                for i,art in enumerate(articles):
                    ax,ay=ctr(art); dd=(cx-ax)**2+(cy-ay)**2
                    if dd<d: d,bi=dd,i
                out[bi].append(r)
        return out
    
    def merge_overlapping(regs,thr=0.30):
        regs=[list(r) for r in regs]; changed=True
        while changed:
            changed=False; out=[]; used=[False]*len(regs)
            for i in range(len(regs)):
                if used[i]: continue
                cur=regs[i][:]
                for j in range(i+1,len(regs)):
                    if used[j]: continue
                    if iou(cur,regs[j])>thr:
                        cur=[min(cur[0],regs[j][0]),min(cur[1],regs[j][1]),
                             max(cur[2],regs[j][2]),max(cur[3],regs[j][3])]
                        used[j]=True; changed=True
                out.append(cur); used[i]=True
            regs=out
        return regs
    
    def order_columns(regs,wtol=0.5):
        if not regs: return []
        regs=sorted(merge_overlapping(regs),key=lambda r:r[0]); cols=[]
        for r in regs:
            placed=False
            for col in cols:
                lo=min(c[0] for c in col); hi=max(c[2] for c in col)
                if min(r[2],hi)-max(r[0],lo) > wtol*min(r[2]-r[0],hi-lo):
                    col.append(r); placed=True; break
            if not placed: cols.append([r])
        cols.sort(key=lambda c:min(x[0] for x in c)); out=[]
        for col in cols: col.sort(key=lambda r:r[1]); out+=col
        return out
    
    ''')

PKGS = ['core', 'layers', 'layers/l2_select', 'layers/l3_segment',
        'layers/l4a_title', 'layers/l4b_body', 'layers/l5_assemble',
        'layers/l6_speech', 'app', 'tests']



# names we must never write into or delete
PROTECTED = {'.git', '.github', 'Work', 'work_research', 'node_modules',
             '.venv', 'venv', '__pycache__'}


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--into', default='system',
                    help="where to create the system (default: system)")
    ap.add_argument('--dry-run', action='store_true',
                    help='show what would happen, write nothing')
    ap.add_argument('--force', action='store_true',
                    help='overwrite scaffold files that already exist')
    a = ap.parse_args()

    repo = Path.cwd().resolve()
    root = (repo / a.into).resolve() if a.into != '.' else repo

    # refuse to write outside the repo
    if repo not in root.parents and root != repo:
        sys.exit(f'refusing: {root} is outside {repo}')

    # refuse to write into a protected folder
    try:
        rel_parts = root.relative_to(repo).parts
    except ValueError:
        rel_parts = ()
    if rel_parts and rel_parts[0] in PROTECTED and rel_parts[0] != 'Work':
        sys.exit(f"refusing: '{rel_parts[0]}' is protected")

    print(f'repo   : {repo}')
    print(f'target : {root}')
    if a.dry_run:
        print('mode   : DRY RUN — nothing will be written')
    elif a.force:
        print('mode   : FORCE — existing scaffold files WILL be overwritten')
    else:
        print('mode   : safe — existing files are left alone')

    # ---- what is already here ----
    existing = sorted(p.name for p in repo.iterdir()) if repo.exists() else []
    if existing:
        print('\nalready in the repo (untouched):')
        for n in existing[:12]:
            mark = '  [protected]' if n in PROTECTED else ''
            print(f'   {n}{mark}')
        if len(existing) > 12:
            print(f'   ... and {len(existing)-12} more')

    created, skipped, made_dirs = [], [], []

    def mkdir(p: Path):
        if p.exists():
            return
        made_dirs.append(p)
        if not a.dry_run:
            p.mkdir(parents=True, exist_ok=True)

    for pk in PKGS:
        mkdir(root / pk)
        init = root / pk / '__init__.py'
        if init.exists() and not a.force:
            skipped.append(init)
        else:
            created.append(init)
            if not a.dry_run:
                init.write_text('', encoding='utf-8')

    for d in ('web', 'work', 'android', 'docs', 'models_here'):
        mkdir(root / d)

    for rel, content in FILES.items():
        p = root / rel
        if p.exists() and not a.force:
            skipped.append(p)
            continue
        created.append(p)
        if not a.dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')

    # a note so nobody wonders where the weights go
    note = root / 'models_here' / 'PUT_MODELS_HERE.md'
    if not note.exists() or a.force:
        created.append(note)
        if not a.dry_run:
            note.write_text(textwrap.dedent("""
                # Model files go here (or anywhere — pass the path with --root)

                Download from Drive, ~1.26 GB total:

                    models/mt5_plain/
                        config.json
                        generation_config.json
                        model.safetensors        ~1.2 GB  <- check this size
                        tokenizer.json           ~16 MB
                        tokenizer_config.json

                    layout/runs/articles_full/weights/
                        best.pt                  ~40.5 MB

                Do NOT use Drive Desktop sync for model.safetensors — it often
                leaves a small placeholder instead of the real weights.

                Then run:

                    python -m app.server --root "<this folder>"
                """).lstrip(), encoding='utf-8')

    # ---- report ----
    print(f'\nfolders created : {len(made_dirs)}')
    print(f'files created   : {len(created)}')
    print(f'files skipped   : {len(skipped)} (already existed)')
    if skipped:
        for p in skipped[:8]:
            print(f'   kept: {p.relative_to(repo)}')
        if len(skipped) > 8:
            print(f'   ... and {len(skipped)-8} more')
        if not a.force:
            print('   (use --force to overwrite these)')

    if a.dry_run:
        print('\nDRY RUN — nothing written. Re-run without --dry-run to create.')
        return

    rel = root.relative_to(repo) if root != repo else Path('.')
    print(f"""
next:
  cd {rel}
  python -m venv venv && venv\\Scripts\\activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  pytest -q                       <- must print 11 passed
  copy ..\\reader.html web\\        <- if reader.html is at the repo root
""")


if __name__ == '__main__':
    main()

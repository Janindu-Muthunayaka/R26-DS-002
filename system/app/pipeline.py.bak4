"""
The orchestrator. Wires the layers together and owns model loading.

Nothing here implements algorithm logic — that belongs in the layers.
"""
import time, cv2
from pathlib import Path

from core.config import (YOLO_WEIGHTS, MT5_PLAIN, MAX_ARTICLES, YOLO_CONF)
from core.imaging import imread_upright
from layers.l3_segment.closeup import analyse as closeup_analyse
from core.schemas import Document, Article, Box, Region
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

        # imread_upright, not cv2.imread — EXIF orientation again
        imgs = [imread_upright(f.path) for f in frames]
        imgs = [i for i in imgs if i is not None]
        ref = imgs[0]

        # CLOSE-UP BRANCH. The article detector was trained on full and half
        # pages; the capture app shoots far closer than that and the frame is
        # one article. Running YOLO there does not fail safely — it returned a
        # confident box over the NEIGHBOURING article. See l3_segment/closeup.
        t0 = time.time()
        info = closeup_analyse(ref)
        if info['is_closeup']:
            x1, y1, x2, y2 = info['bbox']
            box = Box(x1=x1, y1=y1, x2=x2, y2=y2)
            regions = [Region(box=box, label='text')]
            # The headline is LOCATED but not read: title OCR is Layer 4A and
            # belongs to another team member. Emitting the region means their
            # layer works the moment it is delivered, and means the headline is
            # visibly pending rather than silently dropped.
            if info.get('title_bbox'):
                tx1, ty1, tx2, ty2 = info['title_bbox']
                regions.insert(0, Region(
                    box=Box(x1=tx1, y1=ty1, x2=tx2, y2=ty2), label='title'))
            note = (f"close-up: {info['n_lines']} text lines, "
                    f"crop {info['bbox_frac']:.0%} of frame")
            if info.get('n_title_lines'):
                note += (f"; {info['n_title_lines']} headline lines located "
                         f"(title OCR is Layer 4A, not yet delivered)")
            arts = [Article(index=0, box=box, regions=regions,
                            glyph_p75=info['glyph_p75'], verdict='ok',
                            note=note)]
            t['segment'] = round(time.time() - t0, 2)
            t['mode'] = 'closeup'
        else:
            arts = self.seg.run(ref, conf=conf, max_articles=max_articles)
            t['segment'] = round(time.time() - t0, 2)

        t0 = time.time()
        arts = [l4a.extract(ref, a) for a in arts]
        t['title'] = round(time.time() - t0, 2)

        t0 = time.time()
        if info['is_closeup']:
            arts = [self.body.read_page(imgs, a) for a in arts]
        else:
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
        doc.timings['total'] = round(sum(v for v in t.values()
                                 if isinstance(v, (int, float))), 2)
        return doc

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
from core.imaging import glyph_p75, glyph_p90, capture_verdict
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
            # capture_verdict takes p75, NOT p90. Until 20 Aug 2026 this passed
            # a p90 value, so article verdicts were measured on one scale and
            # judged on another — and l5_assemble DROPS articles on this
            # verdict, so it decided what got read aloud.
            #
            # Caveat worth knowing: CAPTURE_MIN_GLYPH_P75 was calibrated on
            # whole pages. An article crop excludes headlines and photos, so
            # its component mix differs. Using the same threshold here is a
            # conservative approximation, not a measured one.
            p75 = glyph_p75(crop) if crop.size else None
            p90 = glyph_p90(crop) if crop.size else None
            v, note = capture_verdict(p75)
            regs = ([Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]),
                            label='title') for r_ in title_by_art.get(i, [])] +
                    [Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]),
                            label='text')
                     for r_ in order_columns(body_by_art.get(i, []))])
            arts.append(Article(index=i,
                                box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                                regions=regs, glyph_p75=p75, glyph_p90=p90,
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

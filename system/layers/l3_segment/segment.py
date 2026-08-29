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
                'figure_title', 'chart_title', 'document_title'}
IMAGE_LABELS = {'image', 'figure', 'table', 'chart'}
DROP_LABELS  = {'header', 'footer', 'number', 'page_number'}


class Segmenter:
    def __init__(self, yolo, layout=None):
        self.yolo = yolo
        self.layout = layout

    def run(self, img, conf=YOLO_CONF, max_articles=MAX_ARTICLES):
        H, W = img.shape[:2]
        if self.layout is None:
            # Fallback (no layout model): treat whole page as one article
            return self._create_single_article(img, [[0, 0, W, H]])

        # 1. First Pass: Detect all layout blocks on full image
        res = list(self.layout.predict(img, batch_size=1, layout_nms=True))[0]
        regs = [{'label': b['label'], 'box': [float(v) for v in b['coordinate']]} for b in res['boxes']]
        
        # Filter valid regions (text and titles)
        valid_regs = [r for r in regs if r['label'] in TITLE_LABELS | {'text', 'image_caption', 'aside_text'}]
        if not valid_regs:
            return self._create_single_article(img, [[0, 0, W, H]])

        # 2. Group into horizontal columns first
        # Sort left-to-right to cluster columns
        valid_regs.sort(key=lambda r: r['box'][0])
        columns = []
        for r in valid_regs:
            placed = False
            rx1, ry1, rx2, ry2 = r['box']
            r_w = rx2 - rx1
            for col in columns:
                col_x1 = min(b['box'][0] for b in col)
                col_x2 = max(b['box'][2] for b in col)
                col_w = col_x2 - col_x1
                overlap = min(rx2, col_x2) - max(rx1, col_x1)
                # Overlap threshold: 30% of the minimum width
                if overlap > 0.3 * min(r_w, col_w):
                    col.append(r)
                    placed = True
                    break
            if not placed:
                columns.append([r])

        # 3. Within each column, sort top-to-bottom and group into Article blocks
        grouped_articles = []
        for col in columns:
            col.sort(key=lambda r: r['box'][1])
            current_article = []
            for r in col:
                if r['label'] in TITLE_LABELS:
                    if current_article:
                        grouped_articles.append(current_article)
                    current_article = [r]
                else:
                    current_article.append(r)
            if current_article:
                grouped_articles.append(current_article)

        # Sort grouped articles left-to-right by their horizontal center coordinates
        grouped_articles.sort(key=lambda g: sum(r['box'][0] + r['box'][2] for r in g) / (2 * len(g)))
        
        # Deduplicate overlapping articles (NMS based on box overlap)
        deduped = []
        for g in grouped_articles:
            gx1 = min(r['box'][0] for r in g)
            gy1 = min(r['box'][1] for r in g)
            gx2 = max(r['box'][2] for r in g)
            gy2 = max(r['box'][3] for r in g)
            area = (gx2 - gx1) * (gy2 - gy1)
            
            keep = True
            for existing in list(deduped):
                ex1, ey1, ex2, ey2 = existing['bbox']
                ix1 = max(gx1, ex1)
                iy1 = max(gy1, ey1)
                ix2 = min(gx2, ex2)
                iy2 = min(gy2, ey2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                
                min_area = min(area, existing['area'])
                if min_area > 0:
                    overlap_ratio = inter / min_area
                    if overlap_ratio > 0.70:
                        # Keep the one with more regions
                        if len(g) <= len(existing['group']):
                            keep = False
                            break
                        else:
                            deduped.remove(existing)
            if keep:
                deduped.append({'group': g, 'bbox': (gx1, gy1, gx2, gy2), 'area': area})
                
        grouped_articles = [d['group'] for d in deduped]
        grouped_articles = grouped_articles[:max_articles]
        
        arts = []
        for i, group in enumerate(grouped_articles):
            # Compute article bounding box
            x1 = min(r['box'][0] for r in group)
            y1 = min(r['box'][1] for r in group)
            x2 = max(r['box'][2] for r in group)
            y2 = max(r['box'][3] for r in group)
            
            # Padding
            PAD = 15
            x1, y1 = max(0, int(x1) - PAD), max(0, int(y1) - PAD)
            x2, y2 = min(W, int(x2) + PAD), min(H, int(y2) + PAD)
            
            crop = img[y1:y2, x1:x2]
            p75 = glyph_p75(crop) if crop.size else None
            p90 = glyph_p90(crop) if crop.size else None
            v, note = capture_verdict(p75)

            final_regs = []
            if crop.size > 0:
                # 3. Second Pass: Re-run PaddleOCR on the crop
                crop_res = list(self.layout.predict(crop, batch_size=1, layout_nms=True))[0]
                crop_regs = [{'label': b['label'], 'box': [float(v) for v in b['coordinate']]} for b in crop_res['boxes']]
                
                # Map coordinates back to full image
                for cr in crop_regs:
                    if cr['label'] in TITLE_LABELS | {'text'}:
                        cx1, cy1, cx2, cy2 = cr['box']
                        mapped_box = Box(x1=cx1 + x1, y1=cy1 + y1, x2=cx2 + x1, y2=cy2 + y1)
                        label = 'title' if cr['label'] in TITLE_LABELS else 'text'
                        final_regs.append(Region(box=mapped_box, label=label))
                
                # Sort final regions
                if final_regs:
                    title_regs = [r for r in final_regs if r.label == 'title']
                    text_regs = order_columns([ [r.box.x1, r.box.y1, r.box.x2, r.box.y2] for r in final_regs if r.label == 'text' ])
                    final_regs = title_regs + [Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]), label='text') for r_ in text_regs]

            # If second pass failed to find anything, fallback to first pass regions
            if not final_regs:
                for gr in group:
                    mapped_box = Box(x1=gr['box'][0], y1=gr['box'][1], x2=gr['box'][2], y2=gr['box'][3])
                    label = 'title' if gr['label'] in TITLE_LABELS else 'text'
                    final_regs.append(Region(box=mapped_box, label=label))

            arts.append(Article(index=i, box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                                regions=final_regs, glyph_p75=p75, glyph_p90=p90,
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

    def _create_single_article(self, img, boxes):
        H, W = img.shape[:2]
        body_by_art, title_by_art = self._regions(img, boxes)
        arts = []
        for i, b in enumerate(boxes):
            crop = img[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])]
            p75 = glyph_p75(crop) if crop.size else None
            p90 = glyph_p90(crop) if crop.size else None
            v, note = capture_verdict(p75)
            regs = ([Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]), label='title') for r_ in title_by_art.get(i, [])] +
                    [Region(box=Box(x1=r_[0], y1=r_[1], x2=r_[2], y2=r_[3]), label='text') for r_ in order_columns(body_by_art.get(i, []))])
            arts.append(Article(index=i, box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                                regions=regs, glyph_p75=p75, glyph_p90=p90, verdict=v, note=note))
        return arts

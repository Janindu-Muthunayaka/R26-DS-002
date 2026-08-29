"""
The orchestrator. Wires the layers together and owns model loading.

Nothing here implements algorithm logic — that belongs in the layers.
"""
import time, cv2
from pathlib import Path

from core.config import (YOLO_WEIGHTS, MT5_PLAIN, MAX_ARTICLES, YOLO_CONF,
                         POLISH_MODE, LAYOUT_MIN_P75, SEGMENT_MODE)
from core.imaging import imread_upright
from layers.l3_segment.closeup import analyse as closeup_analyse
from layers.l3_segment.closeup import headline_for_block as _headline_for_block
from layers.l3_segment.closeup import text_lines as closeup_text_lines
from layers.l3_segment import layout as _layout
from core.schemas import Document, Article, Box, Region
from layers.l2_select.select import select
from layers.l3_segment.segment import Segmenter
from layers.l4a_title import title as l4a
from layers.l4b_body.body import BodyReader
from layers.l4c_polish import polish as _polish
from layers.l5_assemble.assemble import assemble
from layers.l8b_speech import speech as l6


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
                # Determine device for PaddleOCR
                dev_str = 'gpu:0' if torch.cuda.is_available() else 'cpu'
                layout = LayoutDetection(model_name='PP-DocLayout_plus-L',
                                         threshold=0.20,
                                         device=dev_str, enable_mkldnn=False)
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
        extra_warnings = []          # also used by Layer 4C, further down
        info = closeup_analyse(ref)

        # LAYOUT, then the old bbox as a fallback.
        #
        # closeup.text_bbox() is the bounding box of EVERY text line in the
        # frame. That is not an article: on all nine captures in
        # backend/inbox it also contained the next story's headline along the
        # bottom, and on eight of them a column the frame edge had sliced in
        # half. layout.analyse() finds the columns, drops the clipped ones,
        # and crops to the block spanning the frame centre.
        #
        # The crop it returns is in DESKEWED coordinates, so every frame has
        # to be rotated by the same angle before it is used - cropping an
        # un-rotated frame with them is off by H*tan(angle) at the extremes,
        # about 40 px on these captures. Deskewing helps the OCR anyway.
        # LAYOUT FIRST, YOLO ONLY IF LAYOUT REFUSES.
        #
        # The old gate was `is_closeup` (glyph_p75 >= 20), and it sent 29% of
        # real captures to the article detector — the path that returned a
        # confident box over the neighbouring article's headline. Measured:
        # layout analysis succeeds on 16 of those 20 frames, and corpus full
        # pages are still refused by the gutter gate with the p75 gate off.
        # See core/config.py, LAYOUT_MIN_P75.
        # Run the PaddleOCR-based Segmenter first on the full frame
        t0 = time.time()
        arts = self.seg.run(ref, conf=conf, max_articles=max_articles)
        
        # If the segmenter detected more than 1 separate article, we are NOT in a closeup.
        # Otherwise (0 or 1 articles), we treat it as a closeup/single article.
        single_article = (len(arts) <= 1)

        if single_article:
            lay = _layout.analyse(ref, min_p75=0)
            if lay.get('applicable'):
                ang = lay['deskew_deg']
                if abs(ang) > 0.05:
                    imgs = [_layout.deskew(i, ang)[0] for i in imgs]
                    ref = lay['upright']
                    info = closeup_analyse(ref)   # title bbox in the same frame
                x1, y1, x2, y2 = lay['crop']
                extra_warnings += _layout.warnings_for(lay)
            else:
                # Layout refused but the frame is close. The only thing left is
                # the bounding box of ALL text in the frame, which is NOT an
                # article — it is whatever the camera happened to see.
                x1, y1, x2, y2 = info['bbox']
                extra_warnings.append(
                    'could not find the article boundaries in this frame; '
                    'read the text that was visible')

            box = Box(x1=x1, y1=y1, x2=x2, y2=y2)
            
            # Use Paddle OCR (Segmenter) to detect titles and bodies within this block
            try:
                from layers.l3_segment.geometry import order_columns
                crop = ref[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    body_by_art, title_by_art = self.seg._regions(crop, [[0, 0, x2-x1, y2-y1]])
                    regions = ([Region(box=Box(x1=r_[0]+x1, y1=r_[1]+y1, x2=r_[2]+x1, y2=r_[3]+y1), label='title') 
                               for r_ in title_by_art.get(0, [])] +
                              [Region(box=Box(x1=r_[0]+x1, y1=r_[1]+y1, x2=r_[2]+x1, y2=r_[3]+y1), label='text') 
                               for r_ in order_columns(body_by_art.get(0, []))])
                else:
                    regions = []
            except Exception as e:
                print(f"PaddleOCR failed in closeup, fallback: {e}")
                regions = []
                
            if not regions:
                title_box = None
                if lay.get('applicable'):
                    _lines, _med = closeup_text_lines(ref)
                    try:
                        title_box = _headline_for_block(ref, lay['block'], (lay['crop'][0], lay['crop'][2]), _med)
                    except Exception:
                        pass
                if title_box:
                    tx1, ty1, tx2, ty2 = title_box
                    regions = [
                        Region(box=Box(x1=tx1, y1=ty1, x2=tx2, y2=ty2), label='title'),
                        Region(box=box, label='text')
                    ]
                else:
                    regions = [Region(box=box, label='text')]
                
            title_box = next((r.box for r in regions if r.label == 'title'), None)

            note = (f"close-up: {info['n_lines']} text lines, "
                    f"crop {info['bbox_frac']:.0%} of frame")
            if lay.get('applicable'):
                note = (f"close-up: {lay['n_columns']} columns"
                        + (f" ({len(lay['clipped_columns'])} clipped by the "
                           f"frame edge, dropped)"
                           if lay['clipped_columns'] else "")
                        + f", {lay['n_blocks']} blocks, reading the one at "
                          f"frame centre ({lay['lines_in_block']:.0f} lines, "
                          f"pitch {lay['pitch']:.0f}px, "
                          f"deskew {lay['deskew_deg']:+.2f} deg)")
            if title_box:
                note += '; headline located and attached to this article'
            elif info.get('n_title_lines'):
                note += ('; headline bands were found but none could be '
                         'attached to this article with confidence')
            arts = [Article(index=0, box=box, regions=regions,
                            glyph_p75=info['glyph_p75'], verdict='ok',
                            note=note)]
            t['segment'] = round(time.time() - t0, 2)
            t['mode'] = 'closeup'
        else:
            # We already computed `arts` from the layout segmenter, so just store timings
            t['segment'] = round(time.time() - t0, 2)
            t['mode'] = 'layout'

        # SAVE CROPS FOR FRONTEND
        # We save the cropped image pieces directly into the job directory.
        job_dir = Path(paths[0]).parent if 'paths' in locals() else Path(image_paths[0]).parent
        for art in arts:
            # Crop the main article block
            abox = art.box
            ax1, ay1, ax2, ay2 = int(abox.x1), int(abox.y1), int(abox.x2), int(abox.y2)
            art_crop = ref[max(0, ay1):ay2, max(0, ax1):ax2]
            art_path = job_dir / f"art_{art.index}.png"
            if art_crop.size > 0:
                cv2.imwrite(str(art_path), art_crop)
                art.crop_path = str(art_path.name) # Just store the filename so the frontend can append it to the job URL

            # Crop each sub-region (title / text)
            for r_idx, r in enumerate(art.regions):
                rx1, ry1, rx2, ry2 = int(r.box.x1), int(r.box.y1), int(r.box.x2), int(r.box.y2)
                r_crop = ref[max(0, ry1):ry2, max(0, rx1):rx2]
                r_path = job_dir / f"art_{art.index}_{r.label}_{r_idx}.png"
                if r_crop.size > 0:
                    cv2.imwrite(str(r_path), r_crop)
                    r.crop_path = str(r_path.name)

        t0 = time.time()
        arts = [l4a.extract(ref, a) for a in arts]
        t['title'] = round(time.time() - t0, 2)

        t0 = time.time()
        if single_article:
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

        # LAYER 4C — optional LLM post-edit. OFF unless SINHALA_POLISH_MODE
        # says otherwise, and it writes to `body_polished`, never to `body`:
        # `body` is the mT5 output Chapter 4's CER is measured on.
        #
        # Every article it touches carries a warning all the way to the phone,
        # including when a guard REJECTED the rewrite, so a transcript can
        # never be read as unassisted output and a rejection is never silent.
        if POLISH_MODE.lower() != 'off':
            t0 = time.time()
            for a in arts:
                res = _polish.polish(a.body or a.body_raw)
                if res['applied']:
                    a.body_polished = res['text']
                if res['applied'] or 'REJECTED' in res['reason']:
                    extra_warnings.append(f"Article {a.index + 1}: "
                                          f"{res['reason']}")
            t['polish'] = round(time.time() - t0, 2)

        doc = assemble(arts, [f.path for f in frames], t)
        # Sentences the listener can act on ("part of this article is off the
        # right of the frame - move a little to the right"). Without these the
        # system reads a fragment with no indication that it is one.
        doc.warnings = list(extra_warnings) + list(doc.warnings)
        doc.timings['total'] = round(sum(v for v in t.values()
                                 if isinstance(v, (int, float))), 2)
        return doc

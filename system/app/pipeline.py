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
        lay = _layout.analyse(ref, min_p75=LAYOUT_MIN_P75)
        single_article = bool(lay.get('applicable')) or info['is_closeup']
        if lay.get('applicable'):
            ang = lay['deskew_deg']
            if abs(ang) > 0.05:
                imgs = [_layout.deskew(i, ang)[0] for i in imgs]
                ref = lay['upright']
                info = closeup_analyse(ref)   # title bbox in the same frame
            x1, y1, x2, y2 = lay['crop']
            extra_warnings += _layout.warnings_for(lay)
        elif info['is_closeup']:
            # Layout refused but the frame is close. The only thing left is
            # the bounding box of ALL text in the frame, which is NOT an
            # article — it is whatever the camera happened to see. Measured at
            # 3% of real captures. It is a fallback, and the warning says so.
            x1, y1, x2, y2 = info['bbox']
            extra_warnings.append(
                'could not find the article boundaries in this frame; '
                'read the text that was visible')

        if single_article:
            box = Box(x1=x1, y1=y1, x2=x2, y2=y2)
            regions = [Region(box=box, label='text')]
            # THE HEADLINE THAT BELONGS TO THIS BODY.
            #
            # An article is a headline plus the body under it. Until 27 Aug
            # 2026 only the body was read, and the headline — the part a
            # listener uses to decide whether to keep listening — was dropped.
            #
            # `headline_for_block` attaches one only when it can tell which
            # band is the headline, and REFUSES otherwise: these pages carry a
            # masthead, a page number and a section strip above the headline,
            # all of them headline-sized, and reading those aloud as the
            # headline would be worse than saying nothing.
            title_box = None
            if lay.get('applicable'):
                _lines, _med = closeup_text_lines(ref)
                title_box = _headline_for_block(
                    ref, lay['block'], (lay['crop'][0], lay['crop'][2]), _med)
            if title_box:
                tx1, ty1, tx2, ty2 = title_box
                regions.insert(0, Region(
                    box=Box(x1=tx1, y1=ty1, x2=tx2, y2=ty2), label='title'))
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
        elif SEGMENT_MODE.lower() == 'yolo':
            arts = self.seg.run(ref, conf=conf, max_articles=max_articles)
            t['segment'] = round(time.time() - t0, 2)
            t['mode'] = 'yolo'
        else:
            # TOO FAR TO IDENTIFY AN ARTICLE.
            #
            # Measured over 70 real captures (tools/probe_yolo.py): where the
            # detector and the layout path both answered, they disagreed on
            # 69% of frames, and the layout crop is the better-evidenced side.
            # Reading a confidently-wrong story to someone who cannot check it
            # is worse than reading nothing.
            #
            # "Move a little closer" is also the instruction that fixes the
            # frame. See core/config.py, SEGMENT_MODE.
            t['segment'] = round(time.time() - t0, 2)
            t['mode'] = 'too-far'
            return Document(
                warnings=['Could not identify a single article in this frame '
                          '- move a little closer and try again'],
                timings=t)

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

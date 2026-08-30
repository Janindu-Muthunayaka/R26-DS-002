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
                import os, paddle
                # Enforce CPU based on user request
                dev_str = 'cpu'
                model_dir = str(Path.home() / ".paddlex" / "official_models" / "PP-DocLayout_plus-L")
                if os.path.exists(model_dir):
                    layout = LayoutDetection(model_dir=model_dir,
                                             threshold=0.20,
                                             device=dev_str)
                else:
                    layout = LayoutDetection(model_name='PP-DocLayout-L',
                                             threshold=0.20,
                                             device=dev_str)
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
            max_articles=MAX_ARTICLES, correct=True, doc=None) -> Document:
        import concurrent.futures
        from threading import Lock

        if doc is None:
            doc = Document()

        t = {}
        t0 = time.time()
        frames = select(image_paths)
        t['select'] = round(time.time() - t0, 2)
        doc.timings['select'] = t['select']
        if not frames:
            doc.warnings.append('no usable frames')
            return doc

        doc.source_frames = [f.path for f in frames]
        imgs = [imread_upright(f.path) for f in frames]
        imgs = [i for i in imgs if i is not None]
        ref = imgs[0]

        doc.progress_log.append("Started article detection...")
        # 1. First Pass: Segmenter (Layout full image)
        t0 = time.time()
        art_groups = self.seg.get_article_boxes(ref, max_articles=max_articles)
        t['segment_pass1'] = round(time.time() - t0, 2)
        doc.progress_log.append(f"Articles detected: {len(art_groups)}")
        
        # Populate initial empty articles into document
        initial_arts = [ag[0] for ag in art_groups]
        doc.articles = initial_arts
        
        single_article = (len(art_groups) <= 1)
        doc.timings['mode'] = 'closeup' if single_article else 'layout'

        job_dir = Path(image_paths[0]).parent
        extra_warnings = []
        extra_warnings_lock = Lock()

        # Helper function to process a single article
        def process_article(art, group):
            nonlocal extra_warnings
            try:
                doc.progress_log.append(f"Cropping article {art.index + 1}...")
                # 2. Second Pass: Extract Regions (PaddleOCR on crop)
                t_sub0 = time.time()
                art = self.seg.extract_regions(ref, art, group)

                # Save crops for frontend
                abox = art.box
                ax1, ay1, ax2, ay2 = int(abox.x1), int(abox.y1), int(abox.x2), int(abox.y2)
                art_crop = ref[max(0, ay1):ay2, max(0, ax1):ax2]
                art_path = job_dir / f"art_{art.index}.png"
                if art_crop.size > 0:
                    cv2.imwrite(str(art_path), art_crop)
                    art.crop_path = str(art_path.name)

                for r_idx, r in enumerate(art.regions):
                    rx1, ry1, rx2, ry2 = int(r.box.x1), int(r.box.y1), int(r.box.x2), int(r.box.y2)
                    r_crop = ref[max(0, ry1):ry2, max(0, rx1):rx2]
                    r_path = job_dir / f"art_{art.index}_{r.label}_{r_idx}.png"
                    if r_crop.size > 0:
                        cv2.imwrite(str(r_path), r_crop)
                        r.crop_path = str(r_path.name)

                # 3. Title OCR
                art = l4a.extract(ref, art)

                # 4. Body OCR
                if single_article:
                    art = self.body.read_page(imgs, art)
                else:
                    art = self.body.read(imgs, art)

                # 5. Correct OCR
                if correct:
                    art = self.body.correct(art)
                else:
                    art.body = art.body_raw

                doc.progress_log.append(f"Article {art.index + 1} scanned")
                return art
            except Exception as e:
                import traceback
                traceback.print_exc()
                with extra_warnings_lock:
                    extra_warnings.append(f"Article {art.index + 1} failed: {e}")
                return art

        # Process all articles in parallel
        t0 = time.time()
        final_arts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for art, group in art_groups:
                futures.append(executor.submit(process_article, art, group))
            
            for future in concurrent.futures.as_completed(futures):
                final_arts.append(future.result())

        # Sort articles back to reading order (index)
        final_arts.sort(key=lambda a: a.index)
        t['processing_parallel'] = round(time.time() - t0, 2)
        
        # Replace document articles with completed ones
        doc.articles = final_arts

        # 6. Polish (Batch)
        if POLISH_MODE.lower() != 'off':
            t0_polish = time.time()
            from layers.l4c_polish import polish as _polish
            results = _polish.polish_articles(final_arts)
            for art, res in zip(final_arts, results):
                art.polish_reason = res['reason']
                if res['applied']:
                    art.body_polished = res['body']
                    art.title_polished = res['title']
                if res['applied'] or 'REJECTED' in res['reason']:
                    extra_warnings.append(f"Article {art.index + 1}: {res['reason']}")
            t['polish_batch'] = round(time.time() - t0_polish, 2)

        # Final assembly
        t0 = time.time()
        new_doc = assemble(final_arts, [f.path for f in frames], t)
        new_doc.warnings = list(extra_warnings) + list(new_doc.warnings)
        new_doc.timings['total'] = round(sum(v for v in t.values() if isinstance(v, (int, float))), 2)
        new_doc.progress_log = doc.progress_log
        new_doc.generations = getattr(doc, 'generations', [])
        
        # 7. Layer 8B Speech / TTS data extraction for phone application
        tts_text = l6.get_tts_text(new_doc)
        if tts_text:
            new_doc.tts_text = tts_text
            new_doc.progress_log.append("TTS speech data prepared for phone app")
        
        return new_doc

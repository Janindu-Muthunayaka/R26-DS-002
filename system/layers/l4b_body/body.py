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
from core.textutils import norm, strong_dedup, vote_lines, sentences
from core.config import (TESS_CONFIG, TESS_CONFIG_PAGE, TESS_LANG,
                         CLOSEUP_OCR_SCALE, MT5_NUM_BEAMS,
                         MT5_NO_REPEAT_NGRAM, MT5_MAX_LENGTH, MT5_BATCH)


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

    def read_page(self, frames_imgs, article: Article,
                  scale=CLOSEUP_OCR_SCALE) -> Article:
        """CLOSE-UP PATH — OCR one multi-column crop as a page.

        Differs from read() in two ways that are both measured, not stylistic:

          * fixed downscale instead of rescale_to_optimum. On a close-up the
            per-region p90 rule chose 0.19 and Tesseract returned 0
            characters; a 53px line became 10px, under the ~11px floor at
            which diacritics disappear. 0.40 was selected by eye across
            1.0/0.6/0.4 — see core/config.py, it is a judgement not a CER.

          * TESS_CONFIG_PAGE (psm 3, column-aware) instead of TESS_CONFIG
            (psm 6, single block). psm 6 spliced adjacent columns together
            mid-sentence on a real capture.
        """
        b = article.box
        per_frame = []
        for img in frames_imgs:
            crop = img[max(0, int(b.y1)):int(b.y2), max(0, int(b.x1)):int(b.x2)]
            if crop.size == 0:
                continue
            if scale and abs(scale - 1.0) > 1e-3:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
            txt = self.pt.image_to_string(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                lang=TESS_LANG, config=TESS_CONFIG_PAGE)
            txt = '\n'.join(norm(l) for l in txt.split('\n') if l.strip())
            if txt:
                per_frame.append(txt)

        raw = vote_lines(per_frame) if len(per_frame) > 1 else \
              (per_frame[0] if per_frame else '')
        article.body_raw = strong_dedup(raw)
        article.ocr_scale = scale
        return article

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
                per_frame.append('\n'.join(parts))

        raw = vote_lines(per_frame) if len(per_frame) > 1 else \
              (per_frame[0] if per_frame else '')
        article.body_raw = strong_dedup(raw)
        article.glyph_p90 = article.glyph_p90 or p90
        article.ocr_scale = scale
        return article

    @torch.no_grad()
    def correct(self, article: Article) -> Article:
        if not article.body_raw.strip():
            return article
        # SENTENCES, not lines. strong_dedup() collapses the article to a
        # single line, so splitting on '\n' yields one unit and mT5 truncates
        # the whole article at 128 tokens with no error. See textutils.sentences.
        out = self.correct_lines(sentences(article.body_raw))
        article.body = strong_dedup(' '.join(out))
        return article

    @torch.no_grad()
    def correct_lines(self, lines, batch=None):
        """Correct a list of sentences. Batch size comes from config and
        defaults to 1 — see MT5_BATCH for why that is not simply raised."""
        lines = [l for l in lines if l and l.strip()]
        if not lines:
            return []
        n = int(batch or MT5_BATCH)
        gen = dict(max_length=MT5_MAX_LENGTH, num_beams=MT5_NUM_BEAMS,
                   no_repeat_ngram_size=MT5_NO_REPEAT_NGRAM,
                   early_stopping=True)
        out = []
        for i in range(0, len(lines), max(1, n)):
            chunk = lines[i:i + max(1, n)]
            e = self.tok(chunk, return_tensors='pt', truncation=True,
                         padding=(len(chunk) > 1),
                         max_length=MT5_MAX_LENGTH).to(self.dev)
            g = self.mdl.generate(**e, **gen)
            out.extend(self.tok.batch_decode(g, skip_special_tokens=True))
        return out

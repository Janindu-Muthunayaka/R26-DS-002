"""
LAYER 2 — burst frame selection.
OWNER: Ishara

IN : list of image paths
OUT: List[Frame], best first, weak frames dropped

Two gates, not one:
  * sharpness  — rejects motion blur and focus hunt
  * glyph_p75  — rejects photos taken too far away, where the pilla fall
                 below the recoverable threshold

The second gate is the one that matters. A perfectly sharp photo of a whole
page from arm's length is still unreadable.

CHANGED 20 Aug 2026: this layer measured glyph_p90 against a threshold of 22
until that date. Both were wrong for this stage — p90 >= 22 came from the
OCR-time optimum, and on the 168-page corpus it accepts 154 pages where the
capture app accepts 8. See core/config.py.
"""
from core.schemas import Frame
from core.imaging import (sharpness, glyph_p75, capture_verdict,
                          imread_upright)
from core.config import BURST_KEEP, MIN_SHARPNESS, SHARP_MIN_RATIO


def select(paths, keep=BURST_KEEP):
    frames = []
    for p in paths:
        im = imread_upright(p)   # EXIF-aware; cv2.imread is not
        if im is None:
            continue
        h, w = im.shape[:2]
        s = sharpness(im)
        p75 = glyph_p75(im)
        v, note = capture_verdict(p75)
        frames.append(Frame(path=str(p), width=w, height=h,
                            sharpness=s, glyph_p75=p75, verdict=v, note=note))
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

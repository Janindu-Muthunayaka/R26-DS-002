"""
LAYER 3 (close-up variant) — ARTICLE BOUNDARIES INSIDE ONE FRAME. OWNER: Ishara.

WHAT THIS ANSWERS
-----------------
closeup.py answers "where is the text?" and crops to the bounding box of every
text line it finds. That is not an article. It can contain a neighbouring
story, and it cannot tell whether the article continued past the edge of the
frame. Both are spoken as if they were the whole thing.

This module measures the page structure inside one frame:

    deskew()        in-plane rotation, corrected before anything is projected
    glyph_mask()    ink of BODY-TEXT-SIZED components only
    column_bands()  vertical gutters -> columns, and which columns are CLIPPED
                    by the frame edge
    row_profile()   ink per row, over the columns that are whole
    pitch()         baseline period, by autocorrelation
    blocks()        wide white gaps -> vertically stacked stories
    analyse()       all of it, plus the four open/closed edge verdicts

HOW IT WORKS, AND WHY NOT THE OBVIOUS WAY
-----------------------------------------
Everything here is built on PROJECTIONS of a filtered ink mask, not on the
line boxes from closeup.text_lines(). The first version of this module used
those boxes and it produced nonsense — see "WHAT THE FIRST VERSION GOT WRONG".

Two filtering steps make the projections trustworthy:

  * components are kept only if their height is between GLYPH_H_MIN and
    GLYPH_H_MAX *and* no more than TALL_GLYPH_MULT x the median. That single
    test removes the hand (one enormous component), the photographs, the page
    border, the speckle, AND the headlines — all without knowing what any of
    them are. What survives is body text.

  * the frame is deskewed first. A 1 deg tilt drags a column sideways by
    57 px over 3264 px of height, comparable to a gutter, so gutters smear
    and columns merge. Measured skew on the nine captures: -0.65 to +0.20
    deg, with 0.0-0.1 deg residual after correction. The deskew estimator is
    a projection-profile search, NOT cv2.minAreaRect -- see deskew_angle()
    for why that distinction cost a day. Small, and it still mattered.

The pitch estimate is an autocorrelation of the row profile, not the spacing
of detected boxes. That is the same reasoning as the capture app's line-pitch
estimator, which the research already establishes as blur-invariant: on the
one badly blurred capture in the set (glyph_p75 = 11, 201 spurious line boxes)
this still returns pitch 48 px, in line with the 49 and 48 measured on the two
sharp frames of the same burst.

WHAT THE FIRST VERSION GOT WRONG  (22 Aug 2026, my error)
---------------------------------------------------------
Building columns and pitch from closeup.text_lines() boxes failed on real
frames in three ways at once:

  * one merged box 1811 px wide bridged a gutter and collapsed three columns
    into one band. Column count came out 3, 3, 3, 1, 3, 1, 2, 2, 2 across nine
    frames of three static scenes. It should have been constant.
  * with columns merged, sorting boxes by y interleaved lines from different
    columns, so the median top-to-top spacing halved or thirded. Reported
    pitches were 51.5, 27, 15 and 10 px on frames whose true pitch is 48-56.
  * the reported inter-line gap distribution had a median of MINUS 0.75
    pitch. A negative median gap is not a page property; it is proof the
    boxes were not a single column's lines.

Every constant chosen from that run would have been chosen from noise. The
projection method above gives 3 columns on all nine frames and pitch stable
to +/-1 px within each burst.

CALIBRATION STATUS
------------------
  measured, n=9 phone frames / 3 scenes, one newspaper page:
      gutters 79-125 px on a 2448 px frame (0.032-0.051 W)
      whole columns 588-920 px; clipped edge columns 208-365 px
      pitch 48-56 px, constant to +/-1 px within a burst
      the gap that separates the body from the next story: 1.0-3.8 pitch
  measured, n=6 corpus pages (5 half, 1 full), two mastheads:
      THIS METHOD DOES NOT WORK ON THEM, and analyse() now refuses rather
      than guessing. A corpus "half page" is a whole newspaper page holding
      seven or eight stories with photographs and headlines spanning columns
      at every height, so the vertical projection never reaches zero: 1 or 2
      bands were found where there are six or seven columns, and on
      lankadeepa p20 the pitch came out 12 px against a true 30. Full-page
      layout is the article detector's problem, at the framing it was
      trained on. glyph_p75 separates the two cases cleanly on this sample:
      corpus 18-24, phone captures 30-36, against CLOSEUP_MIN_P75 = 28.
  NOT measured:
      a second newspaper at close range; a close-up of a single-column or
      two-column story; anything at all outside 28 <= glyph_p75 <= 36.
      TALL_GLYPH_MULT, and every threshold below, is a starting value.
"""
import cv2
import numpy as np

from core.config import GLYPH_H_MIN, GLYPH_H_MAX, CLOSEUP_MIN_P75
from core.imaging import glyph_p75

# --------------------------------------------------------------------------
# Starting values. See CALIBRATION STATUS above before quoting any of these.
# --------------------------------------------------------------------------
TALL_GLYPH_MULT  = 2.5    # components taller than this x median are not body
DESKEW_MAX_DEG   = 8.0    # beyond this it is not skew, it is a rotated page
GUTTER_MIN_FRAC  = 0.015  # a gutter must be this fraction of frame width
COL_MIN_FRAC     = 0.06   # a band narrower than this is not a column
COL_CLIP_FRAC    = 0.70   # an edge band below this x median width is clipped
EDGE_INK_GLYPHS  = 1.0    # width of the strip inspected at each frame edge
EDGE_INK_FRAC    = 0.10   # ink in this fraction of its rows = text runs off
COL_INK_FRAC     = 0.12   # column profile: ink threshold, x median
ROW_INK_FRAC     = 0.12   # row profile: ink threshold, x p75
BLOCK_GAP_PITCH  = 1.20   # white gap above this x pitch separates two blocks
BLOCK_MIN_PITCH  = 1.50   # a block shorter than this x pitch is not a story
EDGE_OPEN_PITCH  = 1.00   # text within this x pitch of an edge was cut off
PITCH_MIN_GLYPHS = 1.20   # pitch below this x median glyph height is impossible


# ==========================================================================
# preparation
# ==========================================================================
def _row_sharpness(mask):
    """How crisp is the row profile? Sum of squared first differences.

    Text that is level gives alternating dark and white rows, so the profile
    swings hard. Tilt smears the swings out. Maximising this IS deskewing,
    and it optimises exactly the signal every later step depends on.
    """
    p = mask.sum(1).astype(np.float64)
    if p.size < 8:
        return 0.0
    d = np.diff(p)
    return float(np.dot(d, d))


def _rot_mask(m, a):
    H, W = m.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), a, 1.0)
    return cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)


def deskew_angle(img, max_deg=DESKEW_MAX_DEG, work_w=600):
    """Angle, in degrees, to FEED BACK to cv2.getRotationMatrix2D to level
    the text. 0.0 when undecidable.

    DO NOT REPLACE THIS WITH cv2.minAreaRect. The first version did, and its
    answer depended on the OpenCV build:

        same nine frames, same code
        cv2 4.13 here      0.00, -1.38, -1.00, -0.14, -0.42, -0.83, -0.77 ...
        Ishara's machine  +0.90, +0.86, +0.75, +1.20, +1.11, +0.79, +0.99 ...

    OpenCV changed the angle range minAreaRect returns (it was [-90, 0), it
    became (0, 90]), so `if w < h: a += 90` selects a different family of
    rectangles on different builds. On one of the two machines deskew was
    therefore rotating the WRONG WAY and doubling the tilt, silently. This
    is the third time a library version has changed a result in this project
    -- after EXIF orientation and the transformers CER -- and the lesson is
    the same one: do not build on an API whose convention can move.

    Searched instead: coarse then fine, maximising _row_sharpness(). No
    ambiguous convention, and the sign is defined by what it is used for.
    Measured: exact to 0.00 deg on synthetic tilts of +/-1.5 and +/-3.0, and
    the residual tilt after correcting the nine real captures is 0.0-0.1 deg.
    Costs about 0.2 s on a 2448x3264 frame.
    """
    m, _med = glyph_mask(img)
    if not m.any():
        return 0.0
    H, W = m.shape
    small = cv2.resize(m, (work_w, max(8, int(H * work_w / float(W)))),
                       interpolation=cv2.INTER_AREA)
    small = (small > 0).astype(np.uint8)
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + 1e-9, 0.5):
        v = _row_sharpness(_rot_mask(small, a))
        if v > best:
            best, best_a = v, float(a)
    for a in np.arange(best_a - 0.5, best_a + 0.5 + 1e-9, 0.05):
        v = _row_sharpness(_rot_mask(small, a))
        if v > best:
            best, best_a = v, float(a)
    return round(best_a, 2)


def deskew(img, angle=None):
    """Rotate the frame upright. Returns (rotated, angle_used)."""
    a = deskew_angle(img) if angle is None else angle
    if abs(a) < 0.05:
        return img, a
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), a, 1.0)
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderValue=border), a


def glyph_mask(img, tall_mult=TALL_GLYPH_MULT):
    """Binary mask of BODY-TEXT ink only. Returns (mask 0/1, median height).

    The height filter is doing all the work and it is worth being explicit
    about what it removes, because none of it is removed by name:

        a hand          one component hundreds of px tall      -> dropped
        a photograph    likewise                                -> dropped
        the page border likewise                                -> dropped
        a HEADLINE      glyphs ~4x the body height              -> dropped
        speckle         below GLYPH_H_MIN                       -> dropped

    Dropping headlines is deliberate. A headline spans several columns and
    would bridge every gutter in the vertical projection. closeup.title_lines()
    finds headlines separately; this mask is for structure.
    """
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n < 2:
        return np.zeros(g.shape, np.uint8), 0.0
    h = st[1:, cv2.CC_STAT_HEIGHT].astype(float)
    w = st[1:, cv2.CC_STAT_WIDTH].astype(float)
    ok = (h >= GLYPH_H_MIN) & (h <= GLYPH_H_MAX)
    if not ok.any():
        return np.zeros(g.shape, np.uint8), 0.0
    med = float(np.median(h[ok]))
    ok &= (h <= tall_mult * med) & (w <= tall_mult * med * 8.0)
    keep = np.zeros(n, bool)
    keep[np.arange(1, n)[ok]] = True
    return keep[lab].astype(np.uint8), med


# ==========================================================================
# columns
# ==========================================================================
def _runs(flag, min_len):
    out, s = [], None
    for i, v in enumerate(flag):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(flag) - s >= min_len:
        out.append((s, len(flag)))
    return out


def column_bands(mask, med_glyph, gutter_min_frac=GUTTER_MIN_FRAC,
                 col_min_frac=COL_MIN_FRAC, ink_frac=COL_INK_FRAC):
    """Column bands [(x1, x2), ...] from the vertical ink projection.

    Smoothed over one glyph width first, so the space between two words does
    not read as a gutter. A gutter must additionally be at least
    `gutter_min_frac` of the frame wide; measured gutters on the phone
    captures are 0.032-0.051 W, and paragraph indents are far below that.
    """
    H, W = mask.shape
    prof = mask.sum(0).astype(float)
    k = max(3, int(round(med_glyph)) or 3)
    prof = np.convolve(prof, np.ones(k) / k, 'same')
    pos = prof[prof > 0]
    if not pos.size:
        return []
    thr = ink_frac * float(np.median(pos))
    ink = prof > thr
    gutter_min = max(1, int(round(gutter_min_frac * W)))
    # close gutters that are too narrow to be gutters
    for g0, g1 in _runs(~ink, 1):
        if (g1 - g0) < gutter_min:
            ink[g0:g1] = True
    return [(a, b) for a, b in _runs(ink, 1)
            if (b - a) >= col_min_frac * W]


def clipped_bands(bands, W, clip_frac=COL_CLIP_FRAC, edge_px=3):
    """Which columns are cut off by the left or right frame edge.

    THIS IS THE ANSWER FOR THE CAPTURES IN backend/inbox. Every one of the
    nine has a column running off one side: a band that touches the frame
    edge and is far narrower than its neighbours. The article did not fit
    ACROSS the frame — and nothing in the pipeline noticed, because the
    partial column is still text and Tesseract still reads it, producing the
    first few characters of every line.

    Returns (left_open, right_open, [indices of clipped bands]).
    """
    if not bands:
        return False, False, []
    med = float(np.median([b[1] - b[0] for b in bands]))
    left = right = False
    idx = []
    for i, (a, b) in enumerate(bands):
        if (b - a) >= clip_frac * med:
            continue
        if a <= edge_px:
            left = True
            idx.append(i)
        elif b >= W - edge_px:
            right = True
            idx.append(i)
    return left, right, idx


# ==========================================================================
# rows, pitch, blocks
# ==========================================================================
def edge_ink(mask, med_glyph, band_glyphs=EDGE_INK_GLYPHS,
             y_lo=0.20, y_hi=0.80):
    """Fraction of rows with ink in the strip at each frame edge.

    (left_fraction, right_fraction). This is the AUTHORITATIVE test for "the
    article runs off the side", and it is deliberately independent of gutter
    detection, because gutter detection misses two real cases:

      * a column cut at 75% of its width is not narrow enough to look
        clipped (burst_105901_g26_s3995_2: right band 613 px against a
        median of 613, flagged clean, ink at the right edge in 35% of rows)
      * a narrow gutter merges the clipped column into its neighbour, so no
        edge-touching band is narrow at all (burst_105901_g26_s3995_5:
        26 px gutter, three columns reported as two, clipping missed)

    A page has margins. If body-text ink reaches the outermost glyph-width
    of the frame across a tenth of the rows, the page did not end there --
    the frame did.

    Measured: 0.00 on both edges of a synthetic frame holding whole columns;
    0.16-0.41 on the eight real captures that are clipped. The middle 60% of
    the frame height is used so a headline or a footer cannot vote.
    """
    H, W = mask.shape
    b = max(4, int(round(band_glyphs * med_glyph)) if med_glyph else 4)
    b = min(b, max(1, W // 4))
    seg = mask[int(H * y_lo):int(H * y_hi)]
    if not seg.size:
        return 0.0, 0.0
    return (float((seg[:, :b].sum(1) > 0).mean()),
            float((seg[:, W - b:].sum(1) > 0).mean()))


def row_profile(mask, bands=None):
    """Ink per row, summed over the given column bands (default: all)."""
    if bands is None:
        return mask.sum(1).astype(float)
    p = np.zeros(mask.shape[0], float)
    for a, b in bands:
        p += mask[:, a:b].sum(1)
    return p


def pitch(prof, lo=12, hi=400):
    """Baseline period in px, by autocorrelation of the row profile.

    Not the spacing of detected line boxes. Autocorrelation uses every row of
    evidence in the column at once, so a few merged or missed lines move it
    barely at all, and it survives blur — which is the same property the
    capture app's estimator relies on and the research measured (+52.3% drift
    for component height under hand-held blur, under 1% for line pitch).
    """
    if prof.size < lo * 3:
        return 0.0
    q = prof - prof.mean()
    ac = np.correlate(q, q, 'full')[len(q) - 1:]
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return 0.0
    return float(lo + int(np.argmax(ac[lo:hi])))


def text_rows(prof, ink_frac=ROW_INK_FRAC):
    """Boolean per row: is there enough ink here to be a line of text?

    Thresholded against the 75th percentile of the profile rather than the
    median, because well over half the rows in a column are inter-line white.
    """
    if not prof.size or not (prof > 0).any():
        return np.zeros(prof.shape, bool)
    return prof > max(3.0, ink_frac * float(np.percentile(prof, 75)))


def blocks(rows, p, gap_pitch=BLOCK_GAP_PITCH, min_pitch=BLOCK_MIN_PITCH):
    """Vertically stacked text blocks [(y1, y2), ...] from the row mask.

    A block ends where the white gap exceeds `gap_pitch` x pitch. On the
    phone captures the gap between an article's last body line and the next
    story's headline measured 1.0-3.8 pitch, against inter-line gaps well
    under 1. That separation is what lets the next story be excluded.

    It does NOT distinguish "next article" from "sub-heading" — the geometry
    is the same. This narrows the crop; it does not identify a story.
    """
    if p <= 0:
        p = 1.0
    ys = np.where(rows)[0]
    if not ys.size:
        return []
    out, s, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if (y - prev) > gap_pitch * p:
            out.append((int(s), int(prev)))
            s = y
        prev = y
    out.append((int(s), int(prev)))
    return [b for b in out if (b[1] - b[0]) >= min_pitch * p]


def current_block(bls, H):
    """The block the user is aiming at — the one spanning the frame centre.

    Aiming is the only statement of intent a blind user makes, so the centre
    of the frame is the selection. Falls back to the tallest block.
    """
    if not bls:
        return None
    cy = H // 2
    for b in bls:
        if b[0] <= cy <= b[1]:
            return b
    return max(bls, key=lambda b: b[1] - b[0])


def edges_open(block, H, p, edge_pitch=EDGE_OPEN_PITCH):
    """Did this block run off the top or the bottom of the frame?

    White space of more than about a line above the first row is POSITIVE
    evidence that the text really begins there. No white space means it was
    cut. Margins are returned in px so the threshold can be re-measured.
    """
    if block is None:
        return False, False, 0.0, 0.0
    if p <= 0:
        p = 1.0
    top, bot = float(block[0]), float(H - block[1])
    lim = edge_pitch * p
    return top <= lim, bot <= lim, top, bot


# ==========================================================================
# one pass
# ==========================================================================
def _refused(img, p75, reason):
    H, W = img.shape[:2]
    return {'applicable': False, 'glyph_p75': p75, 'upright': img,
            'reason': reason,
            'crop': (0, 0, W, H), 'columns': [], 'clipped_columns': [],
            'n_columns': 0, 'n_blocks': 0, 'blocks': [], 'block': None,
            'pitch': 0.0, 'deskew_deg': 0.0, 'median_glyph_h': 0.0,
            'top_open': False, 'bottom_open': False,
            'left_open': False, 'right_open': False,
            'left_edge_ink': 0.0, 'right_edge_ink': 0.0,
            'top_margin_px': 0.0, 'bottom_margin_px': 0.0,
            'lines_in_block': 0.0, 'lines_per_frame': 0.0}


def analyse(img, min_p75=None):
    """Column and block structure of one frame, and four edge verdicts.

    `min_p75` overrides the close-up gate. FOR MEASUREMENT ONLY -- the
    deployed path must use the default. It exists because the gate hides the
    frames a distance experiment needs: a framing that holds a whole article
    can sit below CLOSEUP_MIN_P75, and refusing to measure it is how you
    conclude "impossible" from a threshold rather than from the page.

    `crop` excludes any column clipped by a frame edge. A column cut at 27%
    of its width yields the first few characters of every line - text that is
    not wrong so much as meaningless, and that mT5 will confidently repair
    into words nobody printed. Better to drop it and say so.

    KNOWN GAP, and it is a real one. `left_open`/`right_open` come from ink
    at the frame edge and are right on all nine captures; the crop, however,
    can only drop a WHOLE band. When a narrow gutter merges a clipped column
    into a good one (burst_105901_g26_s3995_5 -- 26 px gutter), the verdict
    is correct and the warning fires, but the crop still carries the clipped
    text. Splitting a merged band at its interior minimum would fix it and
    is not written. The user is being told to move anyway, so the next frame
    resolves it; this is a wart, not a hole.
    """
    H, W = img.shape[:2]
    p75 = glyph_p75(img)
    if min_p75 is None:
        min_p75 = CLOSEUP_MIN_P75
    if p75 is None or p75 < min_p75:
        # NOT A CLOSE-UP. Everything below assumes a frame that holds one
        # story in full-height columns. A whole newspaper page does not:
        # photographs and headlines span columns at every height, so the
        # vertical projection never reaches zero and the whole page comes
        # back as ONE column. Measured on five corpus half-pages -- 1 or 2
        # bands where there are six or seven columns, and on one of them a
        # pitch of 12 px against a true 30. Those are not near misses.
        #
        # Full-page layout is what the YOLO detector is for, at the framing
        # it was trained on. This returns "not applicable" rather than a
        # confident wrong answer.
        return _refused(img, p75,
                        f'not a close-up frame (glyph_p75 '
                        f'{p75 if p75 else 0:.0f} < {min_p75:.0f}) - '
                        "full-page layout is the article detector's job")

    up, angle = deskew(img)
    H, W = up.shape[:2]
    mask, med = glyph_mask(up)
    bands = column_bands(mask, med)
    _lb, _rb, clip_idx = clipped_bands(bands, W)
    l_ink, r_ink = edge_ink(mask, med)
    # The verdict comes from the ink at the edge; the band test only decides
    # which band to leave out of the crop. They disagree on two of the nine
    # real frames and the ink test is right both times -- see edge_ink().
    left_open = l_ink >= EDGE_INK_FRAC
    right_open = r_ink >= EDGE_INK_FRAC
    whole = [b for i, b in enumerate(bands) if i not in clip_idx]

    prof = row_profile(mask, whole or bands)
    p = pitch(prof)

    # PITCH CANNOT BE SHORTER THAN A GLYPH. Baselines are at least a glyph
    # apart or the lines would overlap, so a ratio under ~1.2 means the
    # autocorrelation locked onto the wrong peak and every number downstream
    # -- blocks, margins, lines-per-frame -- is meaningless.
    #
    # Caught a real case: work/13de9dea was a far shot whose component
    # heights are bimodal (body 19 px, headline much larger), so glyph_p75
    # came out 37 and it passed the close-up gate, then reported pitch 12 px
    # against a median glyph of 19 -- ratio 0.63 -- and pronounced the frame
    # "whole" with total confidence. Measured ratio on frames that ARE
    # close-ups: 1.50 to 1.74.
    if med <= 0 or p <= 0 or p < PITCH_MIN_GLYPHS * med:
        return _refused(img, p75,
                        f'line pitch {p:.0f}px against a median glyph of '
                        f'{med:.0f}px (ratio {p / med if med else 0:.2f}, '
                        f'need >={PITCH_MIN_GLYPHS:.2f}) - the pitch estimate '
                        'is wrong, so nothing measured here can be trusted')

    bls = blocks(text_rows(prof), p)
    blk = current_block(bls, H)
    top_open, bot_open, top_m, bot_m = edges_open(blk, H, p)

    src = whole or bands
    if src and blk:
        crop = (int(min(b[0] for b in src)), int(blk[0]),
                int(max(b[1] for b in src)), int(blk[1]))
    else:
        crop = (0, 0, W, H)

    return {
        'applicable': True,
        'glyph_p75': p75,
        'reason': '',
        'deskew_deg': angle,
        'upright': up,               # the deskewed frame the crop applies to
        'median_glyph_h': med,
        'pitch': p,
        'n_columns': len(bands),
        'columns': bands,
        'clipped_columns': [bands[i] for i in clip_idx],
        'left_edge_ink': l_ink,
        'right_edge_ink': r_ink,
        'n_blocks': len(bls),
        'blocks': bls,
        'block': blk,
        'crop': crop,
        'top_open': bool(top_open),
        'bottom_open': bool(bot_open),
        'left_open': bool(left_open),
        'right_open': bool(right_open),
        'top_margin_px': top_m,
        'bottom_margin_px': bot_m,
        'lines_in_block': ((blk[1] - blk[0]) / p) if (blk and p > 0) else 0.0,
        'lines_per_frame': (H / p) if p > 0 else 0.0,
    }


def warnings_for(a):
    """Plain sentences a blind listener can act on. Empty when nothing is cut.

    Direction words, not measurements. "glyph 19px" tells a sighted developer
    something; "move a little to the right" tells the user what to do.
    """
    w = []
    if not a.get('applicable', True):
        return w
    if a['right_open']:
        w.append('part of this article is off the right of the frame '
                 '— move a little to the right')
    if a['left_open']:
        w.append('part of this article is off the left of the frame '
                 '— move a little to the left')
    if a['bottom_open']:
        w.append('this article continues below — tilt down slowly')
    if a['top_open']:
        w.append('this article started above the frame — tilt up slowly')
    return w


# ==========================================================================
# joining frames the user panned across
# ==========================================================================
def _norm(s):
    return ''.join(s.split())


def overlap(a, b, min_lines=2, max_look=12, ratio=0.80):
    """How many trailing lines of `a` repeat as leading lines of `b`.

    Panning produces overlapping frames on purpose: the overlap is what
    proves two frames are consecutive rather than two unrelated parts of the
    page. Matched on TEXT, not pixels — the frames differ in scale, angle and
    exposure, which is exactly what feature registration struggles with,
    while the OCR of the same physical line is nearly identical across them.

    Two matching lines minimum. One is not evidence; Sinhala newsprint
    repeats short lines constantly.

    Returns 0 when no overlap is credible. The caller must treat that as
    "these may not be consecutive", NOT as "join them anyway".
    """
    from difflib import SequenceMatcher
    if not a or not b:
        return 0
    for k in range(min(max_look, len(a), len(b)), min_lines - 1, -1):
        ok = sum(1 for x, y in zip(a[-k:], b[:k])
                 if SequenceMatcher(None, _norm(x), _norm(y)).ratio() >= ratio)
        if ok >= max(min_lines, int(round(0.7 * k))):
            return k
    return 0


def join(frames, **kw):
    """Join per-frame line lists captured while panning across one article.

    Returns (lines, seams). A seam of 0 is an UNVERIFIED join and must be
    reported to the user, not hidden — silently concatenating two frames that
    are not consecutive manufactures a sentence nobody printed.
    """
    if not frames:
        return [], []
    out, seams = list(frames[0]), []
    for nxt in frames[1:]:
        k = overlap(out, nxt, **kw)
        seams.append(k)
        out.extend(nxt[k:])
    return out, seams

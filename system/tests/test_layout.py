"""Article boundaries inside one close-up frame.

SYNTHETIC PAGES, and that limitation is stated here rather than buried: a
synthetic page has straight columns, uniform leading and no perspective, so
these tests show the LOGIC is right, not that the THRESHOLDS are. The
thresholds were set from nine real captures (see layout.py, CALIBRATION
STATUS) and are still provisional.

What these do pin down is every failure the module exists for, and the two
failures of its own first version:

  * a column clipped by the frame edge is detected      (the real captures)
  * a story running off the bottom is detected          (a long article)
  * a story that fits is NOT reported as cut off
  * a neighbouring story below is excluded from the crop
  * pitch survives a frame where half the lines merge   (the blur case)
  * a full page is REFUSED, not answered wrongly
"""
import cv2
import numpy as np
import pytest

from layers.l3_segment import layout as L


# --------------------------------------------------------------------------
def page(cols=3, col_w=560, gutter=90, glyph=30, pitch=54, top=200,
         n_lines=40, H=2600, margin=80, gap_at=None, W=None, rng_seed=0):
    """A synthetic close-up: filled bars for text lines, one glyph tall.

    `glyph` is the bar height, which is what glyph_p75 measures, so it has
    to be >= CLOSEUP_MIN_P75 for analyse() to accept the frame at all --
    the same gate the real capture app applies.
    """
    rng = np.random.default_rng(rng_seed)
    W = W or margin * 2 + cols * col_w + (cols - 1) * gutter
    img = np.full((H, W), 245, np.uint8)
    for c in range(cols):
        x0 = margin + c * (col_w + gutter)
        y = top
        for i in range(n_lines):
            if gap_at is not None and i == gap_at:
                y += pitch * 3
            if y + glyph > H:
                break
            cv2.rectangle(img, (x0, y), (x0 + int(col_w * rng.uniform(.85, 1.)),
                                         y + glyph), 20, -1)
            y += pitch
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ---- the gate ------------------------------------------------------------
def test_a_full_page_is_refused_not_answered_wrongly():
    """Measured on five corpus half-pages: the projection method returns one
    or two columns where there are six or seven. A wrong layout silently
    changes what gets read aloud, so the frame is refused instead."""
    img = page(cols=6, col_w=260, gutter=40, glyph=14, pitch=26, n_lines=90)
    a = L.analyse(img)
    assert a['applicable'] is False
    assert 'close-up' in a['reason']
    assert L.warnings_for(a) == []


def test_a_closeup_is_accepted():
    assert L.analyse(page())['applicable'] is True


def test_the_gate_can_be_overridden_for_measurement():
    """A framing that holds a whole article can sit BELOW the deployed
    gate -- measured: the article fits at glyph_p75 25, the gate is 28. If
    the tool refuses to measure there, the experiment concludes 'impossible'
    from a threshold instead of from the page."""
    img = page(cols=4, col_w=380, gutter=60, glyph=16, pitch=28, n_lines=70)
    assert L.analyse(img)['applicable'] is False        # below CLOSEUP_MIN_P75
    assert L.analyse(img, min_p75=12)['applicable'] is True


def test_an_impossible_pitch_is_refused():
    """Baselines are at least a glyph apart or the lines would overlap, so a
    pitch below ~1.2x the median glyph means the autocorrelation locked onto
    the wrong peak.

    A real frame did this: work/13de9dea, a far shot with bimodal component
    heights (body 19 px, headline much larger). glyph_p75 came out 37, it
    passed the close-up gate, and it then reported pitch 12 px against a
    median glyph of 19 -- and pronounced the frame 'whole' with complete
    confidence. Frames that ARE close-ups measure 1.50 to 1.74."""
    prof = np.zeros(2000)
    prof[::13] = 100.0                       # period far below a glyph
    assert L.pitch(prof) < L.PITCH_MIN_GLYPHS * 30

    # end-to-end: force an impossible pitch and check analyse() refuses
    real = L.pitch
    L.pitch = lambda *_a, **_k: 5.0
    try:
        a = L.analyse(page())
    finally:
        L.pitch = real
    assert a['applicable'] is False
    assert 'pitch estimate is wrong' in a['reason']


# ---- columns -------------------------------------------------------------
def test_three_columns_are_found():
    a = L.analyse(page(cols=3))
    assert a['n_columns'] == 3, a['columns']


def test_ragged_line_ends_do_not_split_a_column():
    assert L.analyse(page(cols=1))['n_columns'] == 1


def test_a_thumb_in_the_gutter_does_not_merge_two_columns():
    """A hand is one enormous connected component, so the height filter in
    glyph_mask() removes it before anything is projected. No heuristic about
    where a hand tends to be."""
    img = page(cols=2, gutter=200)
    H, W = img.shape[:2]
    cv2.rectangle(img, (W // 2 - 70, 0), (W // 2 + 70, H), (35, 35, 35), -1)
    assert L.analyse(img)['n_columns'] == 2


def test_a_headline_spanning_columns_does_not_bridge_the_gutter():
    """The reason headlines are dropped from the mask. A headline crosses
    every gutter it spans; left in, it welds the page into one column."""
    img = page(cols=3, top=400)
    W = img.shape[1]
    cv2.rectangle(img, (100, 120), (W - 100, 260), (20, 20, 20), -1)
    assert L.analyse(img)['n_columns'] == 3


# ---- THE FAILURE THE REAL CAPTURES ACTUALLY HAVE ------------------------
def test_a_column_clipped_by_the_right_edge_is_detected():
    """All nine frames in backend/inbox have this and nothing noticed. The
    partial column still contains text, so Tesseract reads the first few
    characters of every line and mT5 repairs them into words nobody
    printed."""
    full = page(cols=3)
    img = full[:, :full.shape[1] - 420]        # slice the last column short
    a = L.analyse(img)
    assert a['right_open'] is True and a['left_open'] is False
    assert len(a['clipped_columns']) == 1
    assert any('right' in w for w in L.warnings_for(a))


def test_a_column_clipped_by_the_left_edge_is_detected():
    full = page(cols=3)
    img = full[:, 420:]
    a = L.analyse(img)
    assert a['left_open'] is True and a['right_open'] is False


def test_the_clipped_column_is_excluded_from_the_crop():
    full = page(cols=3)
    img = full[:, :full.shape[1] - 420]
    a = L.analyse(img)
    x1, _y1, x2, _y2 = a['crop']
    assert x2 <= a['clipped_columns'][0][0], (a['crop'], a['clipped_columns'])


def test_a_frame_holding_whole_columns_reports_no_lateral_cut():
    a = L.analyse(page(cols=3))
    assert not a['left_open'] and not a['right_open']
    assert a['clipped_columns'] == []
    assert a['left_edge_ink'] == 0.0 and a['right_edge_ink'] == 0.0


def test_clipping_is_caught_even_when_the_gutter_test_misses_it():
    """The gutter test alone missed two of nine real frames: one where the
    clipped column was still 100% of the median width, and one where a 26 px
    gutter merged it into its neighbour. Ink at the frame edge catches both.

    Simulated here by clipping so gently that the last band is still wide."""
    full = page(cols=3)
    img = full[:, :full.shape[1] - 120]        # only a sliver removed
    a = L.analyse(img)
    assert a['right_open'] is True, a['right_edge_ink']
    assert a['right_edge_ink'] >= L.EDGE_INK_FRAC


def test_a_headline_at_the_edge_cannot_trigger_a_false_clip():
    """Only the middle 60% of frame height votes, so furniture at the top or
    bottom of the frame cannot manufacture a lateral warning."""
    img = page(cols=3)
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 40), (W - 1, 150), (20, 20, 20), -1)
    a = L.analyse(img)
    assert not a['left_open'] and not a['right_open']


# ---- vertical ------------------------------------------------------------
def test_pitch_matches_the_layout():
    a = L.analyse(page(pitch=54))
    assert abs(a['pitch'] - 54) <= 2, a['pitch']


def test_pitch_survives_lines_merging_into_each_other():
    """The blurred capture in the set produced 201 spurious line boxes and a
    box-spacing pitch of 10 px. Autocorrelation of the row profile gave 48,
    against 49 and 48 on the sharp frames of the same burst. This is the
    synthetic version of that: blur the frame until components merge."""
    img = cv2.GaussianBlur(page(pitch=54), (0, 0), 5)
    p = L.pitch(L.row_profile(L.glyph_mask(img)[0]))
    assert abs(p - 54) <= 3, p


def test_text_running_off_the_bottom_is_reported_as_cut_off():
    """'when I zoom in as the app tells me it only captures half of a big
    article' -- that case, asserted."""
    a = L.analyse(page(n_lines=200, top=8, H=1400))
    assert a['bottom_open'] and a['top_open']
    assert any('continues below' in w for w in L.warnings_for(a))


def test_an_article_that_fits_is_not_reported_as_cut_off():
    a = L.analyse(page(n_lines=14, top=500, H=2200))
    assert not a['top_open'] and not a['bottom_open']
    assert L.warnings_for(a) == []


def test_a_neighbouring_story_below_is_excluded_from_the_crop():
    """Measured on the real captures: the next article's headline sits in
    frame, separated by a gap of 1.0-3.8 pitch. That gap is what removes
    it."""
    img = page(n_lines=40, gap_at=24, top=200, H=2600)
    a = L.analyse(img)
    assert a['n_blocks'] == 2
    _x1, y1, _x2, y2 = a['crop']
    assert (y2 - y1) < 0.75 * img.shape[0]


def test_current_block_is_the_one_at_frame_centre():
    a = L.analyse(page(n_lines=40, gap_at=6, top=150, H=2600))
    blk = a['block']
    assert blk[0] <= a['upright'].shape[0] // 2 <= blk[1]


def test_a_blank_frame_does_not_crash():
    a = L.analyse(np.full((900, 700, 3), 250, np.uint8))
    assert a['applicable'] is False


# ---- deskew --------------------------------------------------------------
def _tilt(img, deg):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), borderValue=(245, 245, 245))


@pytest.mark.parametrize('deg', [1.5, -1.5, 3.0, 0.0])
def test_skew_is_measured_with_the_right_sign_in_both_directions(deg):
    """BOTH signs, deliberately.

    The first version used cv2.minAreaRect and returned angles of the
    OPPOSITE SIGN on Ishara's OpenCV build to the ones it returned here --
    same nine frames, same code. OpenCV moved the angle range minAreaRect
    reports (from [-90, 0) to (0, 90]), so the `w < h` rule picked a
    different family of rectangles. On one of the two machines deskew was
    doubling the tilt instead of removing it, and nothing failed.

    A one-sided test would not have caught that, which is why this one runs
    both ways: deskew_deg is the angle fed BACK to getRotationMatrix2D, so
    it must be the negative of the tilt applied here.
    """
    a = L.analyse(_tilt(page(cols=3), deg))
    assert abs(a['deskew_deg'] + deg) < 0.3, a['deskew_deg']
    assert a['n_columns'] == 3


def test_the_frame_really_is_levelled_not_just_measured():
    """Version-independent property: re-measuring the corrected frame must
    return ~0. Measured on the nine real captures: 0.0-0.1 deg residual."""
    a = L.analyse(_tilt(page(cols=3), 2.0))
    assert abs(L.deskew_angle(a['upright'])) <= 0.15


# ---- joining panned frames ----------------------------------------------
A = ['මුල් පේළිය', 'දෙවන පේළිය', 'තුන්වන පේළිය', 'සිව්වන පේළිය']
B = ['තුන්වන පේළිය', 'සිව්වන පේළිය', 'පස්වන පේළිය', 'හයවන පේළිය']


def test_overlap_finds_the_repeated_lines():
    assert L.overlap(A, B) == 2


def test_join_does_not_duplicate_the_overlap():
    lines, seams = L.join([A, B])
    assert lines == A + B[2:] and seams == [2]


def test_overlap_tolerates_ocr_differences():
    assert L.overlap(A, ['තුන්වන පෙළිය', 'සිව්වන පේළිය', 'පස්වන පේළිය']) == 2


def test_unrelated_frames_are_not_joined_silently():
    _l, seams = L.join([A, ['වෙනස්', 'කිසිසේත්', 'නොගැලපේ']])
    assert seams == [0]


def test_a_single_matching_line_is_not_enough():
    assert L.overlap(['ක', 'ඛ', 'ග'], ['ග', 'ඝ', 'ඞ']) == 0


def test_the_no_gutter_limit_sits_between_the_two_measured_populations():
    """The second gate, added when CLOSEUP_MIN_P75 dropped 28 -> 20.

    Corpus full pages measure glyph_p75 18-24 and now overlap the close-up
    gate, and their pitch is normal so the pitch check does not catch them.
    What separates them is that no gutter is ever found, and the physical
    signature of that is a text band far wider than a column can be: a
    newspaper column is a roughly fixed number of characters across, so its
    pixel width scales with glyph height.

    Measured, widest band / median glyph height:
        phone close-ups        27 - 59
        corpus dinamina p14        131
        corpus lankadeepa p20      272

    Pinning the limit between the two populations, rather than pinning the
    number, means tightening it onto real captures fails here.
    """
    assert L.COL_MAX_GLYPHS > 59 * 1.2, (
        'the limit has come down onto real close-up captures (widest measured '
        '59 glyph-heights) - they will start being refused')
    assert L.COL_MAX_GLYPHS < 131 * 0.9, (
        'the limit has risen past the corpus full pages it exists to catch')


def test_a_real_closeup_is_not_caught_by_the_gutter_gate():
    """Measured: phone captures put the widest band at 0.33-0.46 of the
    frame, well under the 0.75 limit."""
    a = L.analyse(page(cols=3))
    assert a['applicable'] is True
    assert (max(b[1] - b[0] for b in a['columns'])
            / a['median_glyph_h']) < L.COL_MAX_GLYPHS

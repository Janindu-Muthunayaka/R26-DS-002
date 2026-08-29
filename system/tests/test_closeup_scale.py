"""Adaptive downscaling for the close-up path.

The bug this replaces: CLOSEUP_OCR_SCALE was a FIXED 0.40, chosen on a frame
whose glyph_p75 was 33. Applied to a glyph_p75 22 frame it gives Tesseract
8.8 px, under the ~11 px at which diacritics disappear, and measured mT5 CER
0.2193 against 0.0760 for the same frame at 13.2 px.
"""
import pytest

from core.config import (CLOSEUP_TARGET_GLYPH, CLOSEUP_MIN_GLYPH_PX,
                         OCR_SCALE_MAX, CLOSEUP_MIN_P75)
from core.imaging import closeup_scale


@pytest.mark.parametrize('p75', [20, 22, 25, 28, 33, 38, 45])
def test_effective_glyph_never_falls_below_the_floor(p75):
    """THE POINT OF THE WHOLE CHANGE. Whatever the distance, what reaches
    Tesseract stays above the height at which diacritics survive."""
    assert p75 * closeup_scale(p75) >= CLOSEUP_MIN_GLYPH_PX - 1e-6


@pytest.mark.parametrize('p75', [22, 33, 38])
def test_it_hits_the_target_when_the_target_is_reachable(p75):
    assert abs(p75 * closeup_scale(p75) - CLOSEUP_TARGET_GLYPH) < 0.01


def test_it_never_upscales():
    """2x measured CER 0.336, 3x 0.659, against 0.076 at the optimum."""
    for p75 in (4, 8, 11, 14, 15, 16):
        assert closeup_scale(p75) <= OCR_SCALE_MAX


def test_a_frame_at_the_gate_is_not_scaled_under_the_floor():
    """CLOSEUP_MIN_P75 is the smallest glyph the path accepts; the scaler has
    to still leave something readable at exactly that size."""
    assert CLOSEUP_MIN_P75 * closeup_scale(CLOSEUP_MIN_P75) >= CLOSEUP_MIN_GLYPH_PX


def test_the_old_fixed_factor_would_have_failed_that():
    """Documents the defect in the constant this replaces, so a revert is
    visible rather than silent."""
    assert 22 * 0.40 < CLOSEUP_MIN_GLYPH_PX


def test_undecidable_glyph_height_does_not_scale():
    assert closeup_scale(None) == 1.0
    assert closeup_scale(0) == 1.0


# ---- the capture gate ------------------------------------------------------
def test_the_whole_article_framing_is_not_told_to_move_closer():
    """Measured: the whole article fits at glyph_p75 22 and reads BETTER
    there (mT5 CER 0.0497) than at the close framing (0.0570). Warning
    'closer' at 22 would push the user back to a framing that clips a column
    off every capture - and a blind listener cannot see that the advice is
    wrong."""
    from core.imaging import guidance_verdict
    from core.config import CAPTURE_WARN_BELOW_P75, CLOSEUP_MIN_P75
    for p75 in (20, 22, 25, 28, 33, 38):
        v, msg = guidance_verdict(p75)
        assert v == 'ok', f'p75 {p75} -> {v}: {msg}'
    assert CAPTURE_WARN_BELOW_P75 <= CLOSEUP_MIN_P75, (
        'the guidance gate now warns above the framing the close-up path '
        'accepts - the user would be told to move away from it')


def test_a_genuinely_too_far_frame_is_still_warned_about():
    from core.imaging import guidance_verdict
    assert guidance_verdict(18)[0] == 'warn'
    assert guidance_verdict(10)[0] == 'reject'
    assert guidance_verdict(None)[0] == 'unknown'


def test_the_page_gate_still_answers_at_25_and_the_phone_gate_at_20():
    """The regression that broke two tests on 24 Aug 2026: capture_verdict()
    was switched to the phone threshold, so corpus pages at p75 18-24 that
    the diagnostics CSV calls MARGINAL came back 'ok' and the 168-page
    agreement fell from 98.2% to 29.2%. Chapter 4 cites that agreement.

    These two must disagree in exactly this band, or one of them has been
    pointed at the other one's question."""
    from core.imaging import capture_verdict, guidance_verdict
    from core.config import CAPTURE_MIN_GLYPH_P75
    assert CAPTURE_MIN_GLYPH_P75 == 25.0
    for p75 in (20.0, 22.0, 24.9):
        assert capture_verdict(p75)[0] == 'warn', f'page gate at {p75}'
        assert guidance_verdict(p75)[0] == 'ok', f'phone gate at {p75}'
    assert capture_verdict(25.0)[0] == 'ok'

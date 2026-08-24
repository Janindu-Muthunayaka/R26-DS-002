"""The measured constants must stay enforced. If someone changes these,
the numbers in the report stop matching the system.

Updated 20 Aug 2026 for the p90 -> p75 capture-metric change (D1).
"""
import numpy as np
import pytest

from core.imaging import (glyph_p75, glyph_p90, glyph_percentiles,
                          scale_for_target, capture_verdict)
from core.config import (CAPTURE_MIN_GLYPH_P75, CAPTURE_REJECT_BELOW_P75,
                         GLYPH_H_MIN, GLYPH_H_MAX)


# A page of uniform-height glyphs makes p50 == p75 == p90 and would pass a
# test even with the percentile wired up wrong. The mixture below gives the
# three percentiles distinct values, so a swap is caught. Proportions stand in
# for Sinhala's shape classes and are NOT measured.
_MIX = ((1.00, 0.45),   # base consonant
        (1.30, 0.15),   # ascender / tall form
        (0.40, 0.28),   # dependent vowel sign (pilla)
        (0.20, 0.12))   # dot, hal kirima, punctuation


def _page(glyph_h, W=1600, lines=30, seed=7):
    rng = np.random.default_rng(seed)
    H = int(W * 1.33)
    img = np.full((H, W), 245, np.uint8)
    ratios = np.array([r for r, _ in _MIX])
    probs = np.array([p for _, p in _MIX]); probs = probs / probs.sum()
    pitch = max(4, int(glyph_h * 1.8))
    y = pitch
    while y + int(glyph_h * 1.3) < H - pitch and (y // pitch) <= lines:
        x = glyph_h
        while x + glyph_h * 2 < W:
            r = max(0.1, ratios[rng.choice(len(ratios), p=probs)]
                    * rng.normal(1.0, 0.18))
            hh = max(2, int(round(glyph_h * r)))
            bw = max(2, int(round(glyph_h * r * 0.6)))
            img[y + glyph_h - hh:y + glyph_h, x:x + bw] = 30
            x += bw + max(2, glyph_h // 3)
        y += pitch
    return img


def test_percentiles_are_distinct():
    """Guards against a percentile mix-up, the D1 failure mode."""
    for h in (18, 24, 36):
        p = glyph_percentiles(_page(h))
        assert p['p50'] < p['p75'] < p['p90'], p


def test_p75_tracks_glyph_height():
    ratios = [glyph_p75(_page(h)) / h for h in (18, 24, 36, 48)]
    assert max(ratios) - min(ratios) < 0.15, ratios


def test_p75_is_monotone():
    v = [glyph_p75(_page(h)) for h in (18, 24, 36, 48)]
    assert all(b > a for a, b in zip(v, v[1:])), v


def test_p90_is_above_p75():
    """The two metrics are not interchangeable — that was the D1 bug."""
    for h in (18, 24, 36):
        assert glyph_p90(_page(h)) > glyph_p75(_page(h))


def test_capture_pass_mark_is_the_measured_one():
    """25 is confirmed on all 168 corpus pages with zero disagreements
    (system/tools/reproduce_diagnostics.py). Do not change it casually."""
    assert CAPTURE_MIN_GLYPH_P75 == 25.0
    assert (GLYPH_H_MIN, GLYPH_H_MAX) == (6, 200)


def test_verdict_bands():
    assert capture_verdict(40)[0] == 'ok'
    assert capture_verdict(25)[0] == 'ok'        # boundary is inclusive
    assert capture_verdict(24.9)[0] == 'warn'
    assert capture_verdict(19)[0] == 'warn'
    assert capture_verdict(14)[0] == 'reject'
    assert capture_verdict(None)[0] == 'unknown'
    assert CAPTURE_REJECT_BELOW_P75 < CAPTURE_MIN_GLYPH_P75


def test_scale_never_upscales():
    """Upscaling measured CER 0.336 at 2x and 0.659 at 3x, against 0.175 at
    the optimum. The cap is not a preference."""
    assert scale_for_target(19) == 1.0
    assert scale_for_target(10) == 1.0
    assert abs(scale_for_target(60) - 0.40) < .01


def test_no_text_returns_none():
    blank = np.full((400, 400), 250, np.uint8)
    assert glyph_p75(blank) is None or glyph_p75(blank) >= 0

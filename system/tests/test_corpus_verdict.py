"""Ties core.imaging to measured data, not to synthetic bars.

Synthetic pages prove the estimator is well-behaved. They cannot prove it
agrees with layout/page_diagnostics.csv, which is the file that defines the
project's capture threshold. This test checks that agreement on real pages and
fails if someone changes the filter, the percentile or the threshold.

Skips cleanly when the corpus is not present, so CI without the 168 pages
still runs the rest of the suite.

Measured 20 Aug 2026 over all 168 pages: 165 verdicts agree (98.2%).
This test samples a subset for speed and asserts a floor below that.
"""
import csv
import os
from pathlib import Path

import cv2
import pytest

from core.imaging import glyph_p75, capture_verdict
from core import config

SAMPLE_EVERY = 7          # 168 rows -> 24 pages, a few seconds
MIN_AGREEMENT = 0.90      # measured 98.2% on the full set; floor allows for
                          # sampling variation without hiding a real break


def _corpus_root() -> Path:
    return Path(os.getenv('SINHALA_ROOT', str(config.PROJECT_ROOT)))


def _rows():
    csv_path = _corpus_root() / 'layout' / 'page_diagnostics.csv'
    if not csv_path.exists():
        pytest.skip(f'corpus not present at {csv_path}')
    with open(csv_path, newline='', encoding='utf-8') as fh:
        return [r for r in csv.DictReader(fh) if r.get('file')]


def _image_for(stem: str, mode: str):
    root = _corpus_root() / 'layout'
    dirs = (['raw_halfpages', 'raw_pages'] if mode == 'half'
            else ['raw_pages', 'raw_halfpages'])
    for d in dirs:
        for ext in ('.jpg', '.jpeg', '.png'):
            p = root / d / (stem + ext)
            if p.exists():
                return p
    return None


def test_verdict_matches_page_diagnostics():
    rows = _rows()[::SAMPLE_EVERY]
    checked, agreed, misses = 0, 0, []
    for r in rows:
        p = _image_for(r['file'], r.get('mode', ''))
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        checked += 1
        mine = capture_verdict(glyph_p75(img))[0] == 'ok'
        theirs = r['resolution'] == 'OK'
        if mine == theirs:
            agreed += 1
        else:
            misses.append(f"{r['file']}: csv p75={r['glyph_p75']} "
                          f"{r['resolution']}, got ok={mine}")

    if checked == 0:
        pytest.skip('corpus images not found')
    rate = agreed / checked
    assert rate >= MIN_AGREEMENT, (
        f'verdict agreement {rate:.1%} over {checked} pages, '
        f'below the {MIN_AGREEMENT:.0%} floor.\n' + '\n'.join(misses[:10]))


def test_the_pass_mark_actually_separates_the_corpus():
    """The corpus is under-resolved — that is the project's finding. If a
    change made most pages pass, the threshold has drifted, not the corpus."""
    rows = _rows()
    ok = sum(1 for r in rows if float(r['glyph_p75']) >= 25.0)
    assert len(rows) == 168, f'corpus is {len(rows)} rows, expected 168'
    assert ok == 7, f'{ok} pages marked OK in the CSV, expected 7'
    assert ok / len(rows) < 0.10

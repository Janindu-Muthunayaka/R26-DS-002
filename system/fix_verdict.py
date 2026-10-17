"""Fix the two failing tests. Run from E:\\RP\\R26-DS-002\\system.

    python fix_verdict.py

Refuses to touch anything unless it finds exactly the text it expects, so a
half-applied edit is impossible. Writes .bak5 backups.
"""
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent

OLD_FN = '''def capture_verdict(p75):
    """Accept/reject a captured frame. Argument is glyph_p75, NOT p90.

    Warns below CAPTURE_WARN_BELOW_P75 (20), not below CAPTURE_MIN_GLYPH_P75
    (25). The two answer different questions - see core/config.py. Measured:
    the framing that holds a whole article sits at glyph_p75 22 and reads
    BETTER than the close one, so "closer" at 22 is advice in the wrong
    direction, and the listener cannot see that it is wrong.
    """
    if not p75:
        return 'unknown', 'no text found'
    if p75 < CAPTURE_REJECT_BELOW_P75:
        return 'reject', (f'glyph {p75:.0f}px (need >='
                          f'{CAPTURE_WARN_BELOW_P75:.0f}) \u2014 much closer')
    if p75 < CAPTURE_WARN_BELOW_P75:
        return 'warn', (f'glyph {p75:.0f}px (want >='
                        f'{CAPTURE_WARN_BELOW_P75:.0f}) \u2014 closer')
    return 'ok', \'\''''

NEW_FN = '''def capture_verdict(p75, warn_below=CAPTURE_MIN_GLYPH_P75):
    """Is this page good enough to OCR at all? Argument is glyph_p75, NOT p90.

    DEFAULT THRESHOLD IS 25 AND MUST STAY 25. With the default this function
    reproduces the corpus diagnostics verdict on 168 pages (165/168, 98.2%)
    and Chapter 4 cites that number. Changing the default silently changes a
    thesis result - it did, on 24 Aug 2026, and two tests caught it.

    The phone path asks a DIFFERENT question and must not call this one with
    the default. Use guidance_verdict() below.
    """
    if not p75:
        return 'unknown', 'no text found'
    if p75 < CAPTURE_REJECT_BELOW_P75:
        return 'reject', (f'glyph {p75:.0f}px (need >='
                          f'{warn_below:.0f}) \u2014 much closer')
    if p75 < warn_below:
        return 'warn', (f'glyph {p75:.0f}px (want >='
                        f'{warn_below:.0f}) \u2014 closer')
    return 'ok', ''


def guidance_verdict(p75):
    """Should the USER move? The phone path's question, not the page gate's.

    Warns below CAPTURE_WARN_BELOW_P75 (20), not below CAPTURE_MIN_GLYPH_P75
    (25). Measured 24 Aug 2026: the framing that holds a whole article sits at
    glyph_p75 22 and reads BETTER than the close one (mT5 CER 0.0497 against
    0.0570), so "closer" at 22 is advice in the wrong direction - and the
    listener cannot see that it is wrong.

    Same lesson as p75/p90: two questions, two thresholds, two names.
    """
    return capture_verdict(p75, warn_below=CAPTURE_WARN_BELOW_P75)'''

OLD_TEST = '''    from core.imaging import capture_verdict
    from core.config import CAPTURE_WARN_BELOW_P75, CLOSEUP_MIN_P75
    for p75 in (20, 22, 25, 28, 33, 38):
        v, msg = capture_verdict(p75)
        assert v == 'ok', f'p75 {p75} -> {v}: {msg}'
    assert CAPTURE_WARN_BELOW_P75 <= CLOSEUP_MIN_P75, (
        'the capture gate now warns above the framing the close-up path '
        'accepts - the user would be told to move away from it')


def test_a_genuinely_too_far_frame_is_still_warned_about():
    from core.imaging import capture_verdict
    assert capture_verdict(18)[0] == 'warn'
    assert capture_verdict(10)[0] == 'reject'
    assert capture_verdict(None)[0] == 'unknown\''''

NEW_TEST = '''    from core.imaging import guidance_verdict
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
    assert capture_verdict(25.0)[0] == 'ok\''''

EDITS = [
    (ROOT / 'core' / 'imaging.py', OLD_FN, NEW_FN),
    (ROOT / 'tests' / 'test_closeup_scale.py', OLD_TEST, NEW_TEST),
]


def main():
    plan = []
    for path, old, new in EDITS:
        if not path.exists():
            sys.exit(f'NOT FOUND: {path}\nRun this from the system folder.')
        txt = path.read_text(encoding='utf-8')
        n = txt.count(old)
        if n != 1:
            sys.exit(f'REFUSING: {path.name} contains the expected block '
                     f'{n} times, not 1. Nothing written. The file has been '
                     f'edited since I last saw it - send it to me.')
        plan.append((path, txt, txt.replace(old, new)))

    for path, before, after in plan:
        path.with_suffix(path.suffix + '.bak5').write_text(before,
                                                           encoding='utf-8')
        path.write_text(after, encoding='utf-8')
        print(f'patched {path.relative_to(ROOT)}  (backup .bak5)')

    print('\nNow run:  python -m pytest tests -q')


if __name__ == '__main__':
    main()

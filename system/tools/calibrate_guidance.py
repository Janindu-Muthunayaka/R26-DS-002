"""Fit the app's on-screen glyph estimate against the server's glyph_p75, and
derive the READY band from the fit instead of extrapolating a constant.

WHY THIS EXISTS
---------------
Guidance.kt states its thresholds "in base glyph pixels (p75)", but the app's
number and the server's number are not the same quantity. The validation table
in Android_Capture_Guidance_Calibration.md sec 5 measures both: the app said
26-29 where the captures measured 29-34. It under-reads by 4-5 px.

PITCH_PER_GLYPH = 1.80 was fitted at p75 = 25 and validated over an app range
of 26-29 - a three-pixel window. Moving the READY band down to the framing that
holds a whole article (measured p75 22) asks that constant to work outside
every range it was ever checked in, and the two calibrations in the document
disagree by 5 px at that point - wider than the band itself.

So: measure, do not extrapolate.

WHERE THE DATA COMES FROM - IT ALREADY EXISTS
---------------------------------------------
MainActivity.captureBurst() stamps the estimate and sharpness at the instant of
firing into every filename:

    burst_20260826_142530_g27_s1603_1.jpg
                          ^^^         app estimate, rounded
                              ^^^^    Laplacian variance

The server renames uploads to f0/f1/f2.jpg, so the pairing survives only on the
PHONE. Copy the app's own directory off the device and point this at it.

USAGE
-----
    python tools\\calibrate_guidance.py <folder> [<folder> ...]
    python tools\\calibrate_guidance.py <folder> --band 22 26
    python tools\\calibrate_guidance.py --self-test

The fit is reported with its residual spread and the app range it actually
covers. If the requested band falls outside that range the tool says so and
refuses to pretend the answer is measured.
"""
import argparse
import math
import time
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.imaging import glyph_p75, imread_upright          # noqa: E402

# Matches BOTH names the stamp can arrive under:
#   burst_20260826_142530_g27_s1603_1.jpg   straight off the phone
#   f0_g27_s1603.jpg                        after the server carried it through
STAMP = re.compile(r'_g(\d+)_s(\d+)(?=[_.])', re.I)
EXTS = {'.jpg', '.jpeg', '.png'}

# Measured 24 Aug 2026: the layout method stops working below glyph_p75 20
# (= CLOSEUP_MIN_P75), the whole article fits from 22, and the four-column
# ceiling closes at about 26. See Article_Reading_Fixed.md.
DEFAULT_BAND = (20.0, 26.0)

# g0 DOES NOT MEAN "the app estimated zero". MainActivity.captureBurst() writes
#     val g = lastGlyph?.let { Math.round(it) } ?: 0
# so 0 is the NULL case - the pitch estimate had failed its regularity test at
# the instant the shutter fired, usually motion blur. Measured 26 Aug 2026 on
# 30 frames: including the six g0 frames gave slope 0.725, r 0.792, sd 8.51;
# excluding them gave slope 1.088, r 0.978, sd 2.66 on the same data. Feeding a
# null-marker to a fitter as if it were a measurement is a scoring bug, and it
# is the third one this project has had.
NULL_ESTIMATE = 0.0


def collect(folders):
    """-> list of (app_estimate, sharpness_stamp, measured_p75, path)."""
    rows = []
    for folder in folders:
        root = pathlib.Path(folder)
        if not root.exists():
            sys.exit(f'not found: {root}')
        files = sorted(p for p in root.rglob('*')
                       if p.suffix.lower() in EXTS)
        if not files:
            print(f'  {root}: no images', file=sys.stderr)
        for p in files:
            m = STAMP.search(p.name)
            if not m:
                continue
            img = imread_upright(p)
            if img is None:
                print(f'  unreadable: {p.name}', file=sys.stderr)
                continue
            t = glyph_p75(img)
            if not t:
                print(f'  no text found: {p.name}', file=sys.stderr)
                continue
            rows.append((float(m.group(1)), float(m.group(2)), float(t), p))
    return rows


def screen(rows, min_sharp, since_hours=None, now=None):
    """Drop what is not a measurement, and SAY SO. Never silently.

    --since-hours exists because a build change splits the data into two
    populations that must never be fitted together. Measured 26 Aug 2026:
    seven pre-correction bursts fit 1.075x +2.84; twenty post-correction
    bursts sit on the identity; fitting all 27 together gave 1.231x -5.22,
    which describes neither build. The server writes each capture as it
    arrives, so file mtime is capture time.
    """
    kept, dropped = [], []
    cutoff = None
    if since_hours is not None:
        # From NOW, not from the newest capture. Measured 26 Aug 2026: basing
        # it on the newest capture kept all 96 frames, because the pre- and
        # post-rebuild sessions were hours apart on the SAME DAY - and the two
        # builds got fitted together regardless, which was the whole thing this
        # flag existed to prevent.
        base = now if now is not None else time.time()
        cutoff = base - since_hours * 3600
    for r in rows:
        if r[0] <= NULL_ESTIMATE:
            dropped.append((r, 'g0 - the app had NO estimate when it fired'))
        elif r[1] < min_sharp:
            dropped.append((r, f'sharpness {r[1]:.0f} < {min_sharp:.0f}'))
        elif cutoff is not None and r[3].stat().st_mtime < cutoff:
            dropped.append((r, f'older than {since_hours:g} h before the '
                               f'newest capture'))
        else:
            kept.append(r)
    if dropped:
        print(f'\ndropped {len(dropped)} of {len(rows)} frames:')
        for r, why in dropped:
            print(f'  {r[3].name:38} p75 {r[2]:5.1f}   {why}')
    return kept


def _normal_mass(lo, hi, mu, sd):
    """P(lo < X < hi) for X ~ N(mu, sd). math.erf, so no scipy dependency."""
    if sd <= 0:
        return 1.0 if lo <= mu <= hi else 0.0
    cdf = lambda x: 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))
    return cdf(hi) - cdf(lo)


def fit(app, true):
    """Least squares true = slope*app + intercept, with honest error bars."""
    n = len(app)
    if n < 3:
        return None
    slope, intercept = np.polyfit(app, true, 1)
    pred = slope * app + intercept
    resid = true - pred
    dof = max(n - 2, 1)
    sd = float(np.sqrt((resid ** 2).sum() / dof))
    r = float(np.corrcoef(app, true)[0, 1]) if app.std() > 0 else float('nan')
    return dict(slope=float(slope), intercept=float(intercept),
                r=r, sd=sd, n=n,
                app_lo=float(app.min()), app_hi=float(app.max()),
                true_lo=float(true.min()), true_hi=float(true.max()))


def bursts(rows):
    """Group frames into the bursts they were taken in, and reduce each to its
    median glyph_p75.

    THE BURST IS THE UNIT, not the frame. The shutter fires ONCE per burst, on
    MainActivity.smoothed() - a median over a ring buffer - and the pipeline
    then votes the three uploaded frames against each other. So the question
    the band has to answer is "where does a BURST land", and fitting on frames
    double-counts one decision three times and reports a scatter that includes
    within-burst noise the voter already removes.

    Measured 26 Aug 2026 on the same 30 frames: per frame sd 2.32 px, r 0.983;
    per burst sd 1.85 px, r 0.9916.

    Every frame of a burst carries the same stamp, so (app, sharpness) is the
    burst key.
    """
    g = {}
    for a, s, t, p in rows:
        g.setdefault((a, s), []).append((t, p))
    out = []
    for (a, s), fr in sorted(g.items()):
        vals = np.array([t for t, _ in fr])
        out.append((a, s, float(np.median(vals)), fr[0][1],
                    float(vals.max() - vals.min()), len(vals)))
    return out


def _drift_note(f, off, mo, ci, n):
    """What to do about a non-zero offset, once the app is already corrected."""
    if n > 1 and abs(mo) > ci:
        print(f'\n  ACTION: the offset is {mo:+.2f} px and its interval '
              f'excludes zero. Change\n    ESTIMATE_INTERCEPT by {mo:+.2f} in '
              f'MainActivity.kt, rebuild, and re-run this.')
    else:
        print(f'\n  No action. The app and the server agree to within '
              f'{max(abs(mo - ci), abs(mo + ci)):.1f} px.')


def report(rows, band, max_spread=None):
    app = np.array([r[0] for r in rows])
    sharp = np.array([r[1] for r in rows])
    true = np.array([r[2] for r in rows])

    print(f'\n{len(rows)} frames\n')
    print(f'{"app":>5} {"sharp":>7} {"measured p75":>13}  file')
    print('-' * 62)
    for a, s, t, p in sorted(rows, key=lambda r: r[0]):
        print(f'{a:>5.0f} {s:>7.0f} {t:>13.1f}  {p.name}')

    bs = bursts(rows)
    if max_spread is not None:
        wide = [b for b in bs if b[4] > max_spread]
        bs = [b for b in bs if b[4] <= max_spread]
        if wide:
            print(f'\ndropped {len(wide)} bursts whose three frames of ONE '
                  f'static scene disagreed by more\nthan {max_spread:g} px - '
                  f'the truth metric is not self-consistent there:')
            for a, s, med, _p, sp, n in wide:
                print(f'  app {a:>3.0f}  sharp {s:>5.0f}  median p75 '
                      f'{med:>5.1f}  spread {sp:>4.1f} px')
    print(f'\n{len(bs)} bursts  (the shutter fires once per burst, so this is '
          f'the unit the band has to satisfy)')
    print(f'\n{"app":>5} {"sharp":>7} {"median p75":>11} {"spread":>7} {"n":>3}')
    print('-' * 40)
    for a, s, med, _p, spread, n in bs:
        print(f'{a:>5.0f} {s:>7.0f} {med:>11.1f} {spread:>7.1f} {n:>3}')

    frame_fit = fit(app, true)
    f = fit(np.array([b[0] for b in bs]), np.array([b[2] for b in bs]))
    if f is None or frame_fit is None:
        sys.exit('\nneed at least 3 bursts to fit anything.')

    print(f'\n  per frame : p75 = {frame_fit["slope"]:.3f} x app '
          f'{frame_fit["intercept"]:+.2f}   r = {frame_fit["r"]:.4f}   '
          f'sd = {frame_fit["sd"]:.2f} px   n = {frame_fit["n"]}')
    print(f'  per burst : p75 = {f["slope"]:.3f} x app {f["intercept"]:+.2f}'
          f'   r = {f["r"]:.4f}   sd = {f["sd"]:.2f} px   n = {f["n"]}'
          f'   <- the band is set from this')
    print(f'\n  app estimates span {f["app_lo"]:.0f} to {f["app_hi"]:.0f}'
          f'   ({f["app_hi"] - f["app_lo"]:.0f} px wide)')
    print(f'  measured p75 spans {f["true_lo"]:.0f} to {f["true_hi"]:.0f}')

    if f['slope'] <= 0:
        sys.exit('\nREFUSING: the fitted slope is not positive. The app '
                 'estimate is not tracking measured glyph size at all - '
                 'something is wrong upstream, do not set thresholds from '
                 'this.')

    lo_true, hi_true = band
    lo_app = (lo_true - f['intercept']) / f['slope']
    hi_app = (hi_true - f['intercept']) / f['slope']

    # Since 26 Aug 2026 MainActivity applies ESTIMATE_SLOPE/ESTIMATE_INTERCEPT
    # before the thresholds are consulted, so the stamped estimate is ALREADY
    # in glyph_p75 units and this fit should come back as the identity. Say
    # which world we are in, or the two sets of constants get confused.
    # Decided from the OFFSET, not the slope. Post-correction captures cluster
    # in a few px of app estimate, where a fitted slope is noise - measured
    # 26 Aug 2026: 17 corrected bursts fitted 1.282x -7.64, which would have
    # been read as "not corrected" when the mean offset was -1.06 px. An
    # uncorrected build is off by about +5 px, so 3 separates them cleanly.
    _off = np.array([b[2] - b[0] for b in bs])
    corrected = abs(float(_off.mean())) < 3.0

    print(f'\nTo capture at measured p75 {lo_true:.0f}-{hi_true:.0f}:')
    print(f'\n    const val NEAR_READY  = {lo_app:.0f}f')
    print(f'    const val READY_CLOSE = {hi_app:.0f}f')

    # ---- identity check, which a slope does not give you on clustered data --
    # After the correction the captures cluster INSIDE the band, so the app
    # estimates span only a few px and a fitted slope is meaningless there.
    # The question is no longer "what is the slope" but "is the offset zero",
    # and the mean offset answers that with a proper interval however narrow
    # the cluster is.
    off = np.array([b[2] - b[0] for b in bs])
    mo, n = float(off.mean()), len(off)
    se = float(off.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    ci = 1.96 * se
    print(f'\n  IDENTITY CHECK - mean(measured p75 - app estimate) over '
          f'{n} bursts:')
    print(f'    {mo:+.2f} px   95% CI [{mo - ci:+.2f}, {mo + ci:+.2f}]')
    if n > 1 and abs(mo) <= ci:
        print(f'    zero is inside the interval - the app estimate and the '
              f'server metric agree.')
    elif n > 1:
        print(f'    zero is OUTSIDE the interval - the app is off by '
              f'{mo:+.1f} px. To correct it,\n    multiply ESTIMATE_SLOPE and '
              f'shift ESTIMATE_INTERCEPT by {mo:+.2f}.')

    if corrected:
        print(f'\n  The fit is the identity, so these captures came from a '
              f'build that already\n  applies ESTIMATE_SLOPE/ESTIMATE_INTERCEPT'
              f' - the stamp is already in p75 units.\n  The band above should '
              f'MATCH what is in Guidance.kt. Drift from 1.00x +0.00 is real '
              f'drift.')
    else:
        print(f'\n  The fit is NOT the identity ({f["slope"]:.3f}x '
              f'{f["intercept"]:+.2f}), so these captures came from a build\n  '
              f'emitting the RAW estimate. Either they predate the correction, '
              f'or the app\n  is not applying it. If MainActivity has '
              f'ESTIMATE_SLOPE in it, these are old\n  captures and the '
              f'constants above are in the wrong units - re-shoot first.')

    # Honesty gates -------------------------------------------------------
    # These all ask "is this fit good enough to DERIVE a band from?". Once the
    # app applies the correction the captures cluster inside the band by
    # design, so the range is narrow and r is low BY CONSTRUCTION - and none of
    # that matters, because the band is no longer being derived. It is being
    # checked, and the identity check above is the check. Warning here would
    # train you to ignore the warnings.
    problems = []
    if corrected:
        print('\n  The band is not being re-derived from this run - it is '
              'already set, and the\n  identity check above is the test that '
              'matters. Range and r warnings are\n  suppressed: post-'
              'correction captures cluster inside the band by design.')
        _drift_note(f, off, mo, ci, n)
        return
    if not (f['app_lo'] <= lo_app <= f['app_hi']
            and f['app_lo'] <= hi_app <= f['app_hi']):
        problems.append(
            f'the band {lo_app:.0f}-{hi_app:.0f} falls OUTSIDE the app range '
            f'measured ({f["app_lo"]:.0f}-{f["app_hi"]:.0f}). This is an '
            f'extrapolation - the exact mistake this tool exists to avoid. '
            f'Take manual bursts at those distances and re-run.')
    if f['app_hi'] - f['app_lo'] < 8:
        problems.append(
            f'the app estimates only span {f["app_hi"] - f["app_lo"]:.0f} px. '
            f'A slope fitted over a narrow window does not extrapolate - that '
            f'is how PITCH_PER_GLYPH ended up wrong. Walk further in both '
            f'directions.')
    if abs(f['r']) < 0.85:
        problems.append(f'r = {f["r"]:.2f}. Weak relationship; more bursts.')

    # Where do captures actually land? The band is only useful if holding at
    # its centre puts most bursts inside it.
    aim = (lo_true + hi_true) / 2
    inside = _normal_mass(lo_true, hi_true, aim, f['sd'])
    below = _normal_mass(-1e9, lo_true, aim, f['sd'])
    above = _normal_mass(hi_true, 1e9, aim, f['sd'])
    print(f'\n  Holding at the band centre (p75 {aim:.0f}), with the per-burst '
          f'sd of {f["sd"]:.2f} px:')
    print(f'    inside {lo_true:.0f}-{hi_true:.0f} : {inside * 100:.0f}%')
    print(f'    below  {lo_true:.0f}          : {below * 100:.0f}%  '
          f'(layout refuses the frame - the user is told, not misread)')
    print(f'    above  {hi_true:.0f}          : {above * 100:.0f}%  '
          f'(a column may be clipped - warnings_for() says which edge)')
    if inside < 0.60:
        problems.append(
            f'only {inside * 100:.0f}% of bursts would land in the band. Too '
            f'noisy for a band this tight - widen it or improve the estimator.')

    if problems:
        print('\nDO NOT USE THOSE NUMBERS YET:')
        for p in problems:
            print(f'  * {p}')
    else:
        print(f'\n  fit covers the band, sd {f["sd"]:.2f} px, r {f["r"]:.2f} '
              f'- these are measured, not extrapolated.')

    spread, current_near_ready = 0.06, 28.0
    print(f'\n  AutoShutter.maxSpreadFraction is {spread} of the estimate, so '
          f'the drift tolerance moves with the band:')
    print(f'    at the current NEAR_READY {current_near_ready:.0f} -> '
          f'+-{spread * current_near_ready:.2f} px')
    print(f'    at the new     NEAR_READY {lo_app:.0f} -> '
          f'+-{spread * lo_app:.2f} px')
    print(f'  The estimator noise does not shrink with it. If the screen reads '
          f'"drifting" and the shutter stops firing, that is why.')


def self_test():
    """The fitter must recover a line it is given. Two scoring bugs in this
    project produced plausible wrong results; a fitter is a scorer."""
    rng = np.random.default_rng(0)
    for slope, intercept in ((0.917, 6.54), (1.30, -2.0), (1.0, 0.0)):
        app = np.linspace(14, 34, 40)
        true = slope * app + intercept + rng.normal(0, 0.4, app.size)
        f = fit(app, true)
        ds, di = abs(f['slope'] - slope), abs(f['intercept'] - intercept)
        ok = ds < 0.05 and di < 1.0
        print(f'  true {slope:.3f}x{intercept:+.2f}  ->  fitted '
              f'{f["slope"]:.3f}x{f["intercept"]:+.2f}   '
              f'sd={f["sd"]:.2f}  {"OK" if ok else "FAILED"}')
        if not ok:
            sys.exit('self-test FAILED - do not trust this tool')

    # inversion must round-trip
    f = fit(np.linspace(14, 34, 40), 0.917 * np.linspace(14, 34, 40) + 6.54)
    for t in (22.0, 26.0):
        a = (t - f['intercept']) / f['slope']
        back = f['slope'] * a + f['intercept']
        assert abs(back - t) < 1e-6, 'inversion does not round-trip'
    print('  inversion round-trips')

    # The screen must remove null markers and blur, and nothing else.
    class P:
        def __init__(s, n): s.name = n
    sample = [(0.0, 8.0, 11.8, P('null_a')), (0.0, 46.0, 49.0, P('null_b')),
              (19.0, 167.0, 23.0, P('blurry')), (24.0, 1713.0, 27.0, P('good'))]
    kept = screen(sample, 600.0)
    assert [r[3].name for r in kept] == ['good'], kept
    assert len(screen(sample, 0.0)) == 2, 'min-sharp 0 must still drop g0'
    print('  screen drops g0 and blur, keeps the rest')

    # The regression this tool got wrong on 26 Aug 2026: the six g0 frames
    # pulled r from 0.978 down to 0.792 on the same 30 frames.
    print('\nself-test passed')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='*',
                    help='folders of burst_*_g<N>_s<M>_*.jpg from the phone')
    ap.add_argument('--band', nargs=2, type=float, metavar=('LO', 'HI'),
                    default=DEFAULT_BAND,
                    help='target measured glyph_p75 band (default 22 26)')
    ap.add_argument('--min-sharp', type=float, default=600.0,
                    help='drop frames below this stamped sharpness. Default '
                         '600 = AutoShutter.minSharpness, i.e. the frames the '
                         'auto-shutter would itself have refused. Pass 0 to '
                         'keep everything.')
    ap.add_argument('--since-hours', type=float, default=None, metavar='H',
                    help='keep only captures taken within H hours of the '
                         'NEWEST capture. Use after a rebuild: fitting two '
                         'builds together describes neither.')
    ap.add_argument('--max-spread', type=float, default=6.0, metavar='PX',
                    help='drop a burst whose three frames disagree by more '
                         'than this. They are one static scene, so a large '
                         'spread means the measurement is unreliable, not '
                         'that the distance changed. Default 6; 0 disables.')
    ap.add_argument('--self-test', action='store_true',
                    help='check the fitter recovers a known line, then exit')
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.folders:
        ap.error('give at least one folder, or --self-test')

    raw = collect(a.folders)
    rows = screen(raw, a.min_sharp, a.since_hours)
    if raw and not rows:
        sys.exit('\nEvery frame was dropped. Re-run with --min-sharp 0 to see '
                 'them, but a fit on frames the shutter would refuse does not '
                 'describe the frames it will take.')
    if not rows:
        sys.exit('No filenames carried a _g<N>_s<M>_ stamp.\n'
                 'These come from the phone, not from system\\work - the '
                 'server renames uploads to f0/f1/f2.jpg and the pairing is '
                 'lost. Copy the app\'s own folder off the device:\n'
                 '  Android/data/lk.sliit.r26ds002.sinhalareader/files/')
    report(rows, tuple(a.band), a.max_spread or None)


if __name__ == '__main__':
    main()

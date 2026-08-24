#!/usr/bin/env python3
"""
diagnose_article.py — is the frame ONE article, and is all of it in shot?

The close-up path crops to every text line it can find. That is not an
article: it can contain a neighbouring story, and it cannot tell whether the
article ran off the edge of the frame. Both are spoken as if they were the
whole thing.

    python tools/diagnose_article.py --root <root> --render tools\\out\\layout "F:\\App\\backend\\inbox"

Per frame: columns, which of them are clipped by a frame edge, line pitch,
vertical blocks, and the four open/closed edge verdicts. Frames that are not
close-ups are REFUSED and listed separately — a whole newspaper page has
photographs and headlines crossing columns at every height, so the projection
method used here returns one column where there are seven. That is the
article detector's problem, at the framing it was trained on.

--render writes an overlay per frame: whole columns in grey, a column clipped
by the frame edge in RED, the crop that would be sent to Tesseract in blue.
LOOK AT THOSE. Every number below depends on the columns being right, and
that is something you can check by eye in two seconds and I cannot check at
all.

Reads only. Needs cv2, numpy. No YOLO, no Tesseract, no models, no phone.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def expand(patterns):
    out = []
    for p in patterns:
        hits = glob.glob(p) or ([p] if Path(p).exists() else [])
        for h in hits:
            q = Path(h)
            if q.is_dir():
                # RECURSIVE. The real server writes one folder per capture
                # under system/work/<job>/, so pointing this at work/ has to
                # reach the frames inside. Sorted by path, which puts each
                # capture's frames together.
                out += [f for f in sorted(q.rglob('*'))
                        if f.is_file() and f.suffix.lower() in EXT]
            elif q.suffix.lower() in EXT:
                out.append(q)
    return out


def render(a, path):
    vis = a['upright'].copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]
    clipped = set(map(tuple, a['clipped_columns']))
    for b in a['columns']:
        red = tuple(b) in clipped
        cv2.rectangle(vis, (b[0], 0), (b[1], H - 1),
                      (0, 0, 255) if red else (170, 170, 170), 12 if red else 5)
    x1, y1, x2, y2 = a['crop']
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 130, 0), 10)
    bar = max(8, H // 130)
    if a['top_open']:
        cv2.rectangle(vis, (0, 0), (W - 1, bar), (0, 0, 255), -1)
    if a['bottom_open']:
        cv2.rectangle(vis, (0, H - 1 - bar), (W - 1, H - 1), (0, 0, 255), -1)
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--root', default=os.getenv('SINHALA_ROOT'))
    ap.add_argument('--render', metavar='DIR', default=None)
    ap.add_argument('--csv', metavar='FILE', default=None)
    ap.add_argument('--min-p75', type=float, default=None, dest='min_p75',
                    help='override the close-up gate. FOR MEASUREMENT ONLY: '
                         'a framing that holds a whole article can sit below '
                         'CLOSEUP_MIN_P75, and refusing to measure it is how '
                         'you conclude "impossible" from a threshold instead '
                         'of from the page. The deployed path keeps the '
                         'default.')
    a = ap.parse_args()

    from core import config
    if a.root:
        config.set_root(a.root)
    from core.imaging import imread_upright
    from layers.l3_segment import layout as L

    files = expand(a.images)
    if not files:
        print('no images matched')
        return 2
    if a.render:
        Path(a.render).mkdir(parents=True, exist_ok=True)

    # Print the stack. Three results in this project have already moved with
    # a library version -- EXIF orientation, the transformers CER, and the
    # minAreaRect skew angle that made this tool disagree with itself across
    # two machines. A table of numbers without the stack that produced them
    # is not reproducible.
    print(f'cv2 {cv2.__version__}   numpy {np.__version__}   '
          f'deskew: projection-profile search (version-independent)')
    if a.min_p75 is not None:
        print(f'close-up gate OVERRIDDEN to glyph_p75 >= {a.min_p75:g} '
              f'(deployed value is {config.CLOSEUP_MIN_P75:g}) '
              '-- measurement only')
    print()

    rows, refused = [], []
    print(f'{"frame":32} {"p75":>4} {"skew":>6} {"pitch":>5} {"cols":>4} '
          f'{"clip":>4} {"blk":>3} {"lines":>6} {"top":>7} {"bot":>7} '
          f'{"Link":>5} {"Rink":>5}  open')
    print('-' * 112)

    for f in files:
        # folder/name, because the real server writes every capture as
        # work/<job>/f0.jpg -- the bare filename would be the same for all.
        label = f'{f.parent.name}/{f.name}'[-32:]
        img = imread_upright(str(f))
        if img is None:
            print(f'{label:32} unreadable')
            continue
        an = L.analyse(img, min_p75=a.min_p75)
        if not an['applicable']:
            refused.append((label, an['reason']))
            continue
        p = an['pitch'] or 1.0
        flags = ''.join(c for c, v in zip('TBLR', (
            an['top_open'], an['bottom_open'],
            an['left_open'], an['right_open'])) if v)
        print(f'{label:32} {an["glyph_p75"]:4.0f} {an["deskew_deg"]:+6.2f} '
              f'{an["pitch"]:5.0f} {an["n_columns"]:4d} '
              f'{len(an["clipped_columns"]):4d} {an["n_blocks"]:3d} '
              f'{an["lines_in_block"]:6.1f} '
              f'{an["top_margin_px"]/p:6.2f}p {an["bottom_margin_px"]/p:6.2f}p '
              f'{an["left_edge_ink"]:5.2f} {an["right_edge_ink"]:5.2f}'
              f'  {flags or "-"}')
        widths = [b[1] - b[0] for b in an['columns']]
        rows.append({
            'file': label, 'glyph_p75': an['glyph_p75'],
            'deskew_deg': an['deskew_deg'], 'pitch': an['pitch'],
            'n_columns': an['n_columns'],
            'n_clipped': len(an['clipped_columns']),
            'left_edge_ink': an['left_edge_ink'],
            'right_edge_ink': an['right_edge_ink'],
            'col_w_median': float(np.median(widths)) if widths else 0.0,
            'n_blocks': an['n_blocks'],
            'lines_in_block': an['lines_in_block'],
            'lines_per_frame': an['lines_per_frame'],
            'top_margin_pitch': an['top_margin_px'] / p,
            'bottom_margin_pitch': an['bottom_margin_px'] / p,
            'top_open': an['top_open'], 'bottom_open': an['bottom_open'],
            'left_open': an['left_open'], 'right_open': an['right_open'],
            'warnings': ' | '.join(L.warnings_for(an)),
        })
        if a.render:
            render(an, Path(a.render) /
                   f'{f.parent.name}_{f.stem}_layout.jpg')

    if refused:
        print(f'\nREFUSED — not close-up frames ({len(refused)}):')
        for n, why in refused[:12]:
            print(f'  {n[:44]:46} {why}')
        if len(refused) > 12:
            print(f'  ... and {len(refused) - 12} more')

    if not rows:
        print('\nnothing measurable. If these were corpus pages, that is the '
              'expected answer — use the article detector on those.')
        return 1

    def q(v, ps=(50, 90, 100)):
        v = np.asarray(v, float)
        return '  '.join(f'p{x}={np.percentile(v, x):.2f}' for x in ps)

    print('\n' + '=' * 100)
    print('WHAT THIS TELLS YOU')
    print('=' * 100)

    n = len(rows)
    lat = sum(1 for r in rows if r['left_open'] or r['right_open'])
    ver = sum(1 for r in rows if r['top_open'] or r['bottom_open'])
    print(f'\n{lat}/{n} frames lose a column off the LEFT or RIGHT edge')
    print(f'{ver}/{n} frames have text running off the TOP or BOTTOM')
    print('  These are different failures and they need different fixes.')
    print('  Sideways: the article is wider than the frame — pan across, or')
    print('  back off (and pay in resolution). Vertical: pan down.')

    print(f'\npitch, px                   {q([r["pitch"] for r in rows])}')
    print(f'lines that FIT in a frame   {q([r["lines_per_frame"] for r in rows])}')
    print(f'lines in the block read     {q([r["lines_in_block"] for r in rows])}')
    print('  If "lines in the block" is well under "lines that fit", the '
          'frame is\n  not the constraint — the article really is that short, '
          'or a block\n  boundary cut it early.')

    print(f'\ntop margin / pitch          {q([r["top_margin_pitch"] for r in rows])}')
    print(f'bottom margin / pitch       {q([r["bottom_margin_pitch"] for r in rows])}')
    print(f'  EDGE_OPEN_PITCH (now {L.EDGE_OPEN_PITCH:.2f}) separates "cut off" from')
    print('  "genuinely the end". A frame holding a whole article has BOTH')
    print('  margins well above it.')

    p75 = np.asarray([r['glyph_p75'] for r in rows], float)
    pit = np.asarray([r['pitch'] for r in rows], float)
    ok = p75 > 0
    if ok.any():
        ratio = pit[ok] / p75[ok]
        print(f'\npitch / glyph_p75           {q(ratio)}')
        print(f'  The capture app assumes PITCH_PER_GLYPH = 1.80 to turn a '
              f'measured\n  pitch into a glyph estimate. This is the same '
              f'ratio measured on the\n  captured frame. If the two disagree, '
              f'the guidance is systematically\n  off — but they are NOT '
              f'measured the same way (the app works on a\n  640x480 centre '
              f'crop of the preview), so treat a difference as\n  something to '
              f'check, not as a result.')

    # ---- the distance question, answered directly ---------------------
    # Captured the same article at several distances? Then the row with the
    # LARGEST glyph_p75 that still has no ink at either edge is the answer:
    # the closest you can stand and still get the whole article across.
    print('\n' + '-' * 60)
    print('CLIPPING vs DISTANCE  (sorted by glyph_p75, closest first)')
    print('-' * 60)
    for r in sorted(rows, key=lambda r: -r['glyph_p75']):
        v = 'CLIPPED' if (r['left_open'] or r['right_open']) else 'whole'
        print(f'  p75 {r["glyph_p75"]:4.0f}   L {r["left_edge_ink"]:4.2f}  '
              f'R {r["right_edge_ink"]:4.2f}   {v:8} {r["file"]}')
    clean = [r for r in rows if not (r['left_open'] or r['right_open'])]
    if clean:
        best = max(clean, key=lambda r: r['glyph_p75'])
        print(f'\n  Closest framing that keeps the whole article: '
              f'glyph_p75 {best["glyph_p75"]:.0f}  ({best["file"]})')
        print(f'  Capture gate is {config.CAPTURE_MIN_GLYPH_P75:.0f}, close-up '
              f'threshold {config.CLOSEUP_MIN_P75:.0f}.')
        if best['glyph_p75'] >= config.CLOSEUP_MIN_P75:
            print('  -> ABOVE both. Aiming the guidance here costs no '
                  'resolution.\n     The fix is a guidance constant.')
        else:
            print('  -> BELOW the close-up threshold. Fitting the article '
                  'across the frame\n     and meeting the resolution '
                  'requirement are in conflict here.\n     Panning is the '
                  'only answer; do not lower the gate.')
    else:
        print('\n  EVERY frame is clipped. Either none was far enough back, '
              'or this\n  article does not fit at any framing above the gate. '
              'Capture one\n  clearly further away before concluding the '
              'second.')

    if a.csv:
        import csv
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {a.csv}')
    if a.render:
        print(f'overlays in {a.render} — red column = clipped by the frame '
              'edge.\nLOOK AT THEM before trusting anything above.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

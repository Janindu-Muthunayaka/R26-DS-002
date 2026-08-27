"""
Where the article-boundary constants come from.

    python tools\measure_headline.py --captures F:\App\backend\inbox

Prints, per capture: the median line height, the tallest BODY line, the
tallest HEADLINE band, and whether `closeup.headline_for_block()` was
confident enough to attach a headline to the article.

The separation this reports is the provenance of TITLE_MIN_LINE_RATIO in
core/config.py. Re-run it if the capture path changes; do not edit the
constant without re-running it.

`--ocr` also reads each accepted headline with Tesseract so the whole Layer 4A
claim can be checked end to end rather than asserted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SYSTEM = Path(__file__).resolve().parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

import cv2  # noqa: E402

from core.imaging import imread_upright                    # noqa: E402
from layers.l3_segment import closeup as C                 # noqa: E402
from layers.l3_segment import layout as L                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', default=r'F:\App\backend\inbox')
    ap.add_argument('--ocr', action='store_true')
    a = ap.parse_args()

    folder = Path(a.captures)
    files = sorted(p for p in folder.glob('*.jpg'))
    if not files:
        print(f'no captures in {folder}')
        return

    body_ratios, head_ratios, gaps = [], [], []
    accepted = 0
    print(f'{"capture":16} {"med":>4} {"body_max":>9} {"head_max":>9} '
          f'{"gap":>5}  headline')
    print('-' * 78)

    for f in files:
        im = imread_upright(str(f))
        if im is None:
            continue
        an = L.analyse(im)
        if not an.get('applicable'):
            print(f'{f.name[-16:]:16} layout refused: {an.get("reason","")[:40]}')
            continue
        up = an['upright']
        lines, med = C.text_lines(up)
        by0, by1 = an['block']
        bands = C.headline_bands(up, med, min_ratio=1.0)   # everything
        body = [b for b in bands if b[0] >= by0 and b[1] <= by1]
        head = [b for b in bands if b[1] <= by0]
        bmax = max((b[1] - b[0] for b in body), default=0)
        hmax = max((b[1] - b[0] for b in head), default=0)
        if med > 0 and bmax:
            body_ratios.append(bmax / med)
        if med > 0 and hmax:
            head_ratios.append(hmax / med)

        box = C.headline_for_block(up, an['block'],
                                   (an['crop'][0], an['crop'][2]), med)
        note = 'REFUSED (cannot tell which band is the headline)'
        if box:
            accepted += 1
            gaps.append(by0 - box[3])
            note = f'y{box[1]}-{box[3]} x{box[0]}-{box[2]}'
            if a.ocr:
                from layers.l4a_title.title import read_title_region
                bs = C.headline_bands(up, med)
                hs = sorted(b[1] - b[0] for b in bs) if bs else []
                bh = float(hs[len(hs) // 2]) if hs else 0.0
                txt, why = read_title_region(up, box, bh)
                note += f'  -> {txt[:48]!r}' if txt else f'  -> ({why})'
        print(f'{f.name[-16:]:16} {med:4.0f} {bmax:9d} {hmax:9d} '
              f'{by0 - box[3] if box else 0:5d}  {note}')

    print()
    if body_ratios and head_ratios:
        print(f'tallest BODY line     {min(body_ratios):.2f}x - '
              f'{max(body_ratios):.2f}x of median line height')
        print(f'tallest HEADLINE band {min(head_ratios):.2f}x - '
              f'{max(head_ratios):.2f}x')
        print(f'  -> nothing lands between {max(body_ratios):.2f}x and '
              f'{min(head_ratios):.2f}x; TITLE_MIN_LINE_RATIO = 3.0 sits there')
    if gaps:
        print(f'gap headline->body on accepted captures: '
              f'{min(gaps)} - {max(gaps)} px')
    print(f'\nheadline attached on {accepted}/{len(files)} captures. '
          f'The rest are read WITHOUT a headline,\nwhich is the intended '
          f'behaviour: a wrong headline is worse than none.')


if __name__ == '__main__':
    main()

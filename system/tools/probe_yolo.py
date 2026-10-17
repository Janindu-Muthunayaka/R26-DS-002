"""
Does OUR article detector help on the frames the phone actually sends?

    python tools\probe_yolo.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2

WHAT IT ANSWERS. For every real capture in `system/work`, it reports what the
detector returns and whether its box AGREES with the article the layout path
chose:

  agree     the detector found the same story.
  disagree  a different one — the failure Corrections_Register entry 1
            recorded: one confident box over the NEIGHBOURING article's
            headline, and because a box WAS returned the whole-frame fallback
            never fired.
  nothing   no box above the confidence threshold.

RESULT, 27 Aug 2026, over 70 real captures: of the 51 frames where both
produced an answer, **35 (69%) DISAGREED**, 5 partial, 11 agreed. That is why
`SEGMENT_MODE` defaults to 'off'. Re-run this if the capture range changes.

Nothing here changes behaviour. It measures.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

_SYSTEM = Path(__file__).resolve().parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))


def iou_1d(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None, help='SINHALA_ROOT (for the weights)')
    ap.add_argument('--work', default='work')
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--limit', type=int, default=200)
    a = ap.parse_args()

    if a.root:
        from core import config as _cfg
        _cfg.set_root(a.root)

    from core.config import LAYOUT_MIN_P75, YOLO_CONF, YOLO_IMGSZ, YOLO_WEIGHTS
    from core.imaging import imread_upright
    from layers.l3_segment import layout as L

    w = next((p for p in YOLO_WEIGHTS if p.exists()), None)
    if w is None:
        print('YOLO weights not found. Pass --root, or check core/config.py.')
        for p in YOLO_WEIGHTS:
            print('   ', p)
        sys.exit(2)
    print(f'weights: {w}')

    try:
        from ultralytics import YOLO
    except ImportError:
        print('ultralytics is not installed in this interpreter.')
        sys.exit(2)

    model = YOLO(str(w))
    conf = a.conf if a.conf is not None else YOLO_CONF

    files = sorted(p for p in Path(a.work).glob('*/f0*.jpg')
                   if p.stat().st_size > 100 * 1024)[:a.limit]
    if not files:
        print(f'no captures in {a.work}/*/f0*.jpg')
        sys.exit(2)

    verdicts = collections.Counter()
    print(f'\n{"capture":10} {"p75":>4} {"layout":>7} {"boxes":>5} '
          f'{"best conf":>9} {"vert IoU":>8}  verdict')
    print('-' * 72)

    for f in files:
        im = imread_upright(str(f))
        if im is None:
            continue
        lay = L.analyse(im, min_p75=LAYOUT_MIN_P75)
        r = model.predict(im, imgsz=YOLO_IMGSZ, conf=conf, verbose=False)[0]
        boxes = [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]
        confs = [float(c) for c in r.boxes.conf.cpu().numpy()] if boxes else []
        best = max(confs) if confs else 0.0
        iou = 0.0

        if not boxes:
            v = 'nothing'
        elif not lay.get('applicable'):
            v = 'no layout to compare against'
        else:
            cy1, cy2 = lay['crop'][1], lay['crop'][3]
            top = boxes[confs.index(best)]
            iou = iou_1d(cy1, cy2, top[1], top[3])
            v = ('agree' if iou >= 0.5 else
                 'DISAGREE - different story' if iou < 0.2 else 'partial')
        verdicts[v] += 1
        iou_s = f'{iou:8.2f}' if boxes and lay.get('applicable') else '       -'
        print(f'{f.parent.name:10} {lay.get("glyph_p75") or 0:4.0f} '
              f'{"ok" if lay.get("applicable") else "refused":>7} '
              f'{len(boxes):5d} {best:9.2f} {iou_s}  {v}')

    print(f'\n{len(files)} captures')
    for k, v in verdicts.most_common():
        print(f'  {v:4d} ({100 * v / len(files):3.0f}%)  {k}')
    print('\nIf DISAGREE dominates on the frames layout refuses, the detector '
          'is not a\nsafe fallback there and the honest reply is "move '
          'closer", not a confident\nwrong story.')


if __name__ == '__main__':
    main()

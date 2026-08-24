#!/usr/bin/env python3
"""
run_pipeline.py — run the real pipeline on images, from the command line.

WHY: the phone, the server and the pipeline are three things that can each be
broken. Debugging them together is how a demo dies. This runs L2->L5 directly
on files, so the question "does the pipeline produce readable Sinhala?" is
answered before any of the other two are involved.

    # your own phone captures
    python tools/run_pipeline.py ../../App/backend/inbox/burst_*.jpg

    # a corpus page, for comparison against known-good input
    python tools/run_pipeline.py <root>/layout/raw_halfpages/dinamina_20260728_p01_half.jpg

    --no-correct   skip mT5, show raw OCR only (isolates OCR from correction)
    --no-layout    skip PP-DocLayout (it is optional and not installed here)
    --json out.json
    --limit-articles N

Exit code 0 if at least one article produced text, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # so `core`, `layers`, `app` import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--root", default=os.getenv("SINHALA_ROOT"))
    ap.add_argument("--no-correct", action="store_true")
    ap.add_argument("--no-layout", action="store_true")
    ap.add_argument("--limit-articles", type=int, default=None)
    ap.add_argument("--chars", type=int, default=400,
                    help="how much text to print per article")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args()

    # set_root BEFORE importing app.pipeline — pipeline binds YOLO_WEIGHTS and
    # MT5_PLAIN at its own import time, so a later change has no effect.
    from core import config
    if a.root:
        config.set_root(a.root)
    print(f"root      {config.PROJECT_ROOT}")

    paths = []
    for pat in a.images:
        p = Path(pat)
        paths.extend(sorted(p.parent.glob(p.name)) if any(c in pat for c in "*?")
                     else [p])
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no input images matched")
        return 1
    print(f"images    {len(paths)}")
    for p in paths:
        print(f"          {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

    # ---- layer 2 first, on its own, so frame quality is visible -----------
    from layers.l2_select.select import select
    t0 = time.time()
    frames = select([str(p) for p in paths])
    print(f"\n--- L2 select ({time.time()-t0:.1f}s) ---")
    if not frames:
        print("  no usable frames — every one failed to decode")
        return 1
    for f in frames:
        print(f"  {Path(f.path).name:<44} {f.width}x{f.height}  "
              f"sharp {f.sharpness:8.1f}  p75 {f.glyph_p75 if f.glyph_p75 else -1:5.1f}  "
              f"{f.verdict}{('  — ' + f.note) if f.note else ''}")
    kept = len(frames)
    print(f"  kept {kept} of {len(paths)} for the pipeline")
    if all(f.verdict in ('reject', 'unknown') for f in frames):
        print("\n  WARNING: every frame failed the capture gate. The pipeline "
              "will still run, but poor input is the most likely explanation "
              "for poor output. See core/config.py CAPTURE_MIN_GLYPH_P75.")

    # ---- the full pipeline ------------------------------------------------
    print("\n--- loading models (this is the slow part) ---")
    t0 = time.time()
    from app.pipeline import Pipeline
    pipe = Pipeline(use_layout=not a.no_layout)
    print(f"  ready in {time.time()-t0:.1f}s")

    t0 = time.time()
    doc = pipe.run([str(p) for p in paths], correct=not a.no_correct,
                   max_articles=a.limit_articles or config.MAX_ARTICLES)
    total = time.time() - t0

    # ---- report -----------------------------------------------------------
    print(f"\n--- result ({total:.1f}s) ---")
    print(f"  timings   {doc.timings}")
    print(f"  articles  {len(doc.articles)}")
    for w in doc.warnings:
        print(f"  warning   {w}")

    produced = 0
    for art in doc.articles:
        body = (art.body or art.body_raw or '').strip()
        if body:
            produced += 1
        print(f"\n  ================ article {art.index} ================")
        print(f"  box      ({art.box.x1:.0f},{art.box.y1:.0f})-"
              f"({art.box.x2:.0f},{art.box.y2:.0f})   "
              f"{len(art.regions)} regions "
              f"({sum(1 for r in art.regions if r.label=='title')} title / "
              f"{sum(1 for r in art.regions if r.label=='text')} text)")
        print(f"  glyph    p75 {art.glyph_p75}  p90 {art.glyph_p90}  "
              f"ocr_scale {art.ocr_scale}   verdict {art.verdict}")
        if art.title.strip():
            print(f"  title    {art.title[:120]}")
        # lengths make truncation visible. mT5 stopping early looks like
        # perfectly good Sinhala that simply ends, so without this the bug
        # has no symptom at all.
        from core.textutils import sentences
        nraw, nout = len(art.body_raw), len(art.body or '')
        print(f"  length   raw {nraw} chars / {len(sentences(art.body_raw))} "
              f"sentences" + ("" if a.no_correct else
              f"   ->  mT5 {nout} chars  ({nout/nraw:.0%} of raw)"
              if nraw else ""))
        if not a.no_correct and nraw and nout < 0.6 * nraw:
            print(f"  WARNING  output is much shorter than the input — check "
                  f"for generation truncation (MT5_MAX_LENGTH)")
        print(f"  raw      {art.body_raw[:a.chars]}")
        if not a.no_correct:
            print(f"  ---")
            print(f"  mT5      {art.body[:a.chars]}")

    print("\n" + "=" * 60)
    print(f"{produced} of {len(doc.articles)} articles produced text")
    if produced == 0:
        print("\nNothing was read. Work backwards, in this order:")
        print("  1. L2 verdicts above — was the input good enough?")
        print("  2. --no-correct — is the OCR itself empty, or is mT5 eating it?")
        print("  3. articles/regions counts — did YOLO find boxes at all?")
        print("  4. tesseract --list-langs — is 'sin' actually installed?")
    if a.json_out:
        Path(a.json_out).write_text(doc.model_dump_json(indent=2),
                                    encoding="utf-8")
        print(f"\nfull document written to {a.json_out}")
    print("=" * 60)
    return 0 if produced else 1


if __name__ == "__main__":
    sys.exit(main())

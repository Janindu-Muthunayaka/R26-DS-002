#!/usr/bin/env python3
"""
reproduce_diagnostics.py -- recover the estimator behind page_diagnostics.csv

WHY THIS EXISTS
---------------
The project has two glyph metrics that disagree (p90 pass 22 in system/core,
p75 pass 25 in the CSV and the Android app). Before either is changed, the
authoritative one has to be reproducible. page_diagnostics.csv is measured data
whose generating code is not to hand, so this tool recovers it by search and
then says, out loud, how well it matched.

A harness that cannot reproduce a known result cannot be trusted on an unknown
one. If `search` cannot hit the CSV, that is the finding, and nothing
downstream should be changed on the strength of a near miss.

STAGES
------
  selftest  no page images needed. Checks (a) the estimator tracks glyph size
            on synthetic pages, (b) the derived verdict rule reproduces the
            CSV's `resolution` column exactly.
  cache     one connected-components pass per image -> .npz of bounding boxes.
            Slow (minutes). Run once.
  search    grid search over region selection / filters / percentile, scored
            against the CSV, using only the cache. Fast.
  verify    re-run one config over every page and print a full agreement report.

USAGE
-----
  python reproduce_diagnostics.py selftest --csv PATH/page_diagnostics.csv
  python reproduce_diagnostics.py cache    --root PATH/Sinhala_OCR_Correction_v2
  python reproduce_diagnostics.py search   --root PATH --csv PATH/page_diagnostics.csv
  python reproduce_diagnostics.py verify   --root PATH --csv ... --config out/best.json

Only numpy + opencv are required.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# The one rule we derive from the CSV rather than assume.
# Stated here so selftest can falsify it.
VERDICT_P75_PASS = 25.0

CACHE_NAME = "cc_cache.npz"


# ==========================================================================
# component extraction
# ==========================================================================
def components(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    """All connected components of an Otsu-binarised image.

    Returns an (N, 5) int32 array of [x, y, w, h, area]. No filtering is done
    here -- filtering is a search parameter, so it must happen downstream of
    the cache.
    """
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    bw = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return np.zeros((0, 5), np.int32)
    s = stats[1:]  # drop background
    return np.stack([s[:, cv2.CC_STAT_LEFT], s[:, cv2.CC_STAT_TOP],
                     s[:, cv2.CC_STAT_WIDTH], s[:, cv2.CC_STAT_HEIGHT],
                     s[:, cv2.CC_STAT_AREA]], axis=1).astype(np.int32)


def deskew_angle(gray: np.ndarray) -> float:
    """Small-angle deskew estimate, reported in degrees.

    Used only to check against the CSV's deskew_deg column -- a cheap,
    independent signal for whether the CSV was computed on raw or deskewed
    images. Not used by the estimator itself.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 720, 200)
    if lines is None:
        return 0.0
    angs = []
    for rho_theta in lines[:200]:
        theta = float(rho_theta[0][1])
        d = np.degrees(theta) - 90.0
        if abs(d) <= 5.0:
            angs.append(d)
    return float(np.median(angs)) if angs else 0.0


# ==========================================================================
# region selection + filtering -- the unknowns being searched
# ==========================================================================
@dataclass(frozen=True)
class Config:
    region: str = "page"      # page | tiles | centre
    grid: int = 4             # tiles: split page into grid x grid
    tiles: int = 4            # tiles: how many densest tiles to keep
    centre_frac: float = 0.6  # centre: fraction of each axis
    hmin: int = 3             # component height filter, inclusive
    hmax: int = 200
    amin: int = 0             # component area filter
    wmax_frac: float = 1.0    # drop components wider than this frac of image W
    hoff: int = 0             # +/-1 bounding-box height convention offset
    interp: str = "linear"    # np.percentile method

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _select_boxes(cc: np.ndarray, W: int, H: int, cfg: Config) -> np.ndarray:
    if cc.shape[0] == 0:
        return cc
    x, y, w, h, a = cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3], cc[:, 4]

    keep = (h >= cfg.hmin) & (h <= cfg.hmax) & (a >= cfg.amin)
    if cfg.wmax_frac < 1.0:
        keep &= w <= int(W * cfg.wmax_frac)
    cc = cc[keep]
    if cc.shape[0] == 0 or cfg.region == "page":
        return cc

    x, y, w, h = cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3]
    cx, cy = x + w // 2, y + h // 2

    if cfg.region == "centre":
        f = cfg.centre_frac
        x0, x1 = int(W * (1 - f) / 2), int(W * (1 + f) / 2)
        y0, y1 = int(H * (1 - f) / 2), int(H * (1 + f) / 2)
        return cc[(cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)]

    if cfg.region == "tiles":
        g = cfg.grid
        tx = np.clip((cx * g) // max(W, 1), 0, g - 1)
        ty = np.clip((cy * g) // max(H, 1), 0, g - 1)
        tid = ty * g + tx
        counts = np.bincount(tid, minlength=g * g)
        # densest tiles: where the body text is. Headlines and photos live in
        # sparse tiles, which is the behaviour the build record describes.
        best = np.argsort(-counts)[:cfg.tiles]
        return cc[np.isin(tid, best)]

    raise ValueError(f"unknown region mode {cfg.region!r}")


def percentiles(cc: np.ndarray, W: int, H: int, cfg: Config,
                ps=(50, 75, 90)) -> dict:
    sel = _select_boxes(cc, W, H, cfg)
    if sel.shape[0] == 0:
        return {f"p{p}": None for p in ps}
    h = sel[:, 3].astype(float) + cfg.hoff
    out = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(h, p, method=cfg.interp))
    out["n"] = int(sel.shape[0])
    return out


# ==========================================================================
# CSV
# ==========================================================================
def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("file")]
    for r in rows:
        for k in ("glyph_p50", "glyph_p75", "deskew_deg", "MP"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
        for k in ("W", "H"):
            if r.get(k) not in (None, ""):
                r[k] = int(r[k])
    return rows


PREFER = "raw"   # set from --prefer; "raw" or "deskewed"


def find_image(root: Path, row: dict) -> Path | None:
    """CSV `file` is e.g. dinamina_20260728_p01_full -> layout/raw_pages/...,
    ..._half -> layout/raw_halfpages/...  Extensions vary, so glob."""
    stem = row["file"]
    mode = row.get("mode", "")
    dirs = []
    if mode == "half":
        dirs = ["layout/raw_halfpages", "layout/raw_pages"]
    else:
        dirs = ["layout/raw_pages", "layout/raw_halfpages"]
    dirs += ["layout/deskewed", "data/raw/newspapers"]
    if PREFER == "deskewed":
        dirs = ["layout/deskewed"] + [d for d in dirs if d != "layout/deskewed"]
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for cand in itertools.chain(base.glob(stem + ".*"),
                                    base.glob(stem.replace("_full", "") + ".*"),
                                    base.glob(stem.replace("_half", "") + ".*")):
            if cand.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                return cand
    return None


# ==========================================================================
# cache
# ==========================================================================
def build_cache(root: Path, csv_path: Path, out: Path, limit: int | None,
                check_deskew: bool) -> None:
    rows = load_csv(csv_path)
    if limit:
        rows = rows[:limit]
    store, missing = {}, []
    for i, r in enumerate(rows, 1):
        p = find_image(root, r)
        if p is None:
            missing.append(r["file"])
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing.append(r["file"])
            continue
        H, Wd = img.shape[:2]
        cc = components(img)
        store[r["file"]] = cc
        store[r["file"] + "::wh"] = np.array([Wd, H], np.int32)
        if check_deskew:
            store[r["file"] + "::skew"] = np.array([deskew_angle(img)], np.float32)
        print(f"  [{i}/{len(rows)}] {r['file']}  {Wd}x{H}  {cc.shape[0]} components",
              flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    print(f"\ncached {len(store) // (3 if check_deskew else 2)} images -> {out}")
    if missing:
        print(f"MISSING {len(missing)} images, first few: {missing[:5]}")


def load_cache(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ==========================================================================
# scoring
# ==========================================================================
def score(cache: dict, rows: list[dict], cfg: Config, tol: float = 0.5) -> dict:
    n = hit50 = hit75 = hit_both = 0
    err50, err75 = [], []
    sgn50, sgn75 = [], []
    for r in rows:
        cc = cache.get(r["file"])
        if cc is None:
            continue
        W, H = cache[r["file"] + "::wh"]
        got = percentiles(cc, int(W), int(H), cfg)
        if got["p75"] is None:
            continue
        n += 1
        s50 = got["p50"] - r["glyph_p50"]
        s75 = got["p75"] - r["glyph_p75"]
        sgn50.append(s50)
        sgn75.append(s75)
        d50, d75 = abs(s50), abs(s75)
        err50.append(d50)
        err75.append(d75)
        a, b = d50 <= tol, d75 <= tol
        hit50 += a
        hit75 += b
        hit_both += a and b
    if n == 0:
        return {"n": 0, "both": 0.0}
    return {
        "n": n,
        "p50_exact": hit50 / n,
        "p75_exact": hit75 / n,
        "both": hit_both / n,
        "p50_medabs": float(np.median(err50)),
        "p75_medabs": float(np.median(err75)),
        # signed medians: a consistent non-zero value is a convention
        # difference, not noise, and is fixable rather than fatal
        "p50_medbias": float(np.median(sgn50)),
        "p75_medbias": float(np.median(sgn75)),
    }


def grid() -> list[Config]:
    out = []
    for hmin, hmax in ((3, 200), (2, 200), (4, 150), (5, 100), (3, 80)):
        for amin in (0, 8, 20):
            for wmax in (1.0, 0.25):
                for hoff in (-1, 0, 1):
                    out.append(Config(region="page", hmin=hmin, hmax=hmax,
                                      amin=amin, wmax_frac=wmax, hoff=hoff))
                    for g, t in ((3, 3), (4, 4), (4, 6), (5, 5), (6, 8)):
                        out.append(Config(region="tiles", grid=g, tiles=t,
                                          hmin=hmin, hmax=hmax, amin=amin,
                                          wmax_frac=wmax, hoff=hoff))
                    for f in (0.5, 0.6, 0.75):
                        out.append(Config(region="centre", centre_frac=f,
                                          hmin=hmin, hmax=hmax, amin=amin,
                                          wmax_frac=wmax, hoff=hoff))
    return out


# ==========================================================================
# selftest -- runnable with no page images
# ==========================================================================
#  A page of uniform-height glyphs makes p50 == p75 == p90 and would pass a
#  test even if the percentile were wired up wrong. The mixture below is what
#  gives the three percentiles distinct values, so a swap is caught.
#  Proportions are a stand-in for Sinhala's shape classes, NOT measured:
#  base consonants, taller ascenders, dependent vowel signs, small marks.
_MIX = ((1.00, 0.45),   # base consonant
        (1.30, 0.15),   # ascender / tall form
        (0.40, 0.28),   # dependent vowel sign (pilla)
        (0.20, 0.12))   # dot, hal kirima, punctuation


def _synth_page(glyph_h: int, W: int = 1600, lines: int = 30,
                seed: int = 7) -> np.ndarray:
    """Synthetic page with a realistic SPREAD of component heights."""
    rng = np.random.default_rng(seed)
    H = int(W * 1.33)
    img = np.full((H, W), 245, np.uint8)
    ratios = np.array([r for r, _ in _MIX])
    probs = np.array([p for _, p in _MIX])
    probs = probs / probs.sum()

    pitch = max(4, int(glyph_h * 1.8))
    y = pitch
    while y + int(glyph_h * 1.3) < H - pitch and (y // pitch) <= lines:
        x = glyph_h
        while x + glyph_h * 2 < W:
            # per-glyph jitter: without it every component in a class has an
            # identical height, p50 and p75 collapse onto the same value, and
            # the test can no longer tell the percentiles apart.
            r = ratios[rng.choice(len(ratios), p=probs)] * rng.normal(1.0, 0.18)
            hh = max(2, int(round(glyph_h * max(0.1, r))))
            bw = max(2, int(round(glyph_h * max(0.1, r) * 0.6)))
            img[y + glyph_h - hh:y + glyph_h, x:x + bw] = 30
            x += bw + max(2, glyph_h // 3)
        y += pitch
    return img


def selftest(csv_path: Path | None) -> int:
    fails = []
    cfg = Config(region="page", hmin=3, hmax=200)
    heights = (12, 18, 24, 36, 48)
    got = {}

    print("1. the three percentiles are distinct (a swap would be caught)")
    for h in heights:
        img = _synth_page(h)
        got[h] = percentiles(components(img), img.shape[1], img.shape[0], cfg)
        g = got[h]
        spread_ok = g["p50"] < g["p75"] < g["p90"]
        print(f"   glyph {h:>3}  ->  p50 {g['p50']:>5.1f}  p75 {g['p75']:>5.1f}"
              f"  p90 {g['p90']:>5.1f}  n={g['n']:<5} "
              f"{'ok' if spread_ok else 'FAIL: no spread'}")
        if not spread_ok:
            fails.append(f"p50<p75<p90 violated at glyph {h}")

    print("\n2. each percentile scales linearly with glyph height")
    for p in ("p50", "p75", "p90"):
        ratios = [got[h][p] / h for h in heights]
        lo, hi = min(ratios), max(ratios)
        # a linear estimator has a constant ratio to the true glyph height;
        # allow quantisation slack at the smallest sizes
        ok = (hi - lo) <= 0.15 * (sum(ratios) / len(ratios)) + 0.05
        print(f"   {p} / glyph_h  = {[round(r, 3) for r in ratios]}  "
              f"{'ok' if ok else 'FAIL: not scale-invariant'}")
        if not ok:
            fails.append(f"{p} is not linear in glyph height")

    print("\n3. p75 is monotone in glyph height")
    vals = [got[h]["p75"] for h in heights]
    if any(b <= a for a, b in zip(vals, vals[1:])):
        fails.append(f"p75 not monotone: {vals}")
        print(f"   FAIL {vals}")
    else:
        print(f"   ok  {[round(v, 1) for v in vals]}")

    if csv_path is None:
        print("\n4. verdict rule -- SKIPPED (pass --csv to check it)")
    else:
        print(f"\n4. verdict rule: resolution == OK iff glyph_p75 >= "
              f"{VERDICT_P75_PASS:g}")
        rows = load_csv(csv_path)
        bad = [r for r in rows
               if (r["glyph_p75"] >= VERDICT_P75_PASS) != (r["resolution"] == "OK")]
        ok_n = sum(1 for r in rows if r["resolution"] == "OK")
        print(f"   {len(rows)} rows, {ok_n} marked OK, {len(bad)} disagree "
              f"with the rule")
        if bad:
            fails.append(f"verdict rule mismatched on {len(bad)} rows")
            for r in bad[:5]:
                print(f"   MISMATCH {r['file']}  p75={r['glyph_p75']}  "
                      f"{r['resolution']}")
        modes = {}
        for r in rows:
            modes.setdefault(r["mode"], []).append(r["glyph_p75"])
        for m, v in sorted(modes.items()):
            print(f"   {m:>5}: n={len(v):<4} p75 min {min(v):g} "
                  f"max {max(v):g} median {float(np.median(v)):g}")

    print("\n" + ("SELFTEST FAILED: " + "; ".join(fails) if fails
                  else "SELFTEST PASSED"))
    return 1 if fails else 0


# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("selftest", "cache", "search", "verify"))
    ap.add_argument("--root", type=Path, help="Sinhala_OCR_Correction_v2 root")
    ap.add_argument("--csv", type=Path, help="page_diagnostics.csv")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--config", type=Path, help="verify: config json")
    ap.add_argument("--limit", type=int, help="cache/search: first N rows only")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--prefer", choices=("raw", "deskewed"), default="raw",
                    help="which image set the CSV was likely computed on")
    ap.add_argument("--check-deskew", action="store_true",
                    help="cache: also estimate skew, to test raw vs deskewed")
    a = ap.parse_args(argv)

    global PREFER
    PREFER = a.prefer

    if a.stage == "selftest":
        return selftest(a.csv)

    if a.csv is None:
        ap.error("--csv is required")

    if a.stage == "cache":
        if a.root is None:
            ap.error("--root is required")
        build_cache(a.root, a.csv, a.out / CACHE_NAME, a.limit, a.check_deskew)
        return 0

    cache = load_cache(a.out / CACHE_NAME)
    rows = load_csv(a.csv)
    if a.limit:
        rows = rows[:a.limit]

    if a.stage == "search":
        cfgs = grid()
        print(f"scoring {len(cfgs)} configs against {len(rows)} rows "
              f"(tol +-{a.tol})\n")
        res = []
        for i, c in enumerate(cfgs, 1):
            s = score(cache, rows, c, a.tol)
            res.append((s, c))
            if i % 25 == 0:
                print(f"  {i}/{len(cfgs)}", flush=True)
        res.sort(key=lambda t: (-t[0].get("both", 0),
                                t[0].get("p75_medabs", 9e9)))
        print("\n  both%  p50%  p75%   |p50| |p75|  bias50 bias75  config")
        for s, c in res[:12]:
            if not s["n"]:
                continue
            print(f"  {s['both']:5.1%} {s['p50_exact']:5.1%} {s['p75_exact']:5.1%}"
                  f"  {s['p50_medabs']:5.2f} {s['p75_medabs']:5.2f} "
                  f"{s['p50_medbias']:+6.1f} {s['p75_medbias']:+6.1f}  "
                  f"{c.region} g{c.grid}/t{c.tiles} f{c.centre_frac} "
                  f"h{c.hmin}-{c.hmax} a{c.amin} w{c.wmax_frac} o{c.hoff:+d}")
        best_s, best_c = res[0]
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "best.json").write_text(json.dumps(
            {"config": asdict(best_c), "score": best_s}, indent=2))
        print(f"\nbest written to {a.out/'best.json'}")
        if best_s.get("both", 0) < 0.80:
            print("\nNOT REPRODUCED. Best config agrees on "
                  f"{best_s.get('both', 0):.1%} of rows. Do not change the "
                  "project's glyph metric on the strength of this -- the CSV "
                  "was made some other way. Widen the grid or find the "
                  "generating code.")
            return 2
        return 0

    if a.stage == "verify":
        if a.config is None:
            ap.error("--config is required")
        blob = json.loads(a.config.read_text())
        cfg = Config(**blob["config"])
        s = score(cache, rows, cfg, a.tol)
        print(json.dumps(s, indent=2))
        print("\nfile,mode,csv_p50,got_p50,csv_p75,got_p75,got_p90,n")
        for r in rows:
            cc = cache.get(r["file"])
            if cc is None:
                continue
            W, H = cache[r["file"] + "::wh"]
            g = percentiles(cc, int(W), int(H), cfg)
            if g["p75"] is None:
                continue
            print(f"{r['file']},{r['mode']},{r['glyph_p50']:g},{g['p50']:.1f},"
                  f"{r['glyph_p75']:g},{g['p75']:.1f},{g['p90']:.1f},{g['n']}")
        return 0 if s.get("both", 0) >= 0.80 else 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

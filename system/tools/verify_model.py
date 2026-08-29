#!/usr/bin/env python3
"""
verify_model.py — does mT5 on THIS machine reproduce the thesis results?

Nothing in step 9 should be built on a model load that has not been checked.
This machine has transformers 5.1.0; the checkpoint was produced under the 4.x
line, and generation defaults have changed between those. If beam search
behaves differently here, every downstream number silently shifts.

The check is deliberately METRIC-INDEPENDENT. results/per_sentence_results.json
stores the actual B3 output STRING for all 217 locked test sentences, so the
strongest available test is: feed the same 217 inputs through the model and see
whether the same strings come back. That answers "is this model, in this
environment, the one that produced the thesis" without depending on how CER was
defined.

A CER comparison is also printed, but computed the SAME way on both the stored
and the regenerated outputs, so the metric definition cancels out.

    python tools/verify_model.py --root E:\\RP\\corpus\\Sinhala_OCR_Correction_v2
    python tools/verify_model.py --limit 25          # quick smoke run

Exit code 0 = verified, 1 = did not reproduce, 2 = could not run.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---- the generation settings the thesis used --------------------------------
# no_repeat_ngram_size=6 measured CER 0.0847 -> 0.0515 on identical inputs by
# suppressing generation runaway. These must match core/config.py.
NUM_BEAMS = 4
NO_REPEAT_NGRAM = 6
MAX_LENGTH = 128

PASS_EXACT = 0.95      # judgement call, not measured — see the note printed
PASS_CER_DELTA = 0.005 # CER DEGRADATION tolerated vs the stored run.
#
# Deliberately one-sided. An earlier version used abs(delta) and reported
# "NOT REPRODUCED" when the regenerated outputs were BETTER than the stored
# ones — which is a finding, not a failure. A verification that cannot tell
# improvement from regression is worse than none, because it trains you to
# ignore it.


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(gt, hyp):
    return levenshtein(gt, hyp) / max(1, len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("SINHALA_ROOT"))
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N sentences (smoke test)")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--batch", type=int, default=1,
                    help="sentences per generate() call. >1 is what "
                         "MT5_BATCH would enable; this measures whether it "
                         "changes the model's output.")
    ap.add_argument("--show", type=int, default=5,
                    help="how many mismatches to print")
    a = ap.parse_args()

    if not a.root:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core import config
        a.root = str(config.PROJECT_ROOT)
    root = Path(a.root)

    stored_path = root / "results" / "per_sentence_results.json"
    canon_path = root / "results" / "final_results.json"
    model_dir = root / "models" / "mt5_plain"
    for p in (stored_path, canon_path, model_dir):
        if not p.exists():
            print(f"cannot run: {p} not found")
            return 2

    stored = json.load(open(stored_path, encoding="utf-8"))
    canon = json.load(open(canon_path, encoding="utf-8"))
    if a.limit:
        stored = stored[:a.limit]

    print("=" * 70)
    print("mT5 REPRODUCTION CHECK")
    print("=" * 70)
    print(f"model     {model_dir}")
    print(f"sentences {len(stored)}"
          f"{'  (SUBSET — not a full verification)' if a.limit else '  (full locked test set)'}")

    # ---- load ---------------------------------------------------------------
    try:
        import torch
        from transformers import AutoTokenizer, MT5ForConditionalGeneration
        import transformers
    except Exception as e:
        print(f"cannot run: {type(e).__name__}: {e}")
        return 2
    print(f"transformers {transformers.__version__}   torch {torch.__version__}")

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdl = MT5ForConditionalGeneration.from_pretrained(str(model_dir)).to(dev).eval()
    except Exception as e:
        print(f"\nMODEL FAILED TO LOAD on transformers {transformers.__version__}")
        print(f"  {type(e).__name__}: {e}")
        print("\nThis is the transformers 4.x -> 5.x risk. If it cannot be made to "
              "load, pin the old line in a venv:\n"
              "  python -m venv .venv && .venv\\Scripts\\activate\n"
              "  pip install \"transformers<5\" sentencepiece")
        return 2
    print(f"loaded on {dev} in {time.time()-t0:.1f}s\n")

    # ---- generate -----------------------------------------------------------
    gen_kw = dict(max_length=MAX_LENGTH, num_beams=NUM_BEAMS,
                  no_repeat_ngram_size=NO_REPEAT_NGRAM, early_stopping=True)
    outs = []
    t0 = time.time()
    src = [r["ocr"] for r in stored]
    bs = max(1, a.batch)
    print(f"batch size {bs}" + ("  (measuring whether batching changes the "
                                "output)" if bs > 1 else ""))
    with torch.no_grad():
        for i in range(0, len(src), bs):
            chunk = src[i:i + bs]
            enc = tok(chunk, return_tensors="pt", truncation=True,
                      padding=(len(chunk) > 1), max_length=MAX_LENGTH).to(dev)
            try:
                g = mdl.generate(**enc, **gen_kw)
            except TypeError as e:
                print(f"generate() rejected the thesis settings: {e}")
                print("The generation API changed. Do NOT silently drop an "
                      "argument — no_repeat_ngram_size is worth 0.033 CER.")
                return 2
            outs.extend(tok.batch_decode(g, skip_special_tokens=True))
            done = min(i + bs, len(src))
            if done % 25 < bs:
                el = time.time() - t0
                print(f"  {done}/{len(src)}   {el:.0f}s elapsed, "
                      f"~{el/done*(len(src)-done):.0f}s left", flush=True)
    print(f"  generated {len(outs)} in {time.time()-t0:.0f}s\n")

    # ---- compare ------------------------------------------------------------
    exact = sum(1 for o, r in zip(outs, stored) if o == r["b3"])
    rate = exact / len(stored)

    # same CER function applied to both, so the definition cancels
    cer_mine = sum(cer(r["gt"], o) for o, r in zip(outs, stored)) / len(stored)
    cer_stored_same = sum(cer(r["gt"], r["b3"]) for r in stored) / len(stored)
    cer_stored_field = sum(r["cer_b3"] for r in stored) / len(stored)

    print("-" * 70)
    print(f"exact string match      {exact}/{len(stored)} = {rate:.1%}")
    print()
    print(f"CER, my outputs         {cer_mine:.6f}")
    print(f"CER, stored b3 strings  {cer_stored_same:.6f}   "
          f"(same function — this is the fair comparison)")
    print(f"CER, stored cer_b3      {cer_stored_field:.6f}   "
          f"(the thesis metric; differs slightly, see note)")
    if not a.limit:
        print(f"canonical b3            {canon['cer']['b3']:.6f}")
    delta = cer_mine - cer_stored_same
    print(f"\ndelta vs stored strings {delta:+.6f}   "
          f"({'BETTER' if delta < 0 else 'worse' if delta > 0 else 'identical'} "
          f"— lower CER is better)")

    if exact < len(stored) and a.show:
        print("\nfirst mismatches:")
        n = 0
        for o, r in zip(outs, stored):
            if o == r["b3"]:
                continue
            n += 1
            print(f"\n  --- #{n}  (char distance {levenshtein(o, r['b3'])}) ---")
            print(f"  in     {r['ocr'][:100]}")
            print(f"  stored {r['b3'][:100]}")
            print(f"  mine   {o[:100]}")
            if n >= a.show:
                break

    print("\n" + "=" * 70)
    worse = delta > PASS_CER_DELTA
    exact_ok = rate >= PASS_EXACT
    if worse:
        print("REGRESSION — the regenerated outputs are measurably WORSE.")
        print("Do not build on this. Most likely the transformers 4.x -> 5.x")
        print("generation change. Pin the old line in the venv and re-run:")
        print('  python -m pip install --ignore-installed "transformers<5"')
    elif exact_ok and abs(delta) <= PASS_CER_DELTA:
        print("VERIFIED — this environment reproduces the thesis model.")
    elif delta < -PASS_CER_DELTA:
        print("DIFFERENT, AND BETTER — not a failure, but do not quietly")
        print("adopt it as a result. The thesis number was measured under a")
        print("locked protocol; this is a re-run under a different library.")
        print("To make anything of it you would have to re-run the WHOLE")
        print("evaluation under that protocol, including the baselines and")
        print("the paired bootstrap. Until then the canonical number stands")
        print("and this is a reproducibility observation.")
        print("\nNEXT: run the same command with --batch 1 over the full set.")
        print("Without that control you cannot tell whether the difference")
        print("comes from batching or from the library version.")
    else:
        print("DIFFERS in wording but not measurably in CER. Beam search can")
        print("vary across library versions and batch shapes without being")
        print("wrong. Judge on the CER delta, not the string match.")
    print(f"\nNOTE: the {PASS_EXACT:.0%} exact-match bar is a judgement call, not a")
    print("measured threshold. Beam search can differ slightly across library")
    print("versions and hardware without being wrong. The CER delta against the")
    print("stored strings is the number that actually matters.")
    print("=" * 70)
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())

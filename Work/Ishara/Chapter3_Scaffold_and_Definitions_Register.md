# Chapter 3 — methodology scaffold and definitions register

**Project:** R26-DS-002 · Component 2 · IT22259134
**Compiled:** 20 August 2026
**Status:** working document. Part 1 contains decisions you must make; Part 3 is the
writing scaffold that depends on them.
**Last measurement run:** 20 August 2026 — D1 resolved, see below.

---

## 0. How to use this

Chapter 3 is not blocked on prose. It is blocked on **six definitions that are
currently ambiguous across your own documents**. Chapter 4's tables inherit every
one of them. Fix them here first, then write both chapters once.

Part 1 is the register: each open definition, where the competing versions live,
what the evidence says, and a recommendation. Part 2 is an audit of claims that
are still live in documents a marker could read. Part 3 is the chapter skeleton.

**Nothing in this document introduces a number that was not measured.** Where a
figure is needed and does not exist, it is marked `NOT MEASURED`.

---

## 1. Definitions register

Six entries. Four were already known (handoff §9.12, §9.13, §2.6, §4.1); two were
found on 20 August while reading the live documents.

---

### D1 — glyph metric: p90 or p75

**Status:** RESOLVED and APPLIED (20 August). `core/config.py`,
`core/imaging.py`, `layers/l2_select/select.py`, `core/schemas.py` and the
tests now use p75 with pass mark 25.

Verified on the deployed code, on your machine, against the live config:
**165 of 168 verdicts agree — 98.2%.** All three disagreements are boundary
cases within 3 px of the threshold (`dinamina_p64_full` csv 22, MARGINAL;
`lankadeepa_p21_half` csv 23, MARGINAL; `lankadeepa_p22_half` csv 25.0, OK —
the single exact-threshold row). There are no wild disagreements, which is what
you want from a metric that is verdict-equivalent rather than value-exact.

---

#### Measured, 20 August, on all 168 corpus pages

Run: `system/tools/reproduce_diagnostics.py`, cached connected components over
`layout/raw_pages` (107) + `layout/raw_halfpages` (61), scored against
`page_diagnostics.csv`.

**The verdict rule is confirmed.** `resolution == OK` iff `glyph_p75 >= 25` holds
on **all 168 rows, zero disagreements**, with 7 pages marked OK. The pass mark of
25 is now established from data, not from a document.

**The CSV's exact estimator was NOT recovered.** The closest specification —
whole page, global Otsu, 8-connectivity, components with height ≥ 6 and ≤ 200 —
reproduces the CSV to a **median bias of 0.0 px** but only **~20% of pages agree
within ±0.5 px**. Scatter: median |error| 1.0 px, 77% within ±1, ~90% within ±2,
~95% within ±3, on both p50 and p75. Two hypotheses were tested and rejected:
deskewed images instead of raw made no difference at all, and a ±1 bounding-box
convention offset cannot fix p50 and p75 simultaneously.

**But the same specification reproduces the CSV's decision at 98.2%** — 165 of
168 pages, 3 disagreements (2 where it says OK and the CSV does not, 1 the
reverse). So the CSV's *value* is not portable, and its *verdict* is.

> The distinction matters for how this is written up. Do not claim the estimator
> was reproduced. Claim that an explicitly specified estimator agrees with the
> corpus verdicts on 98.2% of pages, and give the specification so the number is
> reproducible by someone else.

#### The measured p90 ↔ p75 conversion

`p90 / p75` across the 168 pages: **median 1.255**, IQR 1.200–1.351. Handoff §4.2
gives 22/18 = 1.222 on its own image set — consistent in shape.

#### The two gates are not the same requirement

Applied to the same 168 pages:

| Gate | Pages passing |
|---|---|
| `p75 >= 25` (CSV, Android app) | **8 of 168 — 4.8%** |
| `p90 >= 22` (`system/core/config.py`) | **154 of 168 — 91.7%** |
| Disagree | **146 pages** |

This settles the §0 conflict. These were never two labels for one threshold. At
the measured ratio, `p75 >= 25` corresponds to `p90 >= 31.4`, not 22. **The
scaffold's gate is roughly seven times more permissive** and would accept almost
the entire corpus that the capture app is built to reject — the corpus whose
under-resolution is your documented finding.

The cause is the conflation identified in the code: `MIN_BASE_GLYPH = 22` came
from §4.2's percentile table, which measures glyph heights **at the OCR-time
optimum, after downscaling**. It is a target for the resize step, not a minimum
for the photograph. It was then applied to whole captured frames in
`l2_select/select.py`.

#### Recomputed corpus distribution

| Set | n | p50 median | p75 median | p75 range | p90 median | p90 range |
|---|---|---|---|---|---|---|
| all | 168 | 17 | 20 | 14–51 | 26 | 19–72 |
| full | 107 | 16 | 19 | 14–51 | 24 | 19–72 |
| half | 61 | 19 | 22 | 16–30 | 27 | 21–55 |

Full per-page table: `Work/Ishara/glyph_metrics_recomputed.csv`.

#### Two reported figures that need correcting

- **"roughly 190 pages"** (handoff §, build record §1, §10) — the corpus is
  **168 pages**: 107 full, 61 half. "Seven of roughly 190" should read
  **"seven of 168"**.
- **"every full-page framing sits at glyph_p75 17–21 and every half-page framing
  at 21–23"** (build record §1) — the CSV's actual ranges are full **17–40.2**
  and half **18–25**. Six full-mode rows sit at 27 or above; they are almost
  certainly close-ups filed under `full`. The claim "every full-page framing
  fails the pass mark" is not true as written and would not survive someone
  opening the CSV.

#### Still open under D1

**§4.2's images are not the corpus pages at 0.40×.** §4.2 reports p50 = 14,
p75 = 18, p90 = 22 "at the optimum". The corpus at **native** scale measures
p50 = 17, p75 = 20, p90 = 26 — slightly *larger* than §4.2's optimum, not 2.5×
larger as a 0.40× downscale would require. So the U-curve's 1.00× baseline is not
the native 3024×4032 photograph. Until what it *was* is established from
`Pipeline_v11_Optimal_Capture.ipynb`, the U-curve cannot be re-expressed in p75
by relabelling the axis, and the 0.40× figure must not be described as "downscale
the captured photo by 0.40×" anywhere in the thesis.

---

#### Original statement of the conflict, for reference

| Where | Metric | Pass mark |
|---|---|---|
| `system/core/config.py`, `system/core/imaging.py`, `tests/test_imaging.py` | p90 connected-component height | `MIN_BASE_GLYPH = 22` |
| `layout/page_diagnostics.csv`, Android app | p75 connected-component height | 25 |

Three things established on 20 August that change the shape of this decision:

1. **`page_diagnostics.csv` has no p90 column.** Its columns are
   `file, page_id, mode, W, H, MP, deskew_deg, glyph_p50, glyph_p75, resolution`.
   "Re-express the capture findings in p75" therefore cannot be done by lookup —
   it needs recomputation from the images.

2. **`glyph_p90` is called on two different kinds of image under one name.**
   `l2_select/select.py` runs it on the whole frame and feeds `capture_verdict`
   (gate). `l4b_body/body.py` runs it on a single text-region crop via
   `rescale_to_optimum` (resize target). A full page contains headlines and
   photo-derived components; a column crop does not. These are not the same
   quantity, and `MIN_BASE_GLYPH = 22` sitting beside `TARGET_GLYPH = 24` implies
   they are.

3. **The two gates disagree about most of the corpus.** `test_imaging.py` asserts
   `capture_verdict(19) == 'warn'` — accepted with a warning. The corpus
   full-page framings sit at p75 17–21, which the CSV marks MARGINAL and the phone
   rejects. Today the backend would accept photographs the capture app is designed
   to refuse.

**Recommendation:**

- Adopt **p75, pass mark 25**, as the single *capture-time* metric. Not on
  principle — on cost. `config.py` is a one-line change; the phone is
  `PITCH_PER_GLYPH = 1.80` fitted *at* p75 = 25, plus the eight-capture validation
  table, plus a recalibration session.
- **Split the two roles in config**, which are currently conflated:
  `CAPTURE_MIN_GLYPH_P75 = 25` (is this photo good enough) and a separately named
  OCR-time resize target. Half the conflict disappears once they have different
  names.
- **Recompute, do not convert.** `system/tools/reproduce_diagnostics.py` recovers
  the CSV's estimator and then emits p50/p75/p90 side by side on the same pages,
  which also produces a real p90↔p75 mapping so the earlier p90 numbers stay
  readable in the thesis.

**The handoff's suggested resolution — "the U-curve shape does not change, only
the axis label" — is now measurably wrong for the pass mark.** It is safe for the
*shape* of the U-curve. It is not safe for the threshold: relabelling p90 ≥ 22 as
a p75 figure would change the gate from 8 passing pages to 154.

#### Change made in the code (20 August)

`system/core/config.py` now carries two separately named constants, because
they are two different measurements at two different pipeline stages:

- `CAPTURE_MIN_GLYPH_P75 = 25.0` — applied to a whole captured frame in
  `l2_select/select.py`, matching the CSV and the Android app.
  `CAPTURE_REJECT_BELOW_P75 = 15.0` is adopted from the app's `FAR_NEAR`
  boundary so phone and backend share one set of bands; it is a design choice
  and is labelled as one, not as a measurement.
- `OCR_TARGET_GLYPH_P90 = 24.0` — the resize target, applied to a single
  text-region crop in `l4b_body/body.py`. Value **unchanged and marked
  PENDING** on the §4.2 question below, rather than guessed.

`core/imaging.py` now exposes `glyph_p75` (capture), `glyph_p90` (OCR target)
and `glyph_percentiles`, all over one specified component pass: whole image,
global Otsu, 8-connectivity, components with `6 <= height <= 200`. The
specification is in the docstring, because a percentile without it is
meaningless.

`core/schemas.py` gained `glyph_p75` on `Frame` and `Article`. The change is
**additive** — `glyph_p90` is retained, so nothing a teammate depends on
breaks.

`tests/test_imaging.py` asserts the new bands and that the three percentiles
stay distinct, which is what would catch a repeat of this bug.
`tests/test_corpus_verdict.py` is new: it checks verdicts against real corpus
pages and asserts the corpus is 168 rows with 7 OK, so a drifting threshold
fails loudly. It skips cleanly when the corpus is not mounted.

**Also fixed while in here:** `app/server.py --root` was silently ignored.
`core.config` is imported at module level, so `PROJECT_ROOT` was already frozen
by the time `main()` set the environment variable. `config.set_root()` now
recomputes the dependent paths, and is called before `app.pipeline` is
imported, because that module binds `YOLO_WEIGHTS` and `MT5_PLAIN` at its own
import time.

---

### D2 — unit system: unnormalised or NFC + whitespace

**Status:** OPEN — your decision, no measurement needed.

The canonical results (CER 0.1197 → 0.0757) are computed unnormalised. The August
work used NFC + whitespace normalisation. These are different metrics; figures
from one cannot appear in a table with figures from the other.

**Recommendation:** report the **canonical unnormalised** numbers as the headline,
because they are what `results/final_results.json` holds and what the locked test
set was evaluated under. State the normalisation of every August figure at the
point of use. Do not retrofit the canonical numbers into NFC — that would mean
re-running the locked evaluation, and the locked evaluation is one of your
strongest methodological assets.

Chapter 3 must define both, name which applies where, and say why the two exist.

---

### D3 — over-correction: which denominator

**Status:** OPEN — your decision.

| Definition | B3 | Gated | Source |
|---|---|---|---|
| Original | 111 of 3,350 = 3.3% | 21 of 3,350 = 0.6% | `Chapter4_Results_FINAL.md`, Research Summary §6, handoff §2.6 |
| Later recomputation, different tokenisation | 117 of 3,486 | NOT MEASURED for gated | handoff §2.6 |

**Recommendation:** use **111 / 3,350**. Three independent documents already carry
it, the gated counterpart exists under the same definition, and the 5.5× safety
ratio — gating's only win, and a point you need for the negative-result argument —
is only computable when both systems share a denominator. The recomputation has no
gated counterpart, so adopting it would cost you the comparison.

Chapter 3 states the tokenisation used. Chapter 4 uses one and never mentions the
other.

---

### D4 — CER granularity: sentence-level or item-level

**Status:** OPEN — presentation decision.

Canonical CER (0.1197, 0.0757) is **sentence-level** on the locked test set. The
capture-resolution U-curve (0.2495 … 0.6593) is **item-level**. The handoff
already flags these as not comparable. The risk is a reader placing 0.1754 beside
0.1197 and concluding the capture work made things worse.

**Recommendation:** never put the two in one table. Label every axis and caption
explicitly. In Chapter 4, present the U-curve as *relative* change from its own
1.00× baseline (0.2205 → 0.1754, a 20.5% reduction) rather than as absolute CER,
which removes the invitation to cross-compare entirely.

---

### D5 — the v1 raw CER figure — NEW, found 20 August

**Status:** OPEN — needs a lookup, not a decision.

Three of your documents give different values for what the invalidated v1
evaluation reported:

| Value | Where |
|---|---|
| CER 0.0274 raw / 0.0065 B3, n = 1,211 | `PROJECT_HANDOFF.md` §3 |
| "raw OCR look like CER 0.0238" | `Research_Summary_Project_Knowledge.md` §3 |
| "v1 reported 0.0238 on such a biased subset" | `Chapter4_Results_FINAL.md` |

The Research Summary's own header lists **0.0274** among the obsolete v1 numbers,
while its §3 body uses **0.0238**. So the summary contradicts itself.

This matters more than its size suggests: the v1 alignment-gate failure is
**contribution #4**, and the number quantifying the inflation is the evidence for
it. An examiner who finds two values for it in your own documents will discount
the contribution.

**Action:** read the value out of the v1 results artefact rather than choosing
between documents. If both are real and measure different things (for example one
on the filtered subset and one on the full v1 set), say so explicitly — that
distinction is itself informative about how the gate worked.

---

### D6 — "38 → 230 pages" refers to what — NEW, found 20 August

**Status:** OPEN — wording, but it will be challenged.

Handoff §2.7 says "38 → 230 **training** pages moved CER 0.0763 → 0.0757". The
Research Summary and the Chapter 4 draft both give the dataset as **230
photographed pages total**, split page-disjoint into test 217 / dev 78 / train 187
**pairs** (98 / 33 / 87 pages). 230 is therefore the total corpus, not the training
set.

The data-scaling claim is one of your cleanest findings — the ceiling is optical,
not data volume — and it is stated in units that do not match your own dataset
description. Fix the wording; the finding is unaffected.

---

## 2. Claim audit — what is still live in readable documents

Your handoff has already narrowed the contribution claims. **Two older documents
still carry the un-narrowed versions**, and they are the ones a marker or
supervisor is most likely to open.

| Claim as written | Where it still lives | Handoff §8 position |
|---|---|---|
| "First Sinhala post-OCR corrector" | `Chapter4_Results_FINAL.md`, contribution 1 | Forbidden. Prior Sinhala OCR correction exists (Balasooriya 2020, rule-based; dictionary methods) |
| "First measured Sinhala OCR error taxonomy" | `Chapter4_Results_FINAL.md`; `Research_Summary` §2 | Must become "first **frequency-weighted** Sinhala OCR error taxonomy **from real photographs**" — Balasooriya's Appendix B predates it and includes ව/ච |
| "Our 36.7% **exceeds** reported low-resource reductions … Frisian 31.1%, Telugu 25.9%, Icelandic 17.9%, Irish 12.4%" | `Research_Summary` §7 | This is the cherry-pick the handoff identifies. The full Guan & Greene list also contains Russian 48.2% and English 39.5%, both above 36.7%. Correct framing: **third of seven**, with Telugu the meaningful comparison as a fellow abugida |
| "camera 13 MP min / 48 MP ideal" | `Research_Summary` §12.8 | Overturned by handoff §4.2. Specify capture as a **target glyph height**; more pixels without downscaling measured *worse* |

**Recommendation:** correct these at source before writing, not while writing. The
Research Summary is in the Claude project and is what a supervisor sees; the
Chapter 4 draft is what you will copy from. Leaving retracted claims in either is
how one reaches the final document.

A note on the TAF, which is worth being deliberate about. Your registered
sub-objective says: *"Generate noisy-clean pairs by adding realistic OCR style
errors, then train a transformer model … evaluate using CER/WER with a few real
OCR examples."* What you actually built evaluates on **217 real photographed pairs
on a locked page-disjoint split** — considerably stronger than "a few real OCR
examples". State this in Chapter 3 as a deliberate strengthening of the registered
plan. Examiners compare the TAF to the thesis, and an unexplained divergence reads
as drift even when it is an improvement.

The registered novelty — *"context-aware transformers, not only rules"* — is
**unharmed by the negative result**. B3 is the context-aware transformer, and it
wins. What failed was the gated refinement. Say it that way.

---

## 3. Chapter 3 skeleton

> The section numbering below is a conventional methodology structure. **Check it
> against the SLIIT IT4010 final report template before adopting it** — the TAF is
> the topic assessment form and does not specify the thesis chapter structure, and
> I have not seen the template.

---

### 3.1 Research design

State the shape of the study before any detail: a **controlled comparison on a
locked, page-disjoint test set**, with a **pre-registered win condition** for the
proposed architecture.

The pre-registration is the single most defensible thing in this chapter. Research
Summary §9 records that the win condition was fixed before evaluation and was not
met. Written properly, that converts "my architecture lost" into "the study was
designed so that it *could* lose, and it did" — which is what makes the negative
result publishable rather than embarrassing.

**Sources:** Research Summary §9; handoff §2.1.

---

### 3.2 Data

- 230 photographed newspaper pages, manually ground-truthed, single-article crops
- 482 correctable aligned pairs after removing OCR coverage failures and
  misalignments
- Bucket composition: CORRECTABLE 88.4%, TRUNCATED 3.6%, BLOAT 6.5%,
  MISALIGNED 1.4%
- Page-disjoint split, locked before modelling, seed 42
  (`data/splits/split_manifest.json`, `LOCKED: true`): test 217 / dev 78 /
  train 187 pairs
- Baseline CER per split — test 0.1197, dev 0.1210, train 0.1141 — as evidence the
  split is not biased. **Include this table.** It pre-empts the obvious challenge
  to any page-disjoint split.
- Synthetic training corpus: 24,153 scraped clean sentences (adaderana ~93%,
  lankadeepa, divaina), leakage-filtered against held-out ground truth (56
  sentences removed)

**Sources:** Research Summary §5; `Chapter4_Results_FINAL.md`.

---

### 3.3 Alignment, and why it is a methods section not a footnote

This is where contribution #4 is earned. Chapter 3 must describe:

- the v1 similarity gate, what it did, and that it removed ~57% of the data —
  specifically the hardest 57%
- the resulting inflation (D5 — resolve the figure first)
- the v2 rule: **retain all 217 sentences, report match-quality distributions
  instead of filtering**
- the August alignment-quality evidence: p10 = 88.1 native / 93.1 consensus, zero
  below 60

Then state the caveat once, here, so Chapter 4 can refer back: the August
end-to-end 2×2 used **ground truth to select spans**. Fair for comparison
(identical treatment both sides), optimistic in absolute terms.

**Sources:** handoff §3, §4.5; Research Summary §3.

---

### 3.4 Error taxonomy and the noise model

- 192 distinct confusions with frequencies, learned from 265 real train+dev pairs
- Top substitutions ව→ච (118), ු→ැ (91), ්→ි (65); top deletions ෙ (30), ි (25),
  ් (18); top insertions underscore variants (55, 53) — Tesseract margin artifacts
- Synthetic noise calibrated to the real word-error rate: ~0.25 generated vs 0.273
  real
- Categories derived from the Sinhala Unicode block U+0D80–U+0DFF

State both facts about dominance, as the handoff instructs: the largest single
confusion is a **consonant** (ව→ච), while **in aggregate vowel signs dominate**
(22.5% vs 20.7%). Reporting only one of these looks like selection.

**Sources:** handoff §2.8, §2.4; Research Summary §5.

---

### 3.5 Systems compared

- B1 raw OCR (Tesseract, no correction)
- B3 plain mT5-small, full-sequence
- Proposed: SinBERT-large token detector → mT5-small span corrector, unflagged
  words copied by code
- Classical controls: dictionary + edit distance; n-gram spell checker

Include the v1 lesson that shaped the design: **no `[ERROR]` markers**, because in
v1 they caused whole-sentence rewriting and hallucination. That is a methods-level
design decision with a measured cause.

Also state the detector threshold selection rule and why it differs from v1:
chosen by **false-positive rate on clean real text**, not F1. At threshold 0.30 —
recall 0.65, precision 0.95, FPR 1.6%, on 78 real dev pairs / 1,816 words. Tuned
to favour precision because under-flagging fails safe when reading aloud.

**Sources:** Research Summary §4, §3, §6; handoff §2.3.

---

### 3.6 Evaluation protocol

- Primary metrics CER and WER; define both, and define the unit system per **D2**
- Over-correction, defined per **D3**, with the tokenisation stated
- Per-error-category analysis over 1,329 error operations, with the **script vs
  formatting split** defined here so Chapter 4 can report both numbers without
  re-explaining: Sinhala script errors n = 698, formatting/junk n = 364,
  MIXED n = 267
- Significance: paired bootstrap, 10,000 resamples, 95% CI on the CER difference.
  Say why paired and why bootstrap rather than a t-test — CER differences per
  sentence are not normally distributed
- The recall-ceiling argument as a stated analysis method, not an afterthought:
  gating's ceiling is detector recall, so `achieved / ceiling` is the quantity that
  decides whether the failure is the detector or the corrector

**Sources:** handoff §2.1, §2.4, §2.5, §2.6.

---

### 3.7 Capture-side methodology (August work)

Scope this explicitly as **deployment engineering supporting the research**, not a
second research contribution. Three parts:

- the capture-resolution sweep and its measurement of glyph heights at the
  optimum — pending **D1**
- multi-frame consensus, and the condition under which voting helps: views of
  **comparable** quality (+7.9%) versus mixed quality (−3.7%), identical voting
  code both times
- the on-device estimator study: three candidates, degradation simulated rather
  than assumed, line pitch selected for <1% drift where connected-component height
  drifts +52.3%

The methodological point worth making explicitly, because it recurs: **each of
these findings only exists because the degradation was simulated rather than
assumed.** That is a method, and it belongs in Chapter 3.

**Sources:** handoff §4.1–4.3, §6; `Android_App_Build_Record.md` §6.

---

### 3.8 Threats to validity

Drawn from handoff §9. The ones that must appear because Chapter 4 cannot defend
itself without them:

single OCR engine and single domain · corpus source imbalance (~93% Adaderana
training vs Lankadeepa/Divaina test) · proximity to the ceiling (38 → 230 pages
moved CER 0.0006) · headline figure includes non-linguistic gains, script-only fix
rate is 54.6% · no rule-based baseline reproduced on this test set · four
categories below n = 50 are not interpretable · oracle alignment in the August
end-to-end test · the two unit systems (**D2**) · the two glyph metrics (**D1**)

---

## 4. What Chapter 3 cannot yet state

| Blocked on | Sections affected |
|---|---|
| **§4.2's 1.00× baseline** — read it out of `Pipeline_v11_Optimal_Capture.ipynb` | 3.7, and whether the U-curve can be re-expressed in p75 at all |
| **D5** — the v1 CER value from the results artefact | 3.3, and contribution #4 in Chapters 4 and 5 |
| The SLIIT IT4010 report template | the section numbering throughout Part 3 |
| Whether §4.2's 30 images are the corpus pages | whether the U-curve can be re-expressed by relabelling, or must be recomputed |

Everything else in Part 3 can be written from numbers that already exist.

---

## 5. Suggested order of work

1. Resolve **D2, D3, D4, D6** — decisions only, no data required. An hour.
   **D1 is resolved** (20 August) apart from applying it to `config.py`.
2. Correct the four claims in Part 2 at source, in the Research Summary and the
   Chapter 4 draft.
3. Look up **D5** in the v1 results artefact.
4. Apply **D1** to `core/config.py`, `core/imaging.py` and `tests/test_imaging.py`,
   and read `Pipeline_v11_Optimal_Capture.ipynb` to close the §4.2 question.
5. Write Chapter 3 against this skeleton.
6. Rewrite Chapter 4 from the corrected draft plus the August findings — once,
   against fixed definitions.

Step 2 is worth doing before step 5 rather than after. Retracted claims that
survive into the thesis are found by examiners, not by authors re-reading their
own prose.

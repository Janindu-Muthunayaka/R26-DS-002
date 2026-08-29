# Supervisor meeting — what to say, and the proof for each thing

**R26-DS-002 · Component 2 · IT22259134 · prepared 27 August 2026**

Read this once tonight, then once more in the morning. You do not need to memorise it.
You need to know **five moves** and the **eight numbers** at the end.

Open the evidence board on your laptop before you sit down. Everything you describe
below is on that page, in the same order.

---

## 0. Before you go in — 15 minutes of setup

Do these tonight, not in the morning. A demo that fails in the room costs more than it wins.

| # | Do this | Why |
|---|---|---|
| 1 | Open the evidence board in a browser tab, full screen | This is what you show her |
| 2 | Open `results/final_results.json` in a second tab | Your headline proof, one click away |
| 3 | Run `python tools/verify_model.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2 --limit 25` | Confirms the model on your machine still reproduces the thesis outputs |
| 4 | Run `python tools/run_pipeline.py <one photograph>` | Confirms the end-to-end path works today |
| 5 | Have the IEEE paper PDF open in a third tab | If she asks "what have you written", you show it, you do not describe it |

**If step 3 or 4 fails tonight**, do not panic and do not debug in the morning.
Say in the meeting: "the pipeline demo is on my machine, I will send you a recording."
Then show the stored outputs in `results/per_sentence_results.json` instead — real model
output on all 217 test sentences. That is proof too.

---

## 1. The opening — 90 seconds, say it almost word for word

> "My component takes the text that comes out of OCR and repairs it before it is spoken
> to the user. It never touches images — text in, text out.
>
> The headline is this. Raw OCR on a photographed Sinhala newspaper article has a word
> error rate of 33.6 percent. One word in three is wrong. My corrector brings that down
> to 16.4 percent, and character error rate from 0.1197 to 0.0757 — a 36.7 percent
> reduction. I tested significance with a paired bootstrap over ten thousand resamples,
> and the confidence interval does not cross zero.
>
> There is a second result, and I want to give it to you before you ask. The two-stage
> architecture I proposed — a detector that flags errors, then a corrector that fixes
> only those spans — **lost** to the plain single-stage model. I measured why, and the
> reason is not the one everybody guesses. The system we deploy is the plain model. I
> treat that negative result as my strongest contribution, not as a failure."

**Why open with the failure.** If you hide it and she finds it, you look like you were
covering. If you volunteer it, you look like a researcher. It is the same fact either way —
only the framing is yours to choose.

---

## 2. Move one — the problem is real, not assumed

**Say:**

> "A screen reader assumes the text is already digital. A blind Sinhala reader holding a
> printed newspaper has nothing. Photograph, recognise, speak is the obvious answer, but
> only if recognition is good enough.
>
> And it is not. A sighted reader skims past a garbled word without noticing. A listener
> cannot go back. So errors that a sighted person absorbs become failures of comprehension
> when the text is read aloud. That is why correction is a component and not an
> afterthought."

**Then add the language point — this is where she sees you understand your own problem:**

> "Sinhala makes it worse. It is an abugida. The vowel is a small mark attached to the
> consonant. At newspaper size, photographed at reading distance, those marks are a few
> pixels. They are the most fragile part of the script — and I measured that they are also
> the single biggest source of error."

**Show:** the pipeline strip on the evidence board — four boxes, yours highlighted.

---

## 3. Move two — the data, and why the split is honest

This is the part supervisors actually probe. Get ahead of it.

**Say:**

> "232 photographs of single newspaper articles, three papers. 230 of them I transcribed
> by hand. Alignment gives 553 sentence pairs, of which 489 are usable — 88.4 percent —
> and after a length filter, 482 pairs from 218 pages.
>
> The important design choice is this: **I split at photograph level, not sentence level.**
> Sentences from one article share the same font, the same print quality, the same lighting
> and the same camera angle. If I split by sentence, those conditions leak into the test
> set and the result is inflated. So the split is by page, seed 42, written to a manifest
> with a LOCKED flag before I trained anything. The test set was never used for training,
> validation, or choosing the detector threshold."

**Then give the fairness check unprompted:**

> "The check that the split is not biased: raw OCR error on train is 0.1141, on dev 0.1210,
> on test 0.1197. Three close numbers. If the test set had accidentally been the easy pages,
> you would see it there."

**Show:** the two tables in the "corpus" section of the evidence board.
**Proof file:** `data/splits/split_manifest.json`

---

## 4. Move three — the result, with the baselines that make it mean something

**Say:**

> "Five systems, all on the same 217 locked pairs.
>
> Raw OCR is 0.1197. A dictionary with edit distance gives 0.1193 — no change at all.
> An n-gram spell checker gives 0.1432, which is **worse than doing nothing**. My mT5
> corrector gives 0.0757."

**Then explain the failure of the classical methods. This is the argument that your
component needs to exist:**

> "The reason the spell checkers fail is the type of error. Spell checking assumes phonetic
> or typing mistakes. OCR mistakes are **visual**. My most frequent confusion in the whole
> corpus is ව turning into ච — 118 times. Those two letters look almost the same and sound
> nothing alike. A lexicon has no reason to prefer that correction. Worse, it replaces
> correct rare words with common ones, which is exactly how the n-gram checker made the
> text measurably worse.
>
> So this is not a case of 'neural is fashionable'. Classical methods were tried, on the
> same data, and they do not work on this error type."

**Show:** the bar chart and the five-system table.
**Proof files:** `results/final_results.json`, `results/baseline_comparison.json`

---

## 5. Move four — what actually gets repaired (your best analysis)

**Say:**

> "An error rate tells you how much a system improves text. It does not tell you *what* it
> improves. So I extracted all 1,329 error operations in the test set and classified each
> one using the Sinhala Unicode block, so every error falls in exactly one category.
>
> Two things came out of that, and one of them is uncomfortable, so I will say it first."

**The uncomfortable one — say it before she finds it:**

> "27.4 percent of the errors are not Sinhala errors at all. They are stray underscores and
> rule-line fragments that Tesseract produces from photographing a page. My model removes
> 82.4 percent of those. On the real Sinhala script errors alone it repairs **54.6 percent**.
> So part of my headline 36.7 percent is junk removal, not linguistic correction. Both
> numbers are in the paper. 54.6 is the honest measure of language ability."

**The finding you are proud of:**

> "The second thing is the more interesting one. Dependent vowel signs are the **largest**
> category — 299 errors, 22.5 percent — and simultaneously the **hardest** to repair, only
> 50.8 percent, against 61.8 percent for consonants.
>
> The reason is physical. When OCR loses a vowel sign, the evidence is not in the text any
> more. The model receives a string where that information never existed. It can predict the
> most likely mark from context, but that is prediction from a prior, not correction.
>
> And that explains something else I measured: going from 38 photographed pages to 230 moved
> CER by 0.0006. Nothing. **More data cannot restore information the camera never captured.**
> The ceiling here is optical, not statistical."

**Show:** the paired category bar chart.
**Proof file:** `results/per_category_results.json`

---

## 6. Move five — the negative result, told properly

This is the centre of the meeting. Take your time here.

**Say:**

> "My proposed architecture was two stages. A SinBERT token classifier flags the words that
> contain errors. Then mT5 corrects only those spans, and everything unflagged is copied
> unchanged. The reasoning was safety: text the detector does not flag cannot be damaged by
> the corrector. For a blind user, who cannot see that a plausible-sounding word is wrong,
> that mattered.
>
> It repairs 22.5 percent of error operations. The plain model repairs 61.8 percent."

**Now the part that makes it research rather than a bad result:**

> "The obvious objection is that my detector recall of 0.65 is too low. So I tested it. I
> swept the threshold so recall moved from 57.5 percent up to 69.3 percent, and re-ran the
> whole pipeline at every point. The line is flat — 0.1158 to 0.1168 — and never gets near
> 0.0757.
>
> Then I made it precise. A gated system can only fix what it flags, so recall is a hard
> ceiling. Achieved 0.225 divided by ceiling 0.656 gives 0.34. On the spans it *does* flag,
> it only repairs a third of what is reachable. So the loss is in the **corrector**, not in
> the detector. **Even a perfect detector would still lose.** That closes the objection
> completely.
>
> And it loses in all ten categories without exception, so it is not an unlucky error
> distribution either."

**Show the worked example — this is the moment that makes it click for her:**

> "Here is one real test sentence. OCR produced two underscore artifacts and a leftover word
> from the next sentence. The plain model removed all three. The gated system left every one
> of them.
>
> The reason is structural. The detector is trained to find **mistyped words**. An underscore
> in the page margin is not a word at all, so it is never flagged, never reaches the
> corrector, and passes straight through. The plain model regenerates the sentence, so the
> junk simply disappears."

**Give the gated system its one win — this shows you are not being self-punishing:**

> "Gating does win one thing. It destroys only 21 already-correct words out of 3,350 — 0.6
> percent — against 111, or 3.3 percent, for the plain model. It is 5.5 times safer. The
> design intent is confirmed. It just does not buy enough correction to be worth it, when
> the alternative for the user is raw OCR at 33.6 percent word error."

**Show:** the ablation chart, the η box, the Sinhala example.
**Proof file:** `results/ablation_sweep.json`

---

## 7. The honesty move — volunteer the v1 bug

Do this near the end, when she already trusts the work.

**Say:**

> "One more thing I want to tell you rather than have you find. My first evaluation reported
> raw OCR at CER 0.0238. That number is wrong and I discarded it.
>
> The alignment stage had a similarity gate — pairs below a threshold were dropped as
> unalignable. But similarity is lowest exactly where OCR did worst. So the gate removed
> about 57 percent of my data, and specifically the hardest 57 percent. What remained was
> the subset where OCR was already nearly correct, and raw OCR looked five times cleaner
> than it really is.
>
> I caught it because 0.0238 is impossible for a photographed newspaper. I threw out every
> v1 number and rebuilt the evaluation with a locked, unfiltered test set.
>
> I now report this as a finding in its own right: any post-OCR benchmark can make this
> mistake, and a paper should state what fraction of pairs its alignment stage discards and
> on what criterion."

This paragraph will do more for you than any result on the page.

---

## 8. Demo run-sheet — the order to click

Keep it to four minutes. Do not narrate the code.

| Order | Do | Say while it runs |
|---|---|---|
| 1 | Show the evidence board top strip | "These four numbers are the whole result." |
| 2 | Open `results/final_results.json` | "This is the file those come from. n = 217." |
| 3 | Run `verify_model.py --limit 25` | "This checks the model on this machine still produces the exact strings that produced the thesis numbers. It is a reproducibility check, not a metric." |
| 4 | Run `run_pipeline.py <photo>` | "Photograph in, corrected Sinhala out. This is the path the blind user actually gets." |
| 5 | Open `results/per_sentence_results.json`, scroll two examples | "Raw OCR, both system outputs, ground truth, for every test sentence. Nothing is hidden." |

If she wants to see the failure live, show the gated column in that same file for the same
sentence. It is the clearest demonstration you have.

---

## 9. Question drill

Read the question, cover the answer, try it, then check. Do this twice.

**Q1. "Your proposed architecture failed. Is your research a failure?"**
> "No. The architecture is one of five contributions. The benchmark, the taxonomy, the
> per-category analysis and the alignment-filtering finding are all independent of it. And a
> measured negative result with a mechanism is more useful to the field than another paper
> reporting a win — detect-then-correct is the intuitive design, most people would build it,
> and now they do not have to."

**Q2. "Why not just use a better OCR engine? Surya reports 2.61 percent word error on Sinhala."**
> "Three reasons. First, that benchmark uses synthetically generated black-text-on-white
> images. On the same benchmark Tesseract scores 14.89 percent; on my photographs Tesseract
> scores 33.58 percent. The condition is materially harder, and all engines degrade in it.
> Second, Tesseract runs offline on modest hardware, which the wearable deployment requires.
> Third, I did attempt Surya integration and could not complete it — it needs a container
> runtime and a served vision-language backend. That is itself a deployability finding, and
> I report it as future work."

**Q3. "Balasooriya got a 70.4 percent word-error reduction in 2020. You got 51.2. Is your work worse?"**
> "On a direct reading, yes, and I state that in the paper rather than omit it. But the
> conditions are not comparable. His test images were rendered from text files in a single
> font — 419 words, two pages. Mine are photographs of newsprint, where paper texture, ink
> bleed, focus and camera angle all degrade the input. And he used a 70,131-entry lexicon
> against a 419-word test vocabulary, which is near-total coverage; that will not transfer to
> unrestricted text. I should also say I did not reproduce his method on my test set, so the
> comparison is indirect."

**Q4. "Is 36.7 percent actually good?"**
> "The fair cross-language metric is relative CER reduction. Guan and Greene report eight
> languages ranging from 12.4 percent for Irish to 48.2 percent for Russian. My 36.7 sits
> between Spanish at 37.3 and Frisian at 31.1. The most meaningful single comparison is
> Telugu at 25.9 percent, because Telugu is also an abugida with dependent vowel signs, so it
> has the same class of difficulty. I exceed it."

**Q5. "Your test set is only 217 sentences. Is that enough?"**
> "It is 217 sentence pairs from 98 separate photographs, all real, all hand-transcribed, with
> a page-disjoint split. Size is why I report a bootstrap confidence interval rather than a
> point estimate — and the interval for the main result does not cross zero. I would rather
> have 217 honest real pairs than several thousand synthetic ones, because measuring on
> synthetic errors measures how well the model inverts its own noise function, not how well it
> corrects real OCR."

**Q6. "Why is your training data synthetic if your test data is real?"**
> "Deliberately asymmetric. There is no annotated post-OCR corpus for Sinhala, so synthetic
> noise is the only way to get training volume — that is also what Guan and Greene, and the
> RoundTripOCR work for Devanagari, do. But the noise is not invented: it is drawn from the
> 192 confusion types I measured from real photographs, weighted by how often each actually
> occurs, and calibrated to the real word error rate of 0.273. The test set is 100 percent
> real, with zero synthetic data, on purpose."

**Q7. "How do I know the test set was really locked and not tuned on?"**
> "The manifest file has the page lists, the seed and a LOCKED flag, and it was written in the
> phase before modelling. The detector threshold was chosen on the 78 development pairs, not
> on test — the rule and the full precision–recall curve are stored in
> `models/sinbert_detector/threshold.json`."

**Q8. "How did you choose the detector threshold?"**
> "By false-positive rate on real text, not by F1. The rule is: highest recall subject to at
> most 10 percent of correct words being flagged. That selects 0.30, giving recall 0.651,
> precision 0.946, false-positive rate 0.016. I favoured precision deliberately, because
> under-flagging fails safe and over-flagging risks corrupting text that was already right.
> In v1 I used F1, and that is part of why v1 failed."

**Q9. "What is your actual novel contribution? mT5 for post-OCR is not new."**
> "Correct, and I say that explicitly. The method is established for other languages. What is
> new is: first transformer post-OCR benchmark for Sinhala; first frequency-weighted Sinhala
> OCR error taxonomy from real photographs — Balasooriya's confusion list predates mine but is
> hand-curated and has no counts; the quantified negative architectural result with the
> recall-ceiling argument; and the alignment-filtering methodological finding."

**Q10. "What would you do differently with more time?"**
> "Three things, all in the paper as future work, none built. Byte-level tokenisation with
> ByT5, because it operates below the level where a vowel sign is one symbol. Canonical
> decomposition of the text, which turns atomic vowel-sign substitutions into mark deletions.
> And multimodal correction conditioned on the image — that is the only one of the three that
> can actually restore the diacritic evidence OCR destroyed, and my per-category analysis is
> what shows it is needed."

**Q11. "Is the article segmentation part of your component?"**
> "No. I trained a YOLO detector and it evaluates well — 95.6 percent correct grouping, zero
> over-merge — but the deployed reading path does not use it. The capture guidance puts the
> phone close enough that the frame is already one article, and at that distance the detector
> misfires on the neighbouring article. So article selection is done by the user aiming the
> camera. That is a defensible design for a blind reader, and I state it rather than gloss it."

**Q12. "What is the state of the write-up?"**
> "The IEEE conference paper is drafted — six pages, five tables, two figures, eleven verified
> references. Chapters 3, 4 and 5 are not written. The methodology scaffold and the definitions
> register are done, so Chapter 3 is not blocked on thinking, only on writing."

**Q13. "What is your risk between now and October?"**
> "Writing time, not results. The research is complete and measured. My plan is to stop
> running optional experiments and write the three chapters. I would like you to confirm that
> is the right call."

**Q14. "Anything you are unsure about?"**
> "Two figures I want to re-verify before the thesis. The upper part of the capture scale sweep
> and the exact v1 CER value are in artefacts stored in Drive rather than my local project, so
> I want to re-read them from source rather than quote a summary document."

---

## 10. Things you must NOT say

Guard rails. Each of these is false and will be checked.

| Do not say | Say instead |
|---|---|
| "First Sinhala OCR" | Sinhala OCR is well studied — this is about **correction** |
| "First OCR correction for Sinhala" | "First **transformer-based post-OCR** correction for Sinhala" |
| "First Sinhala OCR error taxonomy" | "First **frequency-weighted** taxonomy **from real photographs**" |
| "My 36.7% beats published low-resource results" | "It sits fourth of nine, between Spanish and Frisian, and exceeds Telugu" |
| "The gated system is my working result" | The gated system is the negative result; the plain model is what ships |
| "13 MP minimum, 48 MP ideal camera" | Withdrawn. Specify **target glyph height**, not megapixels |

---

## 11. How to close

**Say:**

> "So, to summarise. The correction model works and the improvement is significant. The
> architecture I proposed does not work, and I have a quantified reason rather than a guess.
> I have five contributions I can defend, and I have written the conference paper.
>
> What I need from you is one decision. I have about two weeks of optional experiments I
> could still run, and three unwritten chapters. My plan is to stop experimenting and write.
> Do you agree?"

Then stop talking. Let her answer. Write down what she says before you leave the room.

---

## 12. The eight numbers to memorise

If you remember nothing else, remember these.

| Number | What it is |
|---|---|
| **0.1197 → 0.0757** | Character error rate, raw OCR → corrected. **36.7% reduction** |
| **0.3358 → 0.1640** | Word error rate. **51.2% reduction** |
| **217 / 98** | Test sentence pairs / photographs. Locked, page-disjoint |
| **61.8% vs 22.5%** | Error operations repaired — plain model vs my gated design |
| **0.34** | Reachability ratio. Proves the detector is not the problem |
| **54.6%** | Fix rate on Sinhala script errors alone (the honest number) |
| **192** | Distinct character confusions measured, with counts |
| **57%** | Data the v1 alignment gate silently discarded — and the hardest 57% |

---

## 13. One last thing

You are not going in to defend a result. You are going in to report what you measured,
including the part that did not work. That is a stronger position than having everything
succeed, and it is the position that survives a viva.

Speak slowly. When she asks something you do not know, say "I have not measured that" —
that sentence costs you nothing and buys you everything.

# Short demo script — full system, then my component

**Ishara · Component 2 · ~6 minutes**
`[brackets]` = do, don't say. `—` = pause one second.

---

## A · Full system — 90 seconds

*[Hold up the phone and a newspaper.]*

> This is the whole system in one sentence: point the phone at a newspaper article,
> and it reads it aloud in Sinhala.

*[Point at the article. Let the guidance voice speak. Let the shutter fire.]*

> The phone is guiding me right now — closer, hold steady. It fires by itself when the
> text is big enough and sharp enough.

*[Wait for the audio.]*

> — That is four components working together. Capture and recognition, error correction,
> question answering, and speech. They run as separate services over HTTP, so if one is
> down it loses one feature instead of killing the system.

> Everything you just heard passed through my component on the way. Let me show you what
> it did.

---

## B · My component — 4 minutes

### 1. The problem *(40s)*

*[Show the raw OCR text on screen.]*

> This is the text before my component. Raw recognition.

*[Point.]*

> Two underscores in the middle — those are rule lines from the page, not Sinhala. And a
> word carried in from the next sentence.

> — Read that to a blind user and they hear noise, and they have no way to know it was
> noise. That is what I am fixing.

> On our corpus, raw recognition gets **one word in three wrong**.

### 2. The correction *(40s)*

*[Show the corrected output beside it.]*

> Same sentence after my model. Both underscores gone. The extra word gone.

> The model is mT5-small, fine-tuned on Sinhala noisy-to-clean sentence pairs. I never
> told it about underscores — it regenerates the sentence, so the junk does not survive.

### 3. The evidence *(60s)*

*[Open the evidence board / results file.]*

> One example is not a result. So — 217 sentence pairs from 98 photographs. All real,
> all hand-transcribed by me. Locked before I trained anything.

> Character error rate: **0.1197 to 0.0757**. That is 36.7 percent.
> Word error rate: **0.3358 to 0.1640**. That is 51.2 percent.

> — Significance: paired bootstrap, ten thousand resamples. The interval does not cross
> zero.

> And I ran two classical baselines on the same data. A dictionary gives no improvement.
> An n-gram spell checker makes it **worse**. Because spell checkers assume phonetic
> mistakes, and OCR mistakes are visual. Our most frequent confusion is ව turning into ච
> — looks the same, sounds nothing alike.

> That is why a dedicated model is necessary, not just fashionable.

### 4. The honest number *(30s)*

> One thing I want to say before you ask it.

> Twenty-seven percent of the errors are not Sinhala at all — they are the junk you saw.
> My model removes 82 percent of those. On **real Sinhala script errors alone it repairs
> 54.6 percent.**

> Both numbers are in my paper. 54.6 is the one I would ask you to judge.

### 5. The negative result *(60s)*

*[Show the flat ablation line.]*

> Last thing, and it is the result I am most proud of.

> My proposed architecture was two stages — a classifier flags the wrong words, then the
> corrector fixes only those. It repairs 22.5 percent. The plain single-stage model
> repairs 61.8.

> — The obvious objection is that my detector is too weak. So I tested it. I swept the
> threshold across the whole range and re-ran the pipeline at every point.

*[Point at the flat line.]*

> Flat. Never gets close.

> Then I made it exact: achieved 0.225, divided by the ceiling 0.656, gives **0.34**. So
> the loss is in the corrector, not the detector. **A perfect detector would still lose.**

> — So the system we deploy is the plain model. That is the correct outcome of my own
> measurement, and I report it as my strongest contribution, not as a failure.

---

## C · Hand over — 10 seconds

> That corrected text is what leaves my component. Everything after this — answering
> questions about the article, and speaking it — starts from this string.

> ⟨Name⟩, over to you.

---

## If something breaks

| | Say |
|---|---|
| Demo won't run | "The service is not responding — here is the same workflow recorded this morning." *[Play it, keep going.]* |
| Don't know | "I have not measured that. What I would measure is ⟨…⟩." |
| Number challenged | "That comes from `results/final_results.json`. May I open it?" |

## The card

```
CER  0.1197 → 0.0757   (36.7%)
WER  0.3358 → 0.1640   (51.2%)
95% CI  [−0.0615, −0.0199]
217 pairs / 98 photos / LOCKED
Fix rate: plain 61.8%  gated 22.5%
Reachability 0.34 → detector is not the problem
Script-only 54.6%  ← the honest one
```

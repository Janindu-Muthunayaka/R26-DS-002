# PP2 — spoken script

**R26-DS-002 · Ishara, IT22259134 · Component 2**

Everything in this file is written to be **said**, not read. Short sentences. Say them at
your own pace.

`[square brackets]` = stage direction, do not say it.
`— pause —` = stop for one second. Silence makes you sound in control.
`⟨…⟩` = fill in with your own value before the day.

---

# PART 1 — The 5-minute presentation

Rehearse this three times with a timer. Target **4:40**.

Remember: the evaluators pick who presents. This script is the *whole project*, not only
your component. Every member should be able to give it.

---

## 0:00 – 0:45 · Problem, stakeholders, pain point

*[Stand still. Do not touch the laptop yet.]*

> Good morning. Our project is an interactive Sinhala text reader for people who are blind
> or have low vision.
>
> — pause —
>
> Here is the problem. A screen reader assumes the text is already digital. A blind Sinhala
> speaker holding a printed newspaper has nothing. There is no path from that paper to
> speech.
>
> The obvious answer is: photograph it, recognise it, read it aloud. But that only works if
> the recognition is good enough.
>
> And it is not. On our own corpus of photographed newspaper articles, raw OCR gets **one
> word in three wrong**.
>
> — pause —
>
> That number matters more than it sounds. A sighted reader skims past a broken word without
> noticing. A listener cannot go back. So an error that a sighted person absorbs becomes a
> failure of comprehension when the text is spoken.

---

## 0:45 – 1:30 · Research gap and objectives

> So the research question is not "can we do OCR on Sinhala". That is well studied. The
> question is what to do with the errors that remain.
>
> — pause —
>
> The published work on Sinhala OCR correction is rule-based — Unicode normalisation, a list
> of confusable letter pairs, and dictionary lookup. There is transformer-based error
> correction for Sinhala, but it targets a different problem: dyslexic writing transcribed
> from speech.
>
> **No transformer-based post-OCR correction result has been published for Sinhala.** That
> is our gap.
>
> Our objectives are four. One — capture a newspaper article well enough for recognition to
> work. Two — correct the recognition errors that remain. Three — let the user ask questions
> about what was read. Four — deliver all of it as speech, in Sinhala.
>
> Those four objectives are our four components.

---

## 1:30 – 2:30 · Solution overview and architecture

*[Now put the architecture slide on screen.]*

> The architecture is one gateway and three services.
>
> The gateway runs the reading path in one process: frame selection, layout analysis,
> recognition, correction, assembly. The three other components run as separate services
> and are reached over HTTP.
>
> — pause —
>
> We chose HTTP rather than importing each other's code for two reasons, and both are
> deliberate.
>
> First, the four components pin dependency versions that cannot live in one Python
> interpreter. Our reading path pins numpy 1.26.4 and OpenCV 4.9.0, because a reproducibility
> result in our evaluation was measured under those versions. Importing another component's
> code would change the environment and invalidate a number we report.
>
> Second — and this matters today — **a component that is down degrades one feature instead
> of killing the demo.** Our service layer never raises an exception. A service that is
> unreachable returns a reason, and the reading path continues.
>
> There are two endpoints. `POST /capture` answers *"what does this page say"*. `POST /ask`
> answers *"what did that mean"*. Both use one job id, so the article you just heard is the
> article the question is about.

---

## 2:30 – 3:30 · Integration and completion status

> All four components are integrated and the end-to-end workflow runs.
>
> — pause —
>
> I want to be precise about three switches that are currently off, because two of them are
> off as the result of a measurement, not because the work is unfinished.
>
> **Article segmentation is off.** We trained a YOLO detector and it evaluates well in
> isolation. But we probed it on seventy real captures: of the fifty-one frames we could
> compare, **thirty-five — sixty-nine percent — picked a different story from the one in
> frame**. A detector that disagrees with the layout on two-thirds of frames cannot be
> allowed to choose what is read aloud to someone who cannot check it. So it is off, and the
> system asks the user for a closer frame instead.
>
> **LLM post-editing is off.** It is the only setting under which we may quote our error
> rate. If a general-purpose model rewrites the corrector's output, the number we are
> measuring is no longer the model our research is about.
>
> **Title recognition is on a safe default** until we measure it against plain Tesseract on
> the headline region.
>
> — pause —
>
> So: we would rather show you a smaller system where every number is real, than a larger one
> where some are not.

---

## 3:30 – 4:30 · Evaluation evidence and NFRs

> On evaluation. Our test set is **217 sentence pairs from 98 photographs**, all real, all
> hand-transcribed. It was locked in a manifest before any model was trained, and it has
> never been used for training, validation, or choosing a threshold.
>
> The split is at photograph level, not sentence level — because sentences from one article
> share the same font, lighting and camera angle, and splitting by sentence would leak those
> conditions into the test set and inflate the result.
>
> — pause —
>
> The result. Character error rate falls from **0.1197 to 0.0757** — a 36.7 percent
> reduction. Word error rate falls from **0.3358 to 0.1640** — 51.2 percent. We tested
> significance with a paired bootstrap over ten thousand resamples, and the confidence
> interval does not cross zero.
>
> We also ran two classical baselines on the same data. A dictionary with edit distance gives
> no improvement. An n-gram spell checker makes the text nineteen percent **worse**. That is
> the evidence that a dedicated model is necessary and not just fashionable.
>
> On non-functional requirements: every response carries per-stage timings, the suite is
> **257 test functions**, and the constants we report in the thesis are asserted by tests, so
> the system fails loudly if it drifts from the reported research.

---

## 4:30 – 5:00 · Commercialization and demo roadmap

> On value. Our user is a blind Sinhala reader, and no product serves them today. Our system
> uses a phone they already own. It runs locally, so there is no per-page cost and no
> subscription — which matters, because this is not a user group you can charge. The route to
> market is institutional: schools for the blind, library services, disability organisations.
>
> The defensible asset is not the model. Anyone can fine-tune a transformer. It is the data —
> 230 hand-transcribed photographed pages and 192 measured Sinhala recognition confusions
> that do not exist anywhere else.
>
> — pause —
>
> **In the demonstration, please watch for three things.** One: the complete workflow, from
> photograph to speech, with nobody touching code. Two: the difference between the raw
> recognised text and the corrected text — that is our research contribution, visible on
> screen. Three: what the system does when a component is unavailable.
>
> Thank you. We will begin with the end-to-end workflow.

---

*[Stop. Hand over. Do not add anything.]*

---
---

# PART 2 — Your section of the 30-minute demo

You get roughly **six to eight minutes** inside Phase 2. This is the script.

**Before you start:** the evidence board open in one tab, `results/final_results.json` in a
second, a terminal in the working directory.

---

## Beat 1 — Take the handover (15 seconds)

*[Whoever showed the capture has just produced recognised text on screen.]*

> Thank you. That is the text exactly as the recogniser produced it. From here it is my
> component — this is where the errors get repaired.
>
> — pause —
>
> My component takes text in and gives text out. It never touches the image. That boundary
> is deliberate, and I will come back to why.

---

## Beat 2 — Show the problem in one line (45 seconds)

*[Point at the raw OCR text on screen.]*

> Look at this line. There are two underscores in it that are not Sinhala at all — they are
> rule-line fragments the recogniser picked up from the page. And there is a word carried in
> from the next sentence.
>
> — pause —
>
> Now, if you read that aloud to a blind user, they hear noise in the middle of a sentence
> and they have no way to know it was noise. That is the failure mode I am solving.
>
> — pause —
>
> And there is a harder failure that you cannot see here. Sinhala is an abugida — the vowel
> is a small mark attached to the consonant. At newspaper size, photographed at reading
> distance, those marks are a few pixels wide. When the recogniser loses one, the evidence is
> gone from the text completely.

---

## Beat 3 — Run the correction (60 seconds)

*[Run the pipeline, or show the corrected output already produced.]*

```
python tools/run_pipeline.py <the photograph>
```

*[While it runs, keep talking. Do not watch the terminal in silence.]*

> While that runs — the model is mT5-small, fine-tuned on Sinhala noisy-to-clean sentence
> pairs. It works sentence by sentence rather than on a whole article, and that was a
> measured decision, not a style choice.
>
> Decoding uses beam search with four beams. That is also measured: without a repeat penalty
> the model runs past the end of a sentence and starts inventing text. In one case it
> produced output nearly three times the length of the reference.

*[Output appears.]*

> There. Both underscores are gone. The carried-in word is gone. And that happened without
> me telling the model anything about underscores — it regenerates the sentence, so the junk
> simply does not survive.

---

## Beat 4 — The number, and how it was measured (90 seconds)

*[Switch to the evidence board, top strip.]*

> Now the evidence, because one good example is not a result.
>
> — pause —
>
> Our test set is 217 sentence pairs from 98 photographs. All real, all hand-transcribed by
> me. Locked before any model was trained.
>
> Character error rate goes from **0.1197 to 0.0757**. Word error rate from **0.3358 to
> 0.1640**. That is 36.7 and 51.2 percent.
>
> — pause —
>
> Significance: paired bootstrap, ten thousand resamples, and the interval on the difference
> is minus 0.0615 to minus 0.0199. It does not cross zero.

*[Scroll to the five-system table.]*

> And here is what makes the number mean something. Same test set, five systems.
>
> A dictionary with edit distance gives 0.1193 — no change at all. An n-gram spell checker
> gives 0.1432, which is **worse than doing nothing**.
>
> The reason is the type of error. Spell checkers assume phonetic or typing mistakes. OCR
> mistakes are **visual**. Our most frequent confusion in the whole corpus is ව turning into
> ච — one hundred and eighteen times. Those two letters look nearly identical and sound
> nothing alike. A lexicon has no reason to prefer that correction.

*[If you have the file open, click `results/final_results.json` here for two seconds. Then go back.]*

> That file is where those numbers come from. I can open any of them.

---

## Beat 5 — What actually gets fixed, and the honest number (60 seconds)

*[Scroll to the category chart.]*

> An error rate tells you how much a system improves text. It does not tell you *what* it
> improves. So I classified every one of the 1,329 error operations in the test set using
> the Sinhala Unicode block, so each error falls in exactly one category.
>
> — pause —
>
> Two things came out, and one is uncomfortable, so I will give you that one first.
>
> **Twenty-seven percent of the errors are not Sinhala errors at all.** They are the stray
> underscores and rule fragments you saw earlier. My model removes 82 percent of those. On
> real Sinhala script errors alone it repairs **54.6 percent**. Both numbers are in my paper.
> Fifty-four point six is the honest measure of language ability, and that is the one I would
> ask you to judge.
>
> — pause —
>
> The second finding is the one I am proud of. **Dependent vowel signs are the largest error
> category and the hardest to repair.** Largest at 22.5 percent, and only 50.8 percent
> repaired, against 61.8 for consonants.
>
> And the reason is physical, not statistical. When the recogniser loses a vowel sign, the
> model receives a string in which that information never existed. It can predict the most
> likely mark from context — but that is prediction, not correction.
>
> Which explains something else I measured. Going from 38 photographed pages to 230 moved
> the error rate by 0.0006. Nothing. **More data cannot restore information the camera never
> captured.**

---

## Beat 6 — The negative result (90 seconds)

*[Scroll to the ablation chart.]*

> One more result, and it is the one I would most like you to ask about.
>
> — pause —
>
> The architecture I originally proposed was two stages. A SinBERT classifier flags the words
> that contain errors. Then the corrector fixes only those spans, and everything unflagged is
> copied unchanged. The reasoning was safety — text the detector does not flag cannot be
> damaged. For a blind user, who cannot see that a plausible word is wrong, that mattered.
>
> It repairs 22.5 percent of errors. The plain single-stage model repairs 61.8.
>
> — pause —
>
> Now, the obvious objection is that my detector recall is too low. So I tested it. I swept
> the threshold so recall moved from 57.5 percent up to 69.3, and re-ran the entire pipeline
> at every point.

*[Point at the flat line.]*

> The line is flat. It never approaches the baseline.
>
> Then I made it exact. A gated system can only fix what it flags, so recall is a hard
> ceiling. Achieved 0.225, divided by ceiling 0.656, gives **0.34**. On the spans it does
> flag, it only repairs a third of what is reachable.
>
> So the loss is in the corrector, not the detector. **A perfect detector would still lose.**
>
> — pause —
>
> It also loses in all ten categories without exception, so it is not an unlucky error
> distribution either.

*[Scroll to the Sinhala example.]*

> Here is why, in one sentence. The detector is trained to find **mistyped words**. An
> underscore in a page margin is not a word at all. So it is never flagged, never reaches
> the corrector, and passes straight through. The plain model regenerates the sentence, so
> the junk disappears as a side effect.
>
> — pause —
>
> I should give the gated system its one win. It destroys only 21 already-correct words out
> of 3,350, against 111 for the plain model. It is five and a half times safer. Its design
> intent was confirmed. It just does not buy enough correction to be worth it — when the
> alternative for the user is raw OCR at 33 percent word error.
>
> **So the system we deploy is the plain model. That is the correct outcome of my own
> measurement, and I report it as my strongest contribution rather than as a failure.**

---

## Beat 7 — The engineering around the model (45 seconds)

*[Switch to the terminal or the integration contract.]*

> Finally, briefly, the engineering — because the model is one line of the pipeline.
>
> Around it there is frame selection, sentence segmentation, batched inference, a decoding
> configuration that was measured, and a payload contract that is asserted by tests. 257
> test functions cover this side.
>
> — pause —
>
> One design decision I would point out. The payload I send to the answering component
> carries the corrected text, the raw text, and the changed words — but **no confidence
> scores**. Component 3 originally asked for them.
>
> I cannot produce an honest confidence. The model that would give per-token labels is the
> gated corrector, which is my negative result. Sending a confidence would put a number in
> front of another component that my research does not support. So the payload marks its
> token source as `diff`, and there is a test that fails if anyone adds a confidence field.

---

## Beat 8 — Hand over (15 seconds)

> That is the corrected text, and it is what leaves my component. Everything downstream —
> answering a question about the article, and speaking it in Sinhala — starts from this
> string.
>
> ⟨Name⟩, over to you.

---
---

# PART 3 — Emergency lines

Learn these five. They cost nothing and save the session.

| Situation | Say exactly this |
|---|---|
| The demo will not run | "The service is not responding. Here is the same workflow recorded this morning." *[Play it. Continue at normal speed. Do not apologise twice.]* |
| You do not know an answer | "I have not measured that. What I would measure is ⟨…⟩." |
| They challenge a number | "That comes from ⟨filename⟩. May I open it?" |
| They interrupt mid-sentence | "Yes — let me answer that, then I will come back to the point I was on." |
| They ask something that is another member's component | "That is ⟨name⟩'s component. What I can tell you is the interface: I send them ⟨…⟩." |
| You are running out of time | "I will stop there — the remaining detail is in the evidence pack, and I am happy to take questions on it." |

---

# PART 4 — The three sentences you must not fumble

Say each of these out loud ten times tonight.

1. > "All four components are integrated over HTTP, so a component that is down degrades one
   > feature instead of killing the demo."

2. > "The architecture I proposed lost. I measured exactly why, and the deployed system runs
   > the baseline. That negative result is my strongest contribution."

3. > "Every number I have quoted comes from a file in this repository, and I can open it now."

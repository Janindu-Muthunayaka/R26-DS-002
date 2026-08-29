# PP2 viva — readiness pack

**R26-DS-002 · Interactive Sinhala Text Reader for the Visually Impaired**
**Ishara — IT22259134 — Component 2, post-OCR error correction**

This pack is written directly against the LIC's slides. Section numbers here match
the slide topics, so you can tick them off one by one.

Two other files go with this one:

- `PP2_Demo_Script_Spoken.md` — the word-for-word script for your part of the demo
- The evidence board (already published) — the page you open when the panel says *"show me"*

---

# PART A — What the panel is actually marking

## A1. The rubric, and where your marks really are

| Rubric area | Weight | Who carries it | Your job |
|---|---|---|---|
| Proven gap / creative solution | 10% | Whole team | Show prototype → current implementation |
| **Capability in stream knowledge** | **20%** | **You (AI/DS)** | Model choice, dataset handling, metrics, baseline, validity |
| **Solution implementation** | **40%** | Whole team | Design 30 / Completion 40 / Standards 20 / PLSSE+NFR 10 |
| Communication | 15% | Whichever member is picked | Structure, Q&A, time control |
| Commercialization | 15% | Whole team | Business potential, user benefit |

**Read this carefully: 60% of the marks are stream knowledge + solution implementation.**
That is exactly where your work is strongest — you have a locked dataset, a baseline, a
metric, a significance test and a measured negative result. Most groups cannot show any
of that. Do not let the panel spend the whole session on screens and buttons.

**The 20% stream row for AI/Data Science says: model choice, dataset handling, metrics,
baseline, validity.** That is a five-word summary of your entire component. Memorise
those five words and make sure you say something about each one.

## A2. The slide that decides how you speak

> *"Do not merely name technologies. Demonstrate why you selected them, how you used them,
> and what evidence shows they worked."*

Every time you name a technology, immediately give three things:

| Technology | Why chosen | How used | Evidence it worked |
|---|---|---|---|
| mT5-small | Only multilingual seq2seq with Sinhala in its pre-training that fits a T4 | Fine-tuned on 36,743 noisy–clean pairs | CER 0.1197 → 0.0757, 95% CI excludes zero |
| SinBERT-large | The only Sinhala-pre-trained encoder; the natural choice for token classification | Detector in the gated architecture | Recall 0.651, precision 0.946 — and the architecture still lost |
| Tesseract | Runs offline on modest hardware; wearable target forbids a served backend | `sin` traineddata, default engine | 33.6% WER on photographs, which is the problem being solved |
| Synthetic noise | No annotated post-OCR corpus exists for Sinhala | 192 measured confusions, frequency-weighted | Generated WER 0.273 matches real WER 0.273 |
| difflib alignment | Deterministic, no threshold, reproducible | OCR↔ground-truth alignment | Bucket distribution reproducible from stored outputs |

**Never say "we used mT5".** Say *"we used mT5-small because it is the only multilingual
seq2seq model with Sinhala in its pre-training that fits our hardware, we fine-tuned it on
36,743 pairs generated from a measured confusion table, and it reduced character error rate
by 36.7 percent with a bootstrap interval that excludes zero."*

---

# PART B — The two maturity checklists, answered honestly

## B1. System maturity checklist

The slide's warning: *"Completion is judged against the approved plan — not against what
the team wishes to show on the day."* So answer against the plan, not against hope.

| Checklist item | Status | Evidence you can put on screen |
|---|---|---|
| All four components integrated | **Yes, over HTTP** | `docs/INTEGRATION_CONTRACT.md`; `POST /capture`, `POST /ask`, `/session/{job}`, `/document/{job}`, `/health` in `app/server.py` |
| Core workflows run end-to-end | **Yes** | Photograph → corrected Sinhala → spoken. `tools/run_pipeline.py` runs it without the phone or server |
| Major functional requirements completed | **Yes for the reading path** | Capture guidance, frame selection, OCR, correction, assembly, speech |
| Important NFRs demonstrated | **Partly — re-measure tonight** | Stage timings in the `timings` field of every `/capture` response |
| No major runtime errors | **Yes** | 257 test functions across 22 test files; `core/svc.py` never raises |
| Deployment / demo setup ready | **Yes** | `python -m app.server --root ...`, HTTPS for the phone camera, `/debug` page |
| Test data and backup plan ready | **Yes** | Pre-captured photographs + `results/per_sentence_results.json` as the fallback |

### The integration answer — learn this, it is worth marks

The panel will ask *"are all four components really integrated?"* Your answer:

> "Yes, and deliberately over HTTP rather than by importing each other's code. The four
> components pin dependency sets that cannot live in one Python interpreter — my side pins
> numpy 1.26.4 and cv2 4.9.0 because a reproducibility result in my Chapter 4 was measured
> under those versions, and importing another component's code would change the environment
> and invalidate a reported number.
>
> The second reason matters today: with HTTP and a timeout, a component that is down
> degrades one feature instead of killing the demo. `core/svc.py` never raises. Every
> integration mode defaults to the setting that leaves the reading path exactly as it is."

That paragraph answers *"did you build it properly?"*, *"does it work?"* and design
excellence at the same time. It is the single highest-value thing you can say.

## B2. Research maturity checklist

| Checklist item | Your evidence |
|---|---|
| Research gap and objectives restated clearly | No transformer post-OCR result exists for Sinhala; prior work is rule-based (Balasooriya 2020) or targets dyslexic speech (Perera 2025) |
| Evaluation aligned with objectives | Objective was error reduction on real photographed text; metric is CER/WER on real photographed text |
| Baseline or comparison strategy shown | Four comparators: raw OCR, dictionary + edit distance, n-gram spell checker, full-sequence mT5 |
| Metrics justified and results available | CER and WER, macro-averaged, defined in the paper; `results/final_results.json` |
| Limitations and validity threats acknowledged | Single engine, single domain, corpus imbalance, 27.4% formatting errors, four small-n categories, oracle alignment, two unit systems |
| Claims supported by data, not opinions | Every number names a stored file; paired bootstrap over 10,000 resamples |

**You can tick all six.** Say so plainly: *"I can evidence every line of the research
maturity checklist, and I have the file open for each one."*

---

# PART C — Gap register (this earns marks, it does not lose them)

The rubric says *"~90% completion where applicable, no major delays, **corrective actions
if needed**"*. A gap you name with a corrective action scores. A gap the panel finds
does not.

| Gap | Honest status | Corrective action to state |
|---|---|---|
| Title OCR (`TITLE_MODE`) defaults to `stub` | Janindu's model is wired in but not adopted | "It is off until we measure MAT against plain Tesseract on the title region. Adopting an unmeasured path would put an unsupported number in the pipeline." |
| Article segmentation (`SEGMENT_MODE`) off | **This is a measured decision, not a gap** | `tools/probe_yolo.py` over 70 captures: of 51 comparable frames, 35 (69%) picked a *different* story. "A detector that disagrees with the layout on two thirds of frames cannot choose what is read aloud to someone who cannot check it." |
| LLM post-editing (`POLISH_MODE`) off | Deliberate | "It is the only setting under which a CER may be quoted. If a general-purpose model rewrites mT5's output, the number being measured is no longer the model the thesis is about." |
| Chapters 3, 4, 5 | Not written | "Methodology scaffold and definitions register are complete, so the chapters are blocked on writing, not on thinking. That is my October plan." |
| Two capture-side figures | To re-verify from source notebooks | "I will not quote a number from a summary document when I can read it from the artefact." |
| End-to-end latency after batching | Estimated, not measured on the deployed path | "I re-measure it before the viva." — **do this, see D1** |

**The framing sentence for the whole register:**

> "Three of our integration switches are off, and two of those three are off because we
> measured them and the measurement said off. I would rather show you a smaller system
> where every number is real than a bigger one where some are not."

---

# PART D — Pre-flight, 48 hours before

## D1. Measurements to refresh (do these first — they are marks)

```
# 1. NFR: real end-to-end timings on the deployed path
python tools/run_pipeline.py <one photograph> --json out.json
#    -> read the "timings" block. Write the numbers on a card.

# 2. Reproducibility: the model still produces the thesis outputs
python tools/verify_model.py --root E:\RP\corpus\Sinhala_OCR_Correction_v2

# 3. Reliability: the whole suite
pytest -q          # expect 257 test functions to pass

# 4. Graceful degradation: prove it in front of them
#    Run with all three sidecars DOWN. The reading path must still work.
```

Write the timing numbers on a physical card. If the panel asks *"how fast is it?"* you
should not have to open a file.

## D2. Evidence pack — four folders, exactly as the LIC slide asks

Make one folder `PP2_Evidence/` with four subfolders. Put a `README.md` at the top listing
what is in each. Have it open in a file explorer window.

**1 · Development evidence**

| Item | Where |
|---|---|
| GitHub commits | 39 commits on `main` |
| Branch / PR history | Branches `ishara`, `bumal`, `nadee`, `Janindu-IT22072238`; merged PRs #9, #10 |
| MS Planner tasks | Export a screenshot |
| Integration logs | `timings` and `quality` blocks from real `/capture` responses |
| Deployment notes | `system/README.md` run section, HTTPS cert steps |

**2 · Testing evidence**

| Item | Where |
|---|---|
| Unit tests | `tests/test_text.py`, `test_sentences.py`, `test_imaging.py` |
| Integration tests | `tests/test_api.py`, `test_services_http.py`, `test_ask.py` |
| System tests | `tests/test_contracts.py`, `test_corpus_verdict.py` |
| NFR measurements | The `timings` block; the constants asserted in `test_imaging.py` |
| Regression notes | `test_metric_hygiene.py` — fails if a reported constant drifts |

**3 · Research evidence**

| Item | Where |
|---|---|
| Dataset / test cases | `data/splits/split_manifest.json` (locked, seed 42) |
| Baseline results | `results/baseline_comparison.json` |
| Metrics | `results/final_results.json` |
| Graphs / tables | The evidence board; the IEEE paper's five tables |
| Limitations | Section VI-C of the paper |

**4 · Governance evidence**

| Item | What to write |
|---|---|
| AI disclosure | One page: which AI tools were used, for what, and how output was verified |
| Ethics / privacy | Captured photographs are not retained; session TTL 30 min, max 32 sessions (`core/session.py`) |
| Licences | Tesseract Apache-2.0, mT5 Apache-2.0, SinBERT — check and record |
| Security controls | No image retention; services fail closed; nothing from a stub is reported as a result |
| Supervisor feedback | Meeting notes and what you changed because of them |

## D3. PLSSE — one line each, ready to say

Ten percent of the largest rubric block. Have these memorised.

| | Your answer |
|---|---|
| **Professional** | Locked test set written before modelling; every reported constant asserted by a test; library versions pinned because a result depends on them |
| **Legal** | Newspaper content is used for research only and is not redistributed; the deployed app reads a paper the user physically owns; model licences recorded |
| **Social** | The users are blind Sinhala readers, for whom no service exists today. Zero over-merge in article grouping is an accessibility decision: a sighted reader sees two articles merged instantly, a listener hears nonsense with no way to find the seam |
| **Security** | Photographs of a user's surroundings are processed and discarded, never stored; sessions expire in 30 minutes; a failing service returns a reason, never a stack trace |
| **Ethical** | The system never claims verbatim accuracy. It surfaces warnings — *"some parts were skipped"* — because a blind user cannot check the text against the page. Nothing produced by a stub is reported as a result |

---

# PART E — Commercialization (15%, and your team's weakest area)

Prepare this. It is 15% and it is the row groups most often leave empty.

## E1. The four-sentence pitch

> "Our user is a blind or low-vision Sinhala reader. Today they have no way to read a
> printed newspaper — screen readers assume digital text, and Sinhala is not served by
> commercial OCR products.
>
> Our system turns a phone they already own into a reader: point, capture, listen, and ask
> questions about what was read.
>
> It runs offline on a mid-range phone plus a laptop, so there is no per-page cloud cost and
> no subscription — which matters because the user group is not one you can charge.
>
> The route to market is institutional, not consumer: schools for the blind, library
> services and disability organisations, funded by grants or CSR rather than user fees."

## E2. Business potential — the honest version

| Question | Answer |
|---|---|
| Market size | [INSERT A CITED FIGURE for visual impairment in Sri Lanka — do not guess it] |
| Willingness to pay | Low, by design. This is a public-good product, not a consumer app |
| Cost structure | Zero marginal cost per page: Tesseract and mT5-small run locally, no API calls |
| Defensible asset | The corpus and the taxonomy. 230 hand-transcribed photographed pages and 192 measured Sinhala OCR confusions do not exist anywhere else |
| Adjacent markets | The same corrector applies to any photographed Sinhala print — government forms, exam papers, medical labels, archival digitisation |
| Barrier to a competitor | Not the model — the data. Anyone can fine-tune mT5; nobody else has the Sinhala photographed post-OCR corpus |

## E3. If they push on feasibility

> "The honest position is that at 16 percent word error we are at headline-and-gist
> quality, not verbatim reading quality. That is already useful — knowing what today's
> paper is about is something our users cannot do at all today — but I would not deploy it
> for anything where an error is costly. My per-category analysis shows exactly what would
> close the gap, and it is multimodal correction, not more data."

That answer converts a weakness into evidence that you understand your own product.

---

# PART F — The 5-minute presentation

**The evaluators choose who presents.** That means you must be able to deliver the whole
project briefing, not just your component. Rehearse this at least three times out loud
with a timer.

The full script is in `PP2_Demo_Script_Spoken.md`, Part 1. Here is the shape and the timing:

| Time | Cover | Your anchor sentence |
|---|---|---|
| 0:00–0:45 | Problem, stakeholders, measurable pain point | "One word in three is wrong. A listener cannot skim back." |
| 0:45–1:30 | Research gap + objectives | "No transformer post-OCR result exists for Sinhala." |
| 1:30–2:30 | Solution overview + architecture | "Four components, one gateway, three HTTP sidecars." |
| 2:30–3:30 | Integration and completion status | "All four integrated, and three switches off for measured reasons." |
| 3:30–4:30 | Evaluation evidence + NFRs | "CER 0.1197 to 0.0757, 257 tests, and here are the stage timings." |
| 4:30–5:00 | Commercialization + demo roadmap | "Institutional route. Now let me show you the workflow." |

**The goal of the 5 minutes, per the slide: set up the demo so the panel knows what to
observe and how to judge it.** Your last sentence must tell them what to watch for.

---

# PART G — The 30-minute demo and Q&A

## G1. The three-phase structure the LIC requires

**Phase 1 — end-to-end story first. Never start with a module.**
One photograph goes in, spoken Sinhala comes out. Nobody touches code.

**Phase 2 — component depth, one member at a time.**
Each of you shows your contribution, your integration point, and one technical decision
with the evidence behind it. Your part is scripted word for word in the demo script file.

**Phase 3 — finish with evidence.**
Tests, NFR measurements, Git and Planner, research results, limitations.

## G2. Demo rules from the slide — do not break these

- Use test data prepared **in advance**. Do not photograph something new for the first time.
- Backup recording and screenshots exist, but as **contingency only**. Do not lead with them.
- **Do not show source code unless the panel asks.** If they ask, open one file, not the repo.
- Transitions between members must be smooth. Agree the handover sentence in advance.
- Leave time for questions **during** the demo, not only at the end.

## G3. Your handover sentences

**Coming in** (from whoever showed the capture):

> "That is the text as the recogniser produced it. From here it is mine — this is where the
> errors get repaired."

**Going out** (to whoever has the answering/voice part):

> "That is the corrected text, and it is what leaves my component. Everything downstream —
> answering a question about the article, and speaking it — starts from this string."

---

# PART H — Q&A defence

## H1. The six questions behind the questions

The panel's real questions, and the one thing your answer must contain.

| They ask | They are really asking | Your answer must contain |
|---|---|---|
| "Show us it running" | **Does it work?** | A stable integrated workflow, not a module |
| "Why this architecture?" | **Did you build it properly?** | The HTTP-not-imports answer in B1 |
| "What did *you* do?" | **What did each member do?** | Your files, your commits, your component boundary |
| "How do you know it's better?" | **What is the research evidence?** | Baseline, metric, result, significance, limitation |
| "Who would use this?" | **Can it create value?** | The four-sentence pitch in E1 |
| "Did you use AI tools?" | **Was AI used responsibly?** | Disclose, say how you verified, defend the decision |

## H2. The AI-use answer — prepare this exactly

Do not be defensive and do not be vague.

> "Yes, and it is disclosed in the evidence pack. I used AI assistance for code review,
> for drafting documentation, and for checking my own reasoning. Every number in my results
> comes from a script I ran and a file stored in the repository — none of them came from a
> model. I verified the reference list against the primary sources myself, and I corrected
> two errors that were in my own earlier summaries as a result. The research decisions,
> the experiment design and the interpretation are mine."

## H3. Component-specific questions

These are on top of the fourteen already drilled in your supervisor brief. Re-read that
file too — those questions will come again.

**"Your component is just one model. Where is the engineering?"**
> "The model is one line of the pipeline. Around it there is frame selection, sentence
> segmentation, batching, a decoding configuration that was measured rather than guessed,
> a payload contract asserted by tests, and a failure path that never raises. 257 test
> functions cover it. The research is the model; the engineering is everything that gets
> usable text to it and a safe result out of it."

**"Why does your payload have no confidence scores? Component 3 asked for them."**
> "Because I cannot produce an honest one. mT5 is full-sequence sequence-to-sequence; the
> model that would give per-token labels is the SinBERT-gated corrector, which is my
> reported negative result. I send `corrected_text`, the raw text, and the changed tokens
> derived by diff, marked `token_source: diff`. Inventing a confidence would put a number
> in front of another component that my research does not support. It is asserted in
> `tests/test_rag_payload.py`."

**"How do we know your test set was not tuned on?"**
> "The manifest carries the page lists, the seed and a LOCKED flag, and it was written
> before any model was trained. The detector threshold was chosen on the 78 development
> pairs — the rule and the full curve are stored in `threshold.json`."

**"What is your contribution if mT5 already existed?"**
> "The method is established for other languages and I say that explicitly. What is new is
> the first transformer post-OCR benchmark for Sinhala, the first frequency-weighted Sinhala
> OCR error taxonomy from real photographs, a quantified negative architectural result with
> a recall-ceiling argument that closes the obvious objection, and a methodological finding
> about alignment filtering in post-OCR benchmarks."

**"Which NFRs did you demonstrate?"**
> "Five. Accuracy — CER and WER on a locked test set. Performance — per-stage timings
> returned in every response. Reliability — 257 tests, and a failure path that degrades one
> feature instead of the system. Maintainability — one schema file that every layer imports
> and that tests fail against. Security — no image retention and a 30-minute session
> expiry."

**"What if the demo fails?"**
> Do not apologise twice. Say: *"The service is not responding; here is the same workflow
> recorded this morning"* — then continue at normal speed. Panels forgive a failure handled
> calmly and remember one handled badly.

---

# PART I — The 1-minute video

## I1. Structure, mapped to your project

| Time | Content | Concretely |
|---|---|---|
| 0–10s | Real problem and affected users | A blind person holding a newspaper. Text on screen: *one word in three is wrong* |
| 10–25s | What the solution does | Phone points at an article; guidance voice; shutter fires |
| 25–45s | One end-to-end workflow or result | Raw OCR text on screen, then the corrected text replacing it, then audio |
| 45–55s | Impact / value proposition | "17 million Sinhala speakers. No reader exists for them today." |
| 55–60s | Transition to live presentation | "Here is how we built it." |

## I2. The prohibitions — from the slide

Do **not** use the video to: explain theory · show source code · add unrelated animations ·
hide a weak live demo · use copyrighted or AI-generated media without disclosure.

**The single most effective 20 seconds** you can film is the 25–45s block: the raw OCR
string on screen, then the corrected string appearing over it. It is your whole
contribution in one visual, with no narration needed.

---

# PART J — How to rehearse (three sessions, about 2 hours total)

**Session 1 — content, 40 minutes.**
Read Part F and the demo script aloud once, slowly, not timing yourself. You are checking
that you understand every sentence, not performing. Mark any sentence you would not say in
your own words and rewrite it in your own words.

**Session 2 — timing, 40 minutes.**
Deliver the 5-minute presentation three times with a timer. Target 4:40, not 5:00 — you
will speak faster in the room. Then deliver your demo section twice.

**Session 3 — pressure, 40 minutes.**
Ask a teammate to fire questions from Part H and from your supervisor brief in random
order, interrupting you mid-sentence. That is what the panel does. The skill is stopping
cleanly, answering, and returning to where you were.

## Three habits that carry marks in the Communication 15%

1. **Answer in one sentence, then stop.** Expand only if they ask. Most students lose marks
   by continuing past the answer.
2. **When you do not know, say "I have not measured that."** Then say what you would measure.
   That sentence costs nothing and protects everything else you said.
3. **Name the file.** "That is in `results/final_results.json`" ends a line of questioning
   faster than any explanation.

---

# PART K — The card to carry

Write these on one side of an index card. Nothing else.

```
CER   0.1197 -> 0.0757     36.7% reduction
WER   0.3358 -> 0.1640     51.2% reduction
95% CI on CER diff         [-0.0615, -0.0199]
Test set                   217 pairs / 98 photographs / LOCKED
Fix rate  full-seq 61.8%   gated 22.5%   (all 10 categories)
Reachability ratio         0.34  -> detector is not the problem
Script-only fix rate       54.6%  (the honest number)
Taxonomy                   192 confusions, measured, with counts
Tests                      257 functions, 22 files
Commits                    39, four member branches
Stage timings              [FILL IN FROM D1]
```

Back of the card — the three sentences you must not fumble:

1. "All four components are integrated over HTTP so that a component that is down degrades
   one feature instead of killing the demo."
2. "The architecture I proposed lost, I measured why, and the deployed system runs the
   baseline. That negative result is my strongest contribution."
3. "Every number I have quoted comes from a file in the repository, and I can open it now."

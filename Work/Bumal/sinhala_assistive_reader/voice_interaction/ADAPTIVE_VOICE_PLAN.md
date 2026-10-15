# Adaptive Voice Delivery — Build Plan (Component 4, Stage 4)

Read this file first in every Claude Code session on this feature.

## Goal

Component 4 adapts **what** is said (style: Simple / Detailed / StepByStep — already built)
and, with this feature, **how** it is said (delivery). A feedback reader decides after each
answer whether the user had a *content* problem (didn't understand → style channel) or a
*delivery* problem (didn't hear/follow → delivery channel).

Delivery settings learned per user:

| Setting | How it changes |
|---|---|
| `rate` (1.0 = normal) | explicit "slower/faster" ±10%, REPEAT on the very next turn after an answer −5%, 3 clean answers in a row +2%; clamp 0.6–1.8 |
| `pause_scale` | pause after each segment = base pause (by style) × pause_scale; REPEAT raises it a little |
| `chunk_size` | segments spoken before a "continue?" check-in; repeated mid-answer stops lower it (min 1), long clean listening raises it (max 6) |
| replay strategy | REPEAT → replay the **last spoken text** (not the article, no new Component 3 call) at the current settings; see "Additions (13 Oct)" |
| confirmation length (optional) | first ~10 turns: full confirmations; later: short ones |

Pitch stays fixed (1.0). Possible later use: a different pitch for system messages vs article content.

## Where things stand (checked 12 Oct 2026)

- **This folder (Project) is the newest version of the component** (intent prompt with known
  labels, local GGUF backend via `llm_backend.py`, strict flag grounding, GO_BACK / GO_TO_PAGE /
  GO_TO_SECTION commands). The LLM is **no longer asked for personalization_flags**, so speed
  and REPEAT signals come from keyword checks in our own code (see "Decision (12 Oct)").
- **`Branch Janindu/system/layers/l7_personalization` is an older copy (30 Aug)** plus integration
  edits: `layers.l7_personalization.…` import paths, `fallback_ollama_openai.py`, a try/except
  around `learn_one` in `style_model.py`, `server.py` (FastAPI `/interpret`, `/profile`).
  `main_flow.py` and `logger.py` differ only by line endings.
- Full-system turn: phone → reader `POST /ask` (:8000) → svc-voice `POST /interpret` (:8101)
  → RAG `POST /answer` (:8102) → reader returns `Answer` → **phone speaks only `speakable`**.
  Component 3's reply goes to the reader, not through svc-voice.
- The phone ignores fields it does not read, so an optional `delivery` field in `Answer` is safe.
- REPEAT in the full system is a LOCAL intent that re-reads the **whole article**
  (`l6_generator/generate.py`), not the last answer. Needs a decision with Ishara (Phase 6).
- Android source (`MainActivity.kt`, `GuidanceSpeaker.kt`) is **not in the repo** — on Ishara's
  machine. Group draft: TTS is owned by Component 4, client integration by Component 2.

## New code in this folder

```
delivery/
  __init__.py
  voice_profile.py   # per-user rate, pause_scale, chunk_size, counters, last_spoken; TinyDB table "voice_profiles" in data/db.json
  planner.py         # plan_speech(answer_si, style, profile) -> speech plan (pure function)
  feedback.py        # classify_followup(...) -> channel + profile update (staircase rules)
  mock_c3.py         # canned Sinhala answers per (article, style) from data/mock_answers.json
  laptop_player.py   # gTTS(lang="si") per segment + ffplay atempo=rate + silence; key press = stop
  turn.py            # (Phase 3) one user turn: order of checks, routing, last_spoken, delivery_log
simulate_delivery.py # simulated users -> CSV + summary numbers for the thesis
tests/test_planner.py, tests/test_feedback.py, tests/test_voice_profile.py, tests/test_laptop_player.py,
tests/test_mock_c3.py, tests/test_turn.py
```

### Speech plan (the contract the phone will use later)

```json
{
  "rate": 0.85, "pitch": 1.0,
  "preview": "කරුණු 3ක් තියෙනවා",
  "segments": [{"text": "පළමුව ...", "pause_after_ms": 700}, {"text": "දෙවනුව ...", "pause_after_ms": 700}],
  "checkpoints": [2],
  "replay": false
}
```

- Segmenting: split on sentence ends (`.`, `?`, `!`) and before spoken ordinals
  (පළමුව, දෙවනුව, තෙවනුව, …). Never split inside a number or abbreviation.
- Base pause by style: Simple 400 ms, Detailed 600 ms, StepByStep 800 ms (tunable).
- Preview only when segments > chunk_size.

### Feedback rules (feedback.py)

| Next user turn / playback event | Channel | Update |
|---|---|---|
| REPEAT (any time within 30 min) | delivery if it is the very next user turn after an answer, else none | replay last_spoken; only on the very next turn: rate ×0.95, pause_scale +0.1 |
| CHANGE_SPEED / keywords: slower — "slower", "slow", හෙමින්, සෙමින් | delivery | rate ×0.90 |
| CHANGE_SPEED / keywords: faster — "faster", "quick", ඉක්මනින්, වේගයෙන් | delivery | rate ×1.10 |
| stopped mid-answer (2 of last 3 long answers) | delivery | chunk_size −1 |
| SIMPLIFY / ELABORATE / STEP_BY_STEP correction | content | existing style channel; delivery unchanged |
| 3 answers in a row with no complaint | none | rate ×1.02 (drift toward comfortable max) |

CHANGE_SPEED with no slower/faster keyword → (14 Oct) change nothing yet; decision `ask` = the question
"හෙමින්ද ඉක්මනින්ද?". The next short reply (≤ 4 words: හෙමින් / ඉක්මනින් / ...) goes to
`apply_speed_answer()` and sets the speed like an explicit request; any other reply is a normal new request.
Every turn logs: rate_used, pause_scale, chunk_size, n_segments, completed, stopped_at, channel.

## Laptop setup (no phone needed until Phase 6)

- STT: existing live mic mode (`stt/google_stt_live.py`, si-LK).
- TTS: `pip install gTTS` (supports `si`, needs internet like Google STT) and FFmpeg
  (`winget install Gyan.FFmpeg`, then reopen the terminal; `ffplay -version` must work).
- Playback: `ffplay -nodisp -autoexit -af atempo=<rate> seg.mp3`, `time.sleep(pause)` between
  segments; poll `msvcrt.kbhit()` on Windows to stop mid-answer and record the segment index.
- `rate` means the same thing as Android `TextToSpeech.setSpeechRate()` (1.0 = normal), so a
  learned profile carries over to the phone.
- Cache mp3s by hash of text so replays are instant.

## Phases

0. **Safety** — `git init` in `voice_interaction/`; `.gitignore`: `venv/`, `__pycache__/`,
   `*.gguf`, `*.joblib`, `data/*.pkl`. First commit. Back up `data/db.json`, `data/style_model.pkl`.
1. **Planner + laptop player** — no learning. Done when a fixed Sinhala answer plays in segments
   at rate 0.8 / 1.0 / 1.3 with audible pauses and Enter stops it and reports the segment.
2. **Voice profile + feedback rules** — unit tests for every row of the rules table.
3. **Wire into app.py** — Stage 5 "Component 3 answer (mock)", Stage 6 "Adaptive delivery"
   (plan + play + feedback). Live mic demo works end to end. **Done 15 Oct** — see "Phase 3" below.
4. **Simulation** — users with hidden comfortable rates (0.8 / 1.0 / 1.4) and simple reply
   rules; compare fixed 1.0 vs staircase. Report turns to settle, complaints per 10 turns over
   time, final |learned − hidden|.
5. **Listening check** — 5–6 blindfolded volunteers on the laptop: default vs learned settings;
   count repeats, one comprehension question, preference.
6. **Integrate into Branch Janindu** (only after 1–5 pass):
   a. Sync this folder → `system/layers/l7_personalization`, keeping the integration edits above.
   b. svc-voice `server.py`: add `POST /deliver` (answer_si, style_class, user_id → speech plan)
      and `POST /feedback` (playback events).
   c. Reader `app/server.py /ask`: after `l6gen.answer`, call `/deliver` via `core/svc.post_json`
      (never raises; on failure omit `delivery`, phone behaves as today). Add optional
      `delivery: Optional[dict]` to `Answer` in `core/schemas.py` — **tell the team first**;
      update `docs/INTEGRATION_CONTRACT.md` and `tests/test_services_http.py`.
   d. REPEAT: agree with Ishara — REPEAT replays the last reply the reader stored for this job
      (`sessions.get_answer(job)`), and only if nothing is stored, the article as now.
   e. Phone (Ishara): if `delivery` present → `setSpeechRate`, `setPitch`, speak segments with
      `QUEUE_ADD` + `playSilentUtterance`; `UtteranceProgressListener` → `POST /feedback`.
      Else speak `speakable` as now.
   Plan B if the phone can't change in time: server builds the audio (same player code minus
   playback) and the phone plays it from a URL (`GET /audio/{job}` is already routed).

## Rules for Claude Code sessions

- One phase per session; finish with tests passing and a commit.
- Don't edit the style model. The intent prompt may be edited ONLY in Phase 2b (below).
- `handle_voice_command()` output: existing fields unchanged. (13 Oct, Phase 2b) one new field `frame` =
  `{intent, modifiers, evidence}` from `intent_detection/modifiers.build_frame()`. Delivery is still a
  separate step after Component 3.
- (13 Oct, Phase 2b) REPEAT / slower / faster / style keyword checks live in `intent_detection/modifiers.py`.
  `delivery/feedback.py` keeps only yes/no and the rules, and `apply_turn(profile, frame, now)` reads only the frame.
- Never trust an LLM flag for speed without a slower/faster keyword in the sentence.
- (14 Oct) `frame.evidence.unmatched` lists modifiers that came only from the LLM label (REPEAT with no repeat word,
  CHANGE_SPEED with no slower/faster word, SIMPLIFY / ELABORATE / STEP_BY_STEP with no style word). `handle_voice_command()`
  saves those sentences + frames to the `unmatched_modifiers` table in `data/db.json`: read it to grow the word lists.

## Decision (12 Oct): how slower/faster requests are detected

- Do NOT bring back LLM `personalization_flags`. They were wrong most of the time (speed: fast on "Simplify").
  The payload keeps `"personalization_flags": {}` so the contract stays valid.
- The intent prompt (`approach1_nllb_llm.py`) returns the intent label ONLY. It does not find keywords.
  `CHANGE_SPEED` appears there once, as an example of a "new label", so the 1B model may or may not use it,
  and the label never says slower or faster.
- (Updated 13 Oct) The intent prompt MAY be changed — the submitted thesis is a draft; the final one is due next week.
  The pipeline must stay NLLB-200 + Llama 3.2 1B (as presented). Any prompt change is decided by the Phase 2b
  experiment, and the 45-command benchmark is re-run and the new numbers go in the final thesis.
- Speed requests are detected by plain Python in `intent_detection/modifiers.py` (was `delivery/feedback.py`
  before Phase 2b): `speed_request(sinhala_text, english_text)`
  -> "slower" | "faster" | None. Check the "too fast" / "too slow" phrases first (they mean the opposite of the bare word).
  English lists match the NLLB translation; Sinhala lists (හෙමින්, සෙමින්, ඉක්මනින්, වේගයෙන්) must be checked by a
  Sinhala speaker. Test against real NLLB output, not typed English.
- If a speed keyword is found, treat the turn as a delivery change even when the intent came back as READ_ALOUD / UNKNOWN,
  so it does not trigger "read the whole article again".

## Additions (13 Oct) — REPEAT, yes/no, and keeping other components' changes small

### 1. REPEAT = say the last spoken thing again (same text, never a new answer)
- REPEAT does NOT go to Component 3. It replays the exact last text that was spoken
  (an answer, or an article part), re-planned with the user's current delivery settings.
- Standalone (this folder): REPEAT is not a system command here today, so it would go to Component 3
  as GENERATE. Catch it before the mock Component 3 call; store `last_spoken = {text, kind, finished_at}`
  in the voice profile.
- Learning: (changed 15 Oct, Phase 3) REPEAT on the **very next user turn** after an answer = "didn't hear"
  -> rate x0.95, pause +0.1, however long the user waited. REPEAT on any later turn = replay only, no change
  to settings (they just want to hear it again). There is no 20 s limit any more. `last_spoken.turns_since`
  counts user turns (0 = next turn); `feedback.apply_turn()` counts every turn it reads.
  The 30-minute expiry below still decides whether there is anything to replay.
- Style requests ("simpler", "more detail") are NOT REPEAT: they still go to Component 3 for a new answer.

### 2. REPEAT keywords in our own code (prompt unchanged)
- `intent_detection/modifiers.py: is_repeat(sinhala_text, english_text, intent)` (moved from `delivery/feedback.py`
  in Phase 2b): True if intent == REPEAT,
  OR a short utterance (<= 4 words) contains ආයෙ / ආයෙත් / ආයේ / නැවත / "again" / "repeat" / "didn't hear".
- If a style word is also present ("explain again simply"), it is a style request, not REPEAT.
- ආපසු / ආපහු also means "back" (PREVIOUS). (14 Oct) Repeat ONLY when directly followed by කියන්න / කියවන්න;
  with the word යන්න or පිටුව anywhere it is "go back", never a repeat.

### 3. Yes/no after the "continue?" check-in
- Only active on the turn right after a check-in question. Matched in our code, not the intent detector.
- Yes: ඔව්, හරි, කියවන්න, දිගටම, "yes", "ok", "continue". No: නැහැ, නෑ, එපා, ඇති, නවත්වන්න, "no", "stop".
- Anything else = a normal new request (the check-in is dropped).
- Check-ins need the phone to pause mid-answer and listen: the biggest phone change. Optional at integration.

### 4. What each component must change at integration (keep others' changes small)
| Component | Change | Size |
|---|---|---|
| Component 3 (Nadee, services/rag) | none — already returns `retrieved_sources` with `chunk_id` | 0 |
| Reader (Ishara, app/server.py + l6_generator/generate.py) | (a) call svc-voice `/deliver` after the answer, add optional `delivery` to `Answer`; (b) REPEAT: replay `sessions.get_answer(job)` text if present, else article as now | small (~30 lines) |
| Phone app (Ishara) | use `delivery` if present (rate, segments, pauses); report stopped-at | medium |
| svc-voice (mine) | `/deliver`, `/feedback`, voice profiles | most of the work |
- Add `SINHALA_DELIVERY_MODE` (default `off`) like the team's other modes: off = everything exactly as today.

### 5. IDs in the full system (checked in code)
- `job` (reader) = the captured page in session. Every `/ask` carries it, so a question is always about that page.
- `chunk_id` (`chunk_<hash>_<i>`) = a paragraph piece. Component 3 returns them in `retrieved_sources`; the reader passes
  them to the phone reply as `sources`. Not needed for REPEAT. Optional later signal: the same chunk asked about twice.
- Nothing checks "is this question about this article" by ID before answering: Component 3 always pulls 2 chunks from the
  current page plus a semantic search; if the evidence lacks the answer, the LLM returns the fixed "not enough info" sentence.

### 6. Where `last_spoken` is kept and the order of checks in one turn
- Standalone: `last_spoken = {text, kind ("answer" | "article"), style, finished_at}` per user, in the
  `voice_profiles` table of `data/db.json` (same TinyDB file the logger uses). Overwritten every time
  something is spoken. Ignore it if older than 30 min (same as the reader's session lifetime): say
  "nothing to repeat" instead.
- `finished_at` = when the laptop player finished (or was stopped). The 30-minute expiry is measured from it.
  (Phase 3) `last_spoken` also keeps `completed`, `resume_at` (first segment not yet heard) and `turns_since`.
- Full system: NOT stored by svc-voice. The reader already keeps the last reply per job in memory
  (`core/session.py`, `set_answer` / `get_answer`, 30 min lifetime). svc-voice never sees Component 3's answer.
- Order in one turn (standalone `app.py`):
  1. Check-in pending? -> yes/no check on the Sinhala text. Yes = play the remaining segments; No = stop;
     anything else = drop the check-in and continue to step 2.
     Speed question pending ("හෙමින්ද ඉක්මනින්ද?")? -> `apply_speed_answer()`; None = drop it, continue to step 2.
  2. Very short Sinhala utterance (<= 3 words) with a REPEAT or speed word -> handle it now and skip the
     slow NLLB + Llama call.
  3. Otherwise run `handle_voice_command()` unchanged (translation, intent, style, its own logging).
  4. Check its intent / English translation for REPEAT or speed words (catches what step 2 missed).
  5. SYSTEM_COMMAND -> as today. GENERATE -> mock Component 3 -> planner -> player -> save `last_spoken`
     -> feedback update -> delivery log.
- Turns handled in steps 1-2 never reach `handle_voice_command()`, so log them in the delivery log.

## Phase 2b (13 Oct): intent + modifiers — decide the final approach by experiment

Problem: one sentence can carry more than one request, e.g. "ආයෙත් හෙමින් කියන්න" (say it again, slowly)
= REPEAT + slower; "හෙමින් කියවන්න" (read slowly) = READ_ALOUD + slower; "කෙටියෙන් කියන්න කවුද දිනුවේ"
(briefly, who won) = ASK_QUESTION + Simple. The prompt returns only one intent today.

Observation: in this domain almost every multi-request sentence is ONE main action + MODIFIERS
(delivery: again / slower / faster; style: simple / detailed / steps). True two-action sentences
("stop and go to the next page") are rare.

Target output of the intent stage (same for both options):
  {"intent": "REPEAT", "modifiers": {"repeat": true, "speed": "slower", "style": null}}

Option A — LLM for the main action, rules for modifiers
  - Prompt: still one intent. Add CHANGE_SPEED as a KNOWN intent ("wants it slower or faster") and the rule
    "if the sentence also asks for an action, return the action, not CHANGE_SPEED".
  - Modifiers from `delivery/feedback.py` keyword checks (Phase 2) + existing style keywords (diagnostic.py).
Option B — LLM for everything
  - Prompt returns {"intent": "...", "speed": "slower|faster|none", "repeat": true|false, "style_words": [...]}.

Test set: the existing 45 in `data/test_samples.py` (modifier labels added for the 10 that carry one) +
16 new commands in `data/test_samples_multi.py` (Claude-drafted, Sinhala checked by Bumal on 13 Oct).

Evidence from the saved NLLB results (intent_detection_results_all.json) — why keyword checks must look at the
SINHALA text first, not only the English translation:
  - "ආයෙ" (again) -> "I'm going back."  (no "again"; Llama said PREVIOUS)
  - "කලින් කියපු කොටස තව පාරක් කියන්න" (say that part one more time) -> "Previously on "Tell Me Another Way""
  - "... කෙටියෙන් වේගයෙන් ..." (briefly, fast) -> "I want to get this down without any details" (speed word lost)
  - "මේ වාක්‍යය ටිකක් වෙනස් කරලා ආයෙ කියන්න" -> "Change the sentence a little and say it again" = REPHRASE, so
    "again" next to a change/style word is NOT a repeat.
  - "හරි හරි නවත්තන්න" = STOP, so හරි counts as "yes" ONLY right after a check-in question.
So: keyword checks run on the Sinhala text first, then the English. A second model on the English cannot
recover a word NLLB has already dropped.

Measure for both options, plus the original 45-command benchmark:
  main-intent accuracy, modifier accuracy (exact match per field), invalid-JSON rate, mean latency.
Decision rule: pick Option B only if its modifier accuracy beats A AND main-intent accuracy on the 45-command
benchmark drops by no more than 2 commands. Otherwise Option A. Record both results for the final thesis
(the losing option is a reported negative result, like the flags).

Run Phase 2b after Phase 2 (Option A needs Phase 2's keyword functions) and before Phase 3.

### Phase 2b first run (13 Oct) — results
Script: `python -m intent_detection.evaluate_phase2b` -> `intent_detection/phase2b_results.json`. Code and the fresh set
were committed BEFORE this run (commits 92f98a7, 9f4ab9c); nothing was tuned on `test_samples_fresh.py`.
"current" = production SYSTEM_PROMPT + rule modifiers (reference). Fusion = Sinhala repeat word turns PREVIOUS/NEXT/UNKNOWN into REPEAT.

| Set | Variant | Intent | Intent + fusion | repeat | speed | style | all 3 modifiers | bad JSON |
|---|---|---|---|---|---|---|---|---|
| 45 | current | 35/45 | 37/45 | 45 | 45 | 43 | 43 | 0 |
| 45 | A | 37/45 | 39/45 | 45 | 45 | 43 | 43 | 0 |
| 45 | B | 35/45 | 37/45 | 42 | 39 | 41 | 34 | 0 |
| multi | current | 13/16 | 13/16 | 16 | 16 | 16 | 16 | 0 |
| multi | A | 12/16 | 12/16 | 16 | 16 | 16 | 16 | 0 |
| multi | B | 10/16 | 10/16 | 16 | 9 | 15 | 8 | 0 |
| fresh (held-out) | current | 8/17 | 8/17 | 15 | 17 | 17 | 15 | 0 |
| fresh (held-out) | A | 9/17 | 9/17 | 15 | 17 | 17 | 15 | 0 |
| fresh (held-out) | B | 10/17 | 10/17 | 15 | 12 | 15 | 10 | 0 |

- Decision rule -> **Option A** (B's modifier accuracy is lower on every set). B is a reported negative result.
- Rule modifiers on the 45 and the 16 were developed on those sets (Phase 2), so only the fresh numbers are held-out.
- Fusion fixed sam_31 and sam_33 (both PREVIOUS) for every variant; it never fired on multi or fresh.
- Latency from this run is NOT usable: the three prompts alternate per sentence, which defeats llama.cpp's prompt-prefix
  cache (current ~9.6 s, A ~7.0 s, B ~16 s per LLM call vs ~1.4 s in the thesis run). Needs a separate pass per prompt.
- Open: production SYSTEM_PROMPT not switched to Option A yet (A vs current: 45 +2, multi −1, fresh +1). The A example
  "Read the headlines slowly" pulled m_01 "Read it slowly" to READ_HEADLINES. Style labels sam_2 / sam_16 and the fresh
  "CHECK" labels need Bumal's decision. Fresh repeat misses f_03 / f_12 use ආපසු (deliberately not a repeat word).
  (14 Oct: all resolved, see the next section.)

### Phase 2b after error analysis (14 Oct) — results
Same script and sets -> `intent_detection/phase2b_results_after_fix.json` (first-run file kept unchanged).
"original" = `SYSTEM_PROMPT_ORIGINAL`, the same prompt the first run called "current" (production now uses Option A).

Changes since the first run:
1. Production `SYSTEM_PROMPT` = Option A (does not change this table: each variant names its prompt).
2. Label corrections by Bumal (Sinhala speaker): sam_2 style = Detailed, sam_16 style = Simple; the four fresh
   "CHECK" labels (f_10, f_13, f_14, f_17) confirmed unchanged.
3. ආපසු / ආපහු + කියන්න / කියවන්න = repeat; with යන්න / පිටුව = go back (written after seeing f_03 / f_12 fail).
4. CHANGE_SPEED with no direction asks "හෙමින්ද ඉක්මනින්ද?"; unmatched-modifier logging (neither changes a score).

| Set | Variant | Intent | Intent + fusion | repeat | speed | style | all 3 modifiers | bad JSON |
|---|---|---|---|---|---|---|---|---|
| 45 | original | 36/45 | 38/45 | 45 | 45 | 45 | 45 | 0 |
| 45 | A | 37/45 | 39/45 | 45 | 45 | 45 | 45 | 0 |
| 45 | B | 35/45 | 37/45 | 42 | 37 | 42 | 33 | 0 |
| multi | original | 14/16 | 14/16 | 16 | 16 | 16 | 16 | 0 |
| multi | A | 12/16 | 12/16 | 16 | 16 | 16 | 16 | 0 |
| multi | B | 10/16 | 10/16 | 16 | 9 | 15 | 8 | 0 |
| fresh (NOT held-out any more) | original | 8/17 | 8/17 | 17 | 17 | 17 | 17 | 0 |
| fresh (NOT held-out any more) | A | 10/17 | 10/17 | 17 | 17 | 17 | 17 | 0 |
| fresh (NOT held-out any more) | B | 10/17 | 10/17 | 15 | 15 | 16 | 13 | 0 |

- What the fixes changed: style on the 45 (original / A) 43 -> 45 (the two label corrections); fresh repeat (original / A)
  15 -> 17 (ආපසු rule: f_03, f_12). Fresh numbers here are after error analysis; the held-out result stays the first-run table.
- **No fix changed an intent.** Every intent difference between the two tables is run-to-run variation of the LLM:
  with the same prompt and the same English, 14 of 234 replies differed from the first run (B 10, original 2, A 2),
  although temperature is 0 (llama.cpp floating-point / prompt-cache effects). 4 of them changed the label:
  sam_17 original EXPLAIN -> SIMPLIFY, m_07 original CHANGE_SPEED -> REPEAT, f_03 A IDENTIFY_CONTENT -> REPEAT (all now
  right), sam_24 A SIMPLIFY -> EXPLAIN (wrong both times). So treat about ±1 per set as noise when comparing variants.
- Decision unchanged: **Option A** (B's modifier accuracy is still lower on every set).
- f_12 with A: the intent is CHANGE_SPEED (counted wrong), but the frame now has repeat = True + slower, so delivery
  does the right thing (replay the last text, slower). Fusion does not override CHANGE_SPEED (only PREVIOUS / NEXT / UNKNOWN).
- Remaining A errors on fresh: f_08 "Tell me quickly" -> SUMMARIZE (speed modifier "faster" is still right),
  f_10 / f_14 -> EXPLAIN / IDENTIFY_CONTENT, f_11 / f_12 -> CHANGE_SPEED instead of REPEAT, f_15 / f_16 questions ->
  IDENTIFY_CONTENT.

### Phase 2b timing run (14 Oct) — Option A only
`python -m intent_detection.evaluate_phase2b --timing` -> `intent_detection/phase2b_timing_option_a.json`.
Option A prompt only, on the 45 commands, one prompt for the whole run (as in production), one warm-up call first
(model loading not counted). Run alone on the laptop (no other evaluation running).

| Run | Translation (NLLB) mean | LLM mean | Total mean | Intent + fusion |
|---|---|---|---|---|
| Option A, 14 Oct | 1.126 s | 0.536 s | 1.662 s | 39/45 |
| Thesis run, original prompt (saved `intent_detection_results_all.json`) | 0.920 s | 0.605 s | 1.525 s | 35/45 (no fusion) |

- Medians (Option A): translation 1.01 s, LLM 0.51 s; slowest command 4.67 s total (sam_7, the longest sentence).
- The longer Option A prompt does not make the LLM slower (prompt-prefix cache). The translation difference is the
  same NLLB model on the same sentences, so it is machine load, not the prompt.
- The 39/45 intents are identical to Option A in the after-error-analysis run, sentence by sentence.
- This replaces the unusable latency columns of both Phase 2b tables (prompts alternating defeated the cache).

## Phase 3 (15 Oct) — wired into app.py

Demo: `venv\Scripts\python.exe app.py --mode voice --user demo_user` (live mic). `--mode typed` = type Sinhala,
`--no-audio` = print the plan instead of speaking. Without options the old menus appear.

### Decisions (Bumal, 15 Oct)
1. REPEAT = "didn't hear" only on the **very next user turn** after an answer (no 20 s limit); a later REPEAT replays
   without learning; the 30-minute expiry stays. (Rules table and Additions §1 / §6 updated.)
2. Slower / faster with no repeat word: if the last answer was **stopped** mid-way (Enter, or paused at a check-in),
   continue from the first segment not heard, at the new speed. If it **finished**: no replay, say
   "හරි, හෙමින් කියන්නම්" / "හරි, ඉක්මනින් කියන්නම්" and use the new speed from the next answer.
   "Again + slower" is still a REPEAT: the whole answer at the new speed.
3. The River style model is unchanged in this phase.
4. The unused personalization-flag step (`style_from_flags`, `FLAG_STYLE_HINTS`) is removed from `main_flow.py`,
   `diagnostic.py` and `debug_intents.py`. Component 3 still receives `"personalization_flags": {}`.

### How one turn is routed (`delivery/turn.py`)
| Route | When | What happens |
|---|---|---|
| delivery | REPEAT; slower/faster with no new content (fast path, or intent CHANGE_SPEED / READ_ALOUD / UNKNOWN with no style word); CHANGE_SPEED with no direction; yes/no at a check-in | replay / resume / confirm / ask; no Component 3 call |
| content | a GENERATE request with no speed word | mock Component 3 answer in the decided style, at the current settings |
| both | a GENERATE request with a speed word (e.g. "සරලව හෙමින් කියන්න") | speed changed first, then the new answer at the new speed |
| system | SYSTEM_COMMAND | next / previous article (title spoken); other commands only printed |

- Every answer: plan -> play up to the next check-in -> `last_spoken` (with `completed`, `resume_at`, `turns_since`)
  -> `apply_playback()` when it ends. Every turn is saved in the `delivery_log` table of `data/db.json`.
- Check-in: the player pauses at each checkpoint and asks "දිගටම කියන්නද?". Yes = play the next chunk; no = stop;
  anything else = a new request (a speed word resumes from the check-in at the new speed).
- Check-in endings ("no", or a new request) do NOT go to `apply_playback()`: neither a complaint nor a clean finish.
- Fix found by the routing tests: `yes_no()` said "yes" to "කෙටියෙන් කියන්න" / "හෙමින් කියන්න" / "ආයෙ කියන්න"
  (කියන්න is a yes word). A reply with a repeat, speed or style word is now a new request.
- System messages (confirmations, questions, article titles) are spoken at the user's rate but never saved as `last_spoken`.

### Checked
- 342 tests pass (`venv\Scripts\python.exe -m pytest tests -q`); new: `test_turn.py` (routing with a fake intent
  pipeline and the silent player), `test_mock_c3.py`, `test_laptop_player.py` (start/end range with faked audio).
- Real pipeline (NLLB + Llama), typed, `--no-audio`, 8 turns: question -> check-in -> yes -> "ආයෙ කියන්න" (fast path,
  rate 0.95, pause x1.1) -> "හෙමින් කියන්න" at the check-in (resumed from segment 3 at 0.855) -> "සරල භාෂාවෙන් කියන්න"
  (Simple answer at 0.855) -> next article -> "වේගය වෙනස් කරන්න" (question) -> "ඉක්මනින්" (confirmation, 0.941).
  `data/` files were restored afterwards.
- gTTS + ffplay: all 54 demo clips synthesized and cached (`data/tts_cache/`); one real segment played via `start`/`end`.
- NOT checked by Claude: the live microphone run (needs Bumal's voice).

### Open
- `data/mock_answers.json`: Sinhala drafted by Claude (3 invented articles x 3 styles) — Bumal to check.
- Chunk-size rule: an answer listened to the end counts as "clean" even if that turn also had a REPEAT / slower
  complaint (unchanged Phase 2 rule; in the run above chunk_size went 3 -> 4). Revisit with the simulation (Phase 4).


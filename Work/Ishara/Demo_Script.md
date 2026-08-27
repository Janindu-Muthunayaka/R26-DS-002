# Demo script — 5 minutes

## Before (once)
```
cd E:\RP\R26-DS-002\system
python -m pytest tests -q          # expect 225 passed
```
Phone and laptop on the SAME WiFi. Or, safer for a viva: `adb reverse tcp:8000 tcp:8000`.

---

## DEMO A — no API key, nothing can fail (use this one)

**1. Start**
```
python tools\run_all.py --no-rag
```

**2. Read an article.** Aim the phone at a newspaper. It speaks
*"තවත් ළං වන්න"* → *"නිශ්චලව තබාගන්න"*, then fires **by itself** — no button.
It reads the article aloud.

> Say: *"The shutter is automatic because a blind user cannot see a button.
> The app measures glyph height 30 times a second and fires only when the
> distance is right and the hand is steady."*

**3. Ask about it.** Press **volume-down**, say a command:

| say | what happens |
|---|---|
| `නැවත කියවන්න` | reads the article again |
| `ඊළඟ` | the next part — press again for the one after |
| `කලින් එක` | the part before |
| `වචන කීයද` | *"මෙම ලිපියේ වචන 184 ක් ඇත."* |
| `මොනවද මඟ හැරුණේ` | what the capture missed |
| `නවත්වන්න` | stops |

> Say: *"All of these are answered from the article held in memory. No
> network, no API key. They cannot fail because a service is down."*

**4. The honesty bit.** Say `මේක සාරාංශ කරන්න` (summarise this) → it answers
*"පිළිතුරු සේවාව දැනට නොමැත."*

> Say: *"With the answering service off it says so, rather than inventing an
> answer. Nothing is ever fabricated."*

**5. The quality gate.** Take one capture from too far away → it says
*"මෙම ඡායාරූපයෙන් පැහැදිලිව කියවිය නොහැක. කරුණාකර නැවත උත්සාහ කරන්න."*

> Say: *"A bad capture doesn't fail silently. Without this the phone would
> read OCR garbage aloud in the same confident voice and the user could not
> tell."*

---

## DEMO B — add the answering component (needs the key)

```
copy services\.env.example .env      # put the NEW key + models in it
python tools\check_llm.py            # confirms what the key can reach
python tools\run_all.py
```

**6.** Volume-down → `මේ ලිපිය ගැන කියන්න` → a real Sinhala answer, grounded in
the article.

**7.** Read a SECOND article, then ask about the FIRST one.

> Say: *"It indexes every article it reads, so it can answer about what it has
> read before."*

---

## No phone? Same demo in a browser
`http://127.0.0.1:8000/debug` — upload frames, see raw OCR beside corrected
text, then the preset question buttons.

---

## If something breaks

| symptom | do |
|---|---|
| phone says "Connection failed" | long-press the screen → retype the laptop IP |
| "Speech recognition is not available" | say the command in English instead (`next`, `again`, `how long`) |
| `/ask` 404 | session expired (30 min) — capture again |
| RAG errors | `python tools\run_all.py --no-rag` — Demo A still works entirely |

**Fallback that always works:** `python tools\try_ask.py --frames work\<jobid>`
runs the whole conversation from a past capture, no phone, no camera.

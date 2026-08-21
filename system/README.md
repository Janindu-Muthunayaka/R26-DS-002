# Sinhala Reader — integrated system

Phone captures a newspaper page, the laptop reads it, audio comes back.

## Layers and ownership

| Layer | Does | Owner |
|---|---|---|
| L1 phone app | capture, guidance, upload | Ishara |
| L2 select | pick usable frames (sharpness + glyph height) | Ishara |
| L3 segment | YOLO articles + layout regions | Ishara + Janindu |
| **L4A title** | title OCR | **Janindu** |
| **L4B body** | body OCR + mT5 correction (Component 2) | **Ishara** |
| L5 generator | order, drop rejects, collect warnings | Nadee |
| **L6 speech** | RAG + Sinhala TTS | **Bumal** |

## The rule that keeps this working

**`core/schemas.py` is the contract.** L4A writes `title`, L4B writes `body`,
and neither touches the other's fields. Anyone can develop and test their
layer alone against the schema — no waiting.

Change `core/schemas.py` only after telling the team.

## Run

    pip install -r requirements.txt
    pytest -q                                    # contracts + measured constants
    python -m app.server --root "D:/Sinhala_OCR_Correction_v2"

Phone camera needs HTTPS:

    openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
            -days 90 -subj "/CN=localhost"
    python -m app.server --root "..." --cert cert.pem --key key.pem

Then `https://<laptop-ip>:8000/` on the phone, `/debug` on the laptop.

## Models

Models are **not** in this repo. Set `SINHALA_ROOT` (or pass `--root`) to the
project folder containing:

    models/mt5_plain/               config.json + model.safetensors + tokenizer
    layout/.../best.pt              YOLO article detector

## Constants you must not casually change

`core/config.py` holds measured values, not preferences:

- `TARGET_GLYPH = 24`, `MIN_BASE_GLYPH = 22` — below this the Sinhala vowel
  signs fall under ~11 px and become unrecoverable
- `OCR_SCALE_MAX = 1.0` — upscaling measured CER 0.336 at 2x and 0.659 at 3x
  against 0.175 at the optimum
- `MT5_NO_REPEAT_NGRAM = 6` — moved CER from 0.0847 to 0.0515

`tests/test_imaging.py` enforces these. If a test fails, the system has
drifted from the reported research.

## Cloud later

Nothing here assumes localhost. Moving to cloud means changing the URL the
phone posts to and running the same `app.server` on a GPU instance. Do it
only after the local version works end to end.

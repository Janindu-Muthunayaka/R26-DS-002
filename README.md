# R26-DS-002

**Interactive Sinhala Text Reader for the Visually Impaired with the assistance of OCR, NLP, and AI.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A phone captures a Sinhala newspaper page. A laptop server finds the articles,
reads their titles and body text with OCR, corrects the text, and reads it back
aloud. The user can then ask questions about what was read, by voice, and get
spoken answers in Sinhala.

## Features

- **Guided capture** – the phone app helps the user frame the page and picks the sharpest frames.
- **Article segmentation** – YOLO-based detection of articles and layout regions.
- **Sinhala OCR** – separate title and body OCR, with mT5-based error correction.
- **Spoken output** – Sinhala text-to-speech of titles and articles.
- **Ask about the article** – retrieval-augmented question answering (RAG) over what has been read.
- **Voice control** – speech commands and speech-to-text for questions.
- **Personalization** – learns the user's preferred reading style and detail level.

## Architecture

```
Phone (capture) ──► L2 select ──► L3 segment ──► L4A title OCR ─┐
                                                L4B body OCR  ─┼─► L4C polish ──► L5 assemble
                                                               ┘
L5 ──► L6 generator (RAG) ──► L7 personalization ──► L8B speech ──► audio to phone
                 ▲
       L0 voice / L8A listen (user questions and commands)
```

Layers communicate only through the schemas in `system/core/schemas.py`.
See [system/README.md](system/README.md) for layer details and the rules that keep them in sync.

## Repository structure

```
R26-DS-002/
├── system/              Integrated reader
│   ├── app/             FastAPI server and pipeline
│   ├── core/            Layer contract (schemas.py) and measured constants (config.py)
│   ├── layers/          L0–L8 pipeline layers
│   ├── web/             Phone reader, landing and debug pages
│   ├── tests/           pytest suite
│   ├── tools/           Evaluation, calibration and diagnostic scripts
│   └── docs/            Evaluation results
├── services/rag/        RAG question-answering service (Component 3)
├── Work/                Individual research workspaces per member
├── scripts/             Supporting scripts
└── setup_project.py     Scaffolds the system/ structure
```

## Getting started

### Requirements

- Python 3.11+
- Tesseract OCR with Sinhala language data
- A GPU is recommended but not required
- The trained models (not stored in this repo — see below)

### Install and run

```bash
git clone https://github.com/Janindu-Muthunayaka/R26-DS-002.git
cd R26-DS-002/system
pip install -r requirements.txt
pytest -q
python -m app.server --root "path/to/model/folder"
```

Open `https://<laptop-ip>:8000/` on the phone and `/debug` on the laptop.
The phone camera needs HTTPS; see [system/README.md](system/README.md#run) for
generating a local certificate.

### Models

Set `SINHALA_ROOT` (or pass `--root`) to a folder containing:

```
models/mt5_plain/      config.json + model.safetensors + tokenizer
layout/.../best.pt     YOLO article detector
```

### RAG service

```bash
cd services/rag
pip install -r requirements.txt
python app.py --port 8102
```

It needs `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` and `OPENAI_EMBED_MODEL` in
`system/.env`. See [services/rag/README.md](services/rag/README.md).

## Team

| Member | Responsibility |
|---|---|
| Janindu Muthunayaka | Title OCR (L4A), segmentation, system integration |
| Ishara Madusanka | Phone capture, frame selection, body OCR and mT5 correction (L4B) |
| Nadee | Assembly and RAG answer generation |
| Bumal | Speech and Sinhala TTS (L8B), personalization |

Sri Lanka Institute of Information Technology (SLIIT) — research project R26-DS-002.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
follow our [Code of Conduct](CODE_OF_CONDUCT.md). To report a security issue,
see [SECURITY.md](SECURITY.md).

## License

Released under the [MIT License](LICENSE).

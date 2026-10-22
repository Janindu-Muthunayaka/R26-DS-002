# Contributing to R26-DS-002

Thanks for helping build the Sinhala reader. This guide covers how the code is
organised, how to set up, and what a change needs before it is merged.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## How the project is organised

The system is a pipeline of layers under `system/layers/`. Each layer is owned
by one team member (see [system/README.md](system/README.md)).

- **`system/core/schemas.py` is the contract between layers.** A layer writes
  only its own fields. Do not change this file without telling the team first.
- **`system/core/config.py` holds measured constants, not preferences.**
  Values such as `TARGET_GLYPH`, `OCR_SCALE_MAX` and `MT5_NO_REPEAT_NGRAM`
  come from experiments. Change them only with new measurements, and say so
  in the pull request.
- `Work/<name>/` is each member's personal research area. Do not edit someone
  else's folder.
- Models are not stored in the repo. Point `SINHALA_ROOT` (or `--root`) at the
  folder that holds them.

## Getting set up

```bash
git clone https://github.com/Janindu-Muthunayaka/R26-DS-002.git
cd R26-DS-002/system
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
pytest -q
```

The RAG service has its own requirements; see
[services/rag/README.md](services/rag/README.md).

Secrets go in `system/.env`, which is git-ignored. Never commit API keys.

## Making a change

1. **Open or pick an issue** so others know what you are working on.
2. **Branch from `main`.** Use your name or a short topic, for example
   `ishara/body-ocr-fix` or `fix/title-crop`.
3. **Keep changes focused.** One concern per pull request.
4. **Run the tests** from `system/`:

   ```bash
   pytest -q
   ```

   `tests/test_imaging.py` checks the measured constants. If it fails, the
   system has drifted from the reported research — fix the cause, don't
   loosen the test.
5. **Add tests** for new behaviour in `system/tests/`.
6. **Open a pull request** against `main` and fill in the template.

## Commit messages

- Use the imperative mood: "Add title fallback", not "Added title fallback".
- Keep the first line under about 72 characters.
- Explain *why* in the body when it isn't obvious.

## Code style

- Python 3.11+.
- Follow PEP 8 and match the style of the file you are editing.
- Prefer small, readable functions over clever ones.
- Don't commit generated files, large datasets, model weights, `__pycache__/`
  or `.env` files.

## Reporting bugs and asking for features

Use the [issue templates](.github/ISSUE_TEMPLATE). For bugs, include the layer,
the input (a sample image if you can share it), what you expected, and what
happened.

For security problems, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.

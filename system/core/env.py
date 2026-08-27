"""
Load `.env` from the repository root, once, without a dependency.

WHY NOT python-dotenv: this project pins library versions for measured
reasons (core/config.py). A twenty-line parser is a better trade than another
entry in requirements.txt for something this simple.

WHY A FILE AND NOT A CONSTANT: an API key in a source file is a key in git
history, and git does not forget. `.env` is gitignored at the repository root;
`services/.env.example` documents the shape with no values in it.

Existing environment variables WIN. A value exported in the shell is a
deliberate override — `set OPENAI_CHAT_MODEL=...` for one run must not be
silently replaced by whatever is in the file.
"""
from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def _candidates() -> list:
    here = Path(__file__).resolve()
    system = here.parent.parent          # .../system
    return [system.parent / '.env',      # repository root — the documented spot
            system / '.env',             # convenient when running from system/
            Path.cwd() / '.env']


def load(force: bool = False) -> Path | None:
    """Read the first `.env` found. Returns the file used, or None."""
    global _loaded
    if _loaded and not force:
        return None
    _loaded = True
    for path in _candidates():
        try:
            if not path.is_file():
                continue
            for raw in path.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key, val = key.strip(), val.strip()
                if val[:1] == val[-1:] and val[:1] in ('"', "'"):
                    val = val[1:-1]
                if key and key not in os.environ:     # shell wins
                    os.environ[key] = val
            return path
        except Exception:
            continue          # an unreadable .env must never stop the server
    return None


def redact(secret: str | None) -> str:
    """For logs. Never print a key, and never print so much of one that the
    printed part is useful to whoever reads the log."""
    if not secret:
        return '(unset)'
    return f'{secret[:7]}…{secret[-4:]}' if len(secret) > 14 else '(set)'

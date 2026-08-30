"""
Talking to the other components' services.

ONE JSON POST, WITH A TIMEOUT, THAT NEVER RAISES. That is the whole file.

WHY STDLIB `urllib` AND NOT `requests` OR `httpx`
-------------------------------------------------
Because this project pins library versions for a reason. `core/config.py`
records why cv2 4.9.0 and numpy 1.26.4 are fixed, and Chapter 4 cites a
library-version reproducibility result. Adding a dependency to requirements
for four lines of HTTP is a poor trade in a codebase with that history.

IT NEVER RAISES
---------------
Every caller is on the path to a blind user's ear. A service that is down, a
laptop that is not running Ollama, a RAG container that has not started — none
of those may become a stack trace and silence. They become `ok: False` plus a
reason the caller can turn into a sentence.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional, Tuple


def post_json(url: str, body: dict, timeout_s: float = 120.0
              ) -> Tuple[Optional[dict], str]:
    """POST `body` as JSON to `url`.

    Returns (parsed_json, '') on success, or (None, reason) on any failure.
    `reason` is short and safe to log; it is never shown to the user directly.
    """
    try:
        data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError) as e:
        return None, f'unserialisable request: {type(e).__name__}'

    req = urllib.request.Request(
        url, data=data, method='POST',
        headers={'Content-Type': 'application/json; charset=utf-8',
                 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = e.read().decode('utf-8', errors='replace')[:200]
        except Exception:
            pass
        return None, f'HTTP {e.code}{": " + detail if detail else ""}'
    except urllib.error.URLError as e:
        return None, f'unreachable: {e.reason}'
    except Exception as e:                      # socket timeouts land here
        return None, f'{type(e).__name__}: {e}'

    try:
        parsed = json.loads(payload)
    except ValueError:
        return None, f'reply was not JSON: {payload[:120]!r}'
    if not isinstance(parsed, dict):
        return None, f'reply was {type(parsed).__name__}, expected an object'
    return parsed, ''

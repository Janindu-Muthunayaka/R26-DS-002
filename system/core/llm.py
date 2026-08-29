"""
The OpenAI client. Chat and embeddings, over `urllib`, with no new dependency.

WHY NOT THE `openai` PACKAGE
----------------------------
Two calls are used in this whole project — chat completions and embeddings.
The package brings httpx, pydantic pins and a release cadence into an
environment whose numpy/cv2/transformers versions are pinned because Chapter 4
cites results measured under them. Sixty lines of `urllib` is the cheaper
trade, and it is the same reasoning as `core/svc.py`.

IT NEVER RAISES. A missing key, a rate limit, an outage, a model name this
account cannot reach — each returns `(None, reason)`, never a traceback.

NO MODEL NAME IS GUESSED HERE. `OPENAI_CHAT_MODEL` and `OPENAI_EMBED_MODEL`
come from `.env`. `python tools\\check_llm.py` lists what the key can see.

PARAMETER FALLBACK. Model families disagree about `temperature` and about
`max_tokens` vs `max_completion_tokens`. A 400 that names a parameter is
retried once without it — defensive, not clever, and it logs what it dropped.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from core.env import load as _load_env

_load_env()

DEFAULT_BASE = 'https://api.openai.com/v1'
RETRY_STATUSES = (408, 409, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 3


def key() -> Optional[str]:
    k = (os.getenv('OPENAI_API_KEY') or '').strip()
    return k or None


def base_url() -> str:
    return (os.getenv('OPENAI_BASE_URL') or DEFAULT_BASE).rstrip('/')


def chat_model() -> str:
    return (os.getenv('OPENAI_CHAT_MODEL') or '').strip()


def embed_model() -> str:
    return (os.getenv('OPENAI_EMBED_MODEL') or '').strip()


def timeout_s() -> float:
    try:
        return float(os.getenv('OPENAI_TIMEOUT') or 60)
    except ValueError:
        return 60.0


def available() -> Tuple[bool, str]:
    """Can a call even be attempted? Cheap, no network."""
    if not key():
        return False, 'OPENAI_API_KEY is not set (see services/.env.example)'
    if not chat_model():
        return False, 'OPENAI_CHAT_MODEL is not set — run tools/check_llm.py'
    return True, ''


def _request(path: str, payload: Optional[dict], method='POST'
             ) -> Tuple[Optional[dict], str]:
    k = key()
    if not k:
        return None, 'OPENAI_API_KEY is not set'

    url = f'{base_url()}/{path.lstrip("/")}'
    data = (json.dumps(payload, ensure_ascii=False).encode('utf-8')
            if payload is not None else None)
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={'Authorization': f'Bearer {k}',
                 'Content-Type': 'application/json; charset=utf-8'})

    last = 'no attempt made'
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s()) as r:
                return json.loads(r.read().decode('utf-8')), ''
        except urllib.error.HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', 'replace')
            except Exception:
                pass
            if e.code == 400:
                return None, f'HTTP 400: {_msg(body)}'
            if e.code in (401, 403):
                return None, f'HTTP {e.code}: the key was rejected — {_msg(body)}'
            if e.code == 404:
                return None, (f'HTTP 404: {_msg(body)} — this key may not have '
                              f'access to that model; run tools/check_llm.py')
            last = f'HTTP {e.code}: {_msg(body)}'
            if e.code not in RETRY_STATUSES:
                return None, last
        except urllib.error.URLError as e:
            last = f'unreachable: {e.reason}'
        except Exception as e:
            last = f'{type(e).__name__}: {e}'

        if attempt < MAX_ATTEMPTS:
            time.sleep(min(2 ** attempt, 8))

    return None, f'{last} (after {MAX_ATTEMPTS} attempts)'


def _msg(body: str) -> str:
    try:
        j = json.loads(body)
        return str(j.get('error', {}).get('message') or body)[:300]
    except Exception:
        return (body or '')[:300]


_PARAM = re.compile(r"'([A-Za-z_]+)'|\"([A-Za-z_]+)\"|\b(max_tokens|"
                    r"temperature|top_p|max_completion_tokens)\b")


def _offending_param(reason: str, sent: dict) -> Optional[str]:
    """Which parameter did a 400 complain about? Only ever returns a key that
    was actually sent, so a stray quoted word cannot drop something
    unrelated."""
    for m in _PARAM.finditer(reason or ''):
        name = next((g for g in m.groups() if g), None)
        if name and name in sent and name not in ('model', 'messages', 'input'):
            return name
    return None


def chat(messages, model: str = None, temperature: float = 0.0,
         max_tokens: int = None) -> Tuple[Optional[str], str]:
    """One chat completion. Returns (text, '') or (None, reason)."""
    mdl = (model or chat_model()).strip()
    if not mdl:
        return None, 'OPENAI_CHAT_MODEL is not set — run tools/check_llm.py'

    payload = {'model': mdl, 'messages': messages}
    if temperature is not None:
        payload['temperature'] = temperature
    if max_tokens:
        payload['max_tokens'] = int(max_tokens)

    reply, reason = _request('chat/completions', payload)

    if reply is None and reason.startswith('HTTP 400'):
        drop = _offending_param(reason, payload)
        if drop:
            retry = {k: v for k, v in payload.items() if k != drop}
            if drop == 'max_tokens' and max_tokens:
                retry['max_completion_tokens'] = int(max_tokens)
            reply, reason2 = _request('chat/completions', retry)
            reason = '' if reply is not None else \
                f'{reason} | after dropping {drop}: {reason2}'

    if reply is None:
        return None, reason
    try:
        return reply['choices'][0]['message']['content'] or '', ''
    except (KeyError, IndexError, TypeError):
        return None, f'unexpected reply shape: {json.dumps(reply)[:200]}'


def embed(texts, model: str = None) -> Tuple[Optional[list], str]:
    """Embed a list of strings. Returns (vectors, '') or (None, reason).

    Order is asserted, not assumed: the API returns an `index` per item and the
    list is sorted by it. A silently reordered batch would attach every chunk
    to the wrong text, and the only symptom would be retrieval that is subtly,
    unfalsifiably wrong.
    """
    mdl = (model or embed_model()).strip()
    if not mdl:
        return None, 'OPENAI_EMBED_MODEL is not set — run tools/check_llm.py'
    items = [t for t in (texts or []) if isinstance(t, str)]
    if not items:
        return [], ''

    reply, reason = _request('embeddings', {'model': mdl, 'input': items})
    if reply is None:
        return None, reason
    try:
        rows = sorted(reply['data'], key=lambda d: d.get('index', 0))
        vecs = [r['embedding'] for r in rows]
    except (KeyError, TypeError):
        return None, f'unexpected reply shape: {json.dumps(reply)[:200]}'
    if len(vecs) != len(items):
        return None, f'asked for {len(items)} embeddings, got {len(vecs)}'
    return vecs, ''


def models() -> Tuple[Optional[list], str]:
    """Every model id this key can reach. Used by tools/check_llm.py so no
    model name in this project has to be a guess."""
    reply, reason = _request('models', None, method='GET')
    if reply is None:
        return None, reason
    try:
        return sorted(m['id'] for m in reply['data']), ''
    except (KeyError, TypeError):
        return None, 'unexpected reply shape'

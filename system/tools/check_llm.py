"""
What can this key actually reach?

    python tools\check_llm.py
    python tools\check_llm.py --models          # just list them

NO MODEL NAME IN THIS PROJECT IS A GUESS. `Work/Nadee/generate.py` hardcodes
`gpt-5.4-mini`; whether a given key can reach that is not something source
code can know. This asks.

Nothing here prints the key. `core.env.redact` shows seven characters and the
last four — enough to tell two keys apart in a log and useless to anyone who
reads it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SYSTEM = Path(__file__).resolve().parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

from core import env, llm      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', action='store_true', help='list ids and stop')
    ap.add_argument('--filter', default='',
                    help='only ids containing this substring')
    a = ap.parse_args()

    used = env.load(force=True)
    print(f'.env       : {used or "(none found — using the shell environment)"}')
    print(f'key        : {env.redact(llm.key())}')
    print(f'base url   : {llm.base_url()}')
    print(f'chat model : {llm.chat_model() or "(unset)"}')
    print(f'embed model: {llm.embed_model() or "(unset)"}')
    print()

    if not llm.key():
        print('OPENAI_API_KEY is not set.')
        print('Copy services\\.env.example to .env at the repository root and '
              'fill it in.\n.env is gitignored. Never commit a key: git does '
              'not forget, and a key\nthat reaches the repo has to be ROTATED, '
              'not deleted.')
        sys.exit(2)

    ids, reason = llm.models()
    if ids is None:
        print(f'could not list models: {reason}')
        sys.exit(2)
    if a.filter:
        ids = [m for m in ids if a.filter.lower() in m.lower()]
    print(f'{len(ids)} model(s) reachable with this key:')
    for m in ids:
        print(f'  {m}')
    if a.models:
        return

    print('\nSet OPENAI_CHAT_MODEL and OPENAI_EMBED_MODEL in .env to two of '
          'the ids above.\n')

    if llm.chat_model():
        text, reason = llm.chat(
            [{'role': 'user', 'content': 'Reply with exactly: ok'}],
            max_tokens=8)
        print(f'chat  {llm.chat_model():28} '
              + (f'-> {text.strip()!r}' if text else f'FAILED: {reason}'))
    else:
        print('chat  (OPENAI_CHAT_MODEL unset — not tested)')

    if llm.embed_model():
        vecs, reason = llm.embed(['කුරුණෑගල නගර සභාව'])
        print(f'embed {llm.embed_model():28} '
              + (f'-> {len(vecs[0])} dimensions' if vecs
                 else f'FAILED: {reason}'))
    else:
        print('embed (OPENAI_EMBED_MODEL unset — not tested)')


if __name__ == '__main__':
    main()

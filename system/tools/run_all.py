"""
Start the whole system: the reader, plus whichever components are ready.

    python tools\\run_all.py                     # reader + svc-rag
    python tools\\run_all.py --stubs             # reader + both stand-ins
    python tools\\run_all.py --no-rag            # reader alone

Ctrl-C stops everything. Each child keeps its own console output, prefixed, so
a failure is attributed to the process that had it rather than appearing as an
unexplained error in the reader.

WHY A LAUNCHER AT ALL. Four processes in four environments is the cost of the
architecture in docs/INTEGRATION_CONTRACT.md §3 — components pin dependency
sets that cannot coexist. That cost is worth paying at runtime and is not
worth paying at a viva, with three command windows to start in the right
order while somebody watches.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_SYSTEM = Path(__file__).resolve().parent.parent
_REPO = _SYSTEM.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

from core import env, llm      # noqa: E402

children: list = []


def spawn(label, args, cwd=None, extra_env=None):
    e = dict(os.environ)
    e.update(extra_env or {})
    print(f'  starting {label}: {" ".join(str(a) for a in args)}')
    p = subprocess.Popen([str(a) for a in args], cwd=str(cwd or _SYSTEM),
                         env=e)
    children.append((label, p))
    return p


def stop_all(*_):
    for label, p in children:
        if p.poll() is None:
            print(f'  stopping {label}')
            try:
                p.terminate()
            except Exception:
                pass
    for _, p in children:
        try:
            p.wait(timeout=8)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.getenv(
        'SINHALA_ROOT', r'E:/RP/corpus/Sinhala_OCR_Correction_v2'))
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--rag-port', type=int, default=8102)
    ap.add_argument('--voice-port', type=int, default=8101)
    ap.add_argument('--no-rag', action='store_true')
    ap.add_argument('--stubs', action='store_true',
                    help='use the stand-ins instead of the real services')
    ap.add_argument('--rag-python', default=None,
                    help="the rag venv's python.exe, if it has its own")
    ap.add_argument('--polish', choices=('off', 'auto', 'on'), default=None,
                    help='Layer 4C. Leave unset to keep the configured value '
                         '(off). "auto" only runs on text rated poor.')
    a = ap.parse_args()

    used = env.load(force=True)
    ok, why = llm.available()
    print('R26-DS-002 — starting the system')
    print(f'  .env    : {used or "(none)"}')
    print(f'  key     : {env.redact(llm.key())}')
    print(f'  llm     : {"ready" if ok else "NOT CONFIGURED — " + why}')
    print(f'  polish  : {a.polish or os.getenv("SINHALA_POLISH_MODE", "off")}')
    print()

    reader_env = {'SINHALA_ROOT': a.root}
    if a.polish:
        reader_env['SINHALA_POLISH_MODE'] = a.polish

    signal.signal(signal.SIGINT, lambda *x: (stop_all(), sys.exit(0)))

    try:
        if a.stubs:
            spawn('stub-voice', [sys.executable, 'tools/stub_services.py',
                                 '--role', 'voice', '--port', a.voice_port])
            spawn('stub-rag', [sys.executable, 'tools/stub_services.py',
                               '--role', 'rag', '--port', a.rag_port])
            reader_env['SINHALA_VOICE_MODE'] = 'http'
            reader_env['SINHALA_RAG_MODE'] = 'http'
        elif not a.no_rag:
            if not ok:
                print('  svc-rag needs a key — starting WITHOUT it. The '
                      'reading path and\n  every local command still work; '
                      'only "summarise this" will not.\n')
            else:
                py = a.rag_python or sys.executable
                spawn('svc-rag', [py, 'app.py', '--port', a.rag_port],
                      cwd=_REPO / 'services' / 'rag')
                reader_env['SINHALA_RAG_MODE'] = 'http'
                reader_env['SINHALA_RAG_URL'] = \
                    f'http://127.0.0.1:{a.rag_port}'
                time.sleep(2)

        spawn('reader', [sys.executable, '-m', 'app.server',
                         '--root', a.root, '--port', a.port],
              extra_env=reader_env)

        print(f'\n  phone posts to   http://<this-machine>:{a.port}/capture')
        print(f'  browser test at  http://127.0.0.1:{a.port}/debug')
        print('  ctrl-c to stop everything\n')

        while True:
            time.sleep(1)
            for label, p in list(children):
                if p.poll() is not None:
                    print(f'\n  {label} exited with code {p.returncode}')
                    if label == 'reader':
                        stop_all()
                        sys.exit(p.returncode or 1)
                    children.remove((label, p))
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


if __name__ == '__main__':
    main()

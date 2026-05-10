"""
wrapper_preprocess.py  —  Subprocess launcher for Preprocessing stage
======================================================================
Runs preprocess_helper.py using the 1_Preprocess venv (which has surya).
No direct imports of surya-dependent modules in this process.

Usage (called by app.py via subprocess):
    python wrapper_preprocess.py
        --inputs  <Inputs dir>
        --outputs <Processes dir>
        --images  img1.jpg img2.png ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FRONTEND_DIR    = Path(__file__).resolve().parent
BASE_DIR        = FRONTEND_DIR.parent
PREPROCESS_DIR  = BASE_DIR / "1_Preprocess"

# ── Locate the correct Python for the preprocess venv ─────────────────────────
PREPROCESS_PYTHON = PREPROCESS_DIR / "venv311" / "Scripts" / "python.exe"
if not PREPROCESS_PYTHON.exists():
    # Fallback: try system python
    PREPROCESS_PYTHON = Path(sys.executable)
    print(f"[wrapper_preprocess] WARNING: preprocess venv not found at expected path.")
    print(f"[wrapper_preprocess] Falling back to: {PREPROCESS_PYTHON}")

HELPER_SCRIPT = FRONTEND_DIR / "preprocess_helper.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs",  required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--images",  nargs="+", required=True)
    args = ap.parse_args()

    cmd = [
        str(PREPROCESS_PYTHON),
        str(HELPER_SCRIPT),
        "--inputs",  args.inputs,
        "--outputs", args.outputs,
        "--images",
    ] + args.images

    print(f"[wrapper_preprocess] Using Python: {PREPROCESS_PYTHON}")
    print(f"[wrapper_preprocess] Command: {' '.join(str(c) for c in cmd)}\n")

    proc = subprocess.run(
        cmd,
        cwd=str(FRONTEND_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

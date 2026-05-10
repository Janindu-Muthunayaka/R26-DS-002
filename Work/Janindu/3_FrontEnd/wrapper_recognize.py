"""
wrapper_recognize.py  —  Subprocess launcher for Recognition stage
===================================================================
Runs recognize_helper.py using the 2_Recogniton venv (PyTorch).
No direct imports of torch/model-dependent modules in this process.

Usage (called by app.py via subprocess):
    python wrapper_recognize.py
        --processes <Processes dir>
        --outputs   <Outputs dir>
        --inputs    <Inputs dir>
        --images    img1.jpg img2.png ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FRONTEND_DIR   = Path(__file__).resolve().parent
BASE_DIR       = FRONTEND_DIR.parent
RECOGNIZE_DIR  = BASE_DIR / "2_Recogniton"

# ── Locate correct Python for recognition venv ────────────────────────────────
RECOGNIZE_PYTHON = RECOGNIZE_DIR / "venv311" / "Scripts" / "python.exe"
if not RECOGNIZE_PYTHON.exists():
    RECOGNIZE_PYTHON = Path(sys.executable)
    print(f"[wrapper_recognize] WARNING: recognize venv not found at expected path.")
    print(f"[wrapper_recognize] Falling back to: {RECOGNIZE_PYTHON}")

HELPER_SCRIPT = FRONTEND_DIR / "recognize_helper.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", required=True)
    ap.add_argument("--outputs",   required=True)
    ap.add_argument("--inputs",    required=True)
    ap.add_argument("--images",    nargs="+", required=True)
    args = ap.parse_args()

    cmd = [
        str(RECOGNIZE_PYTHON),
        str(HELPER_SCRIPT),
        "--processes", args.processes,
        "--outputs",   args.outputs,
        "--inputs",    args.inputs,
        "--images",
    ] + args.images

    print(f"[wrapper_recognize] Using Python: {RECOGNIZE_PYTHON}")
    print(f"[wrapper_recognize] Helper: {HELPER_SCRIPT}\n")

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

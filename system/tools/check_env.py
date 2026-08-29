#!/usr/bin/env python3
"""
check_env.py — does this machine have everything step 9 needs?

Stdlib only, so it runs before anything is installed. Reports what is present,
what is missing, and the exact command to fix each gap. Nothing is installed or
changed.

    python check_env.py
    python check_env.py --root E:\\RP\\corpus\\Sinhala_OCR_Correction_v2
"""
import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

OK, BAD, WARN = "  ok ", " MISS", " warn"


def _ver(mod):
    for a in ("__version__", "VERSION", "version"):
        v = getattr(mod, a, None)
        if isinstance(v, str):
            return v
    return "?"


# name -> (import name, pip name, required for)
PKGS = [
    ("numpy",        "numpy",           "numpy",              "core"),
    ("opencv",       "cv2",             "opencv-python",      "core"),
    ("pillow",       "PIL",             "pillow",             "core / exif_transpose"),
    ("pydantic",     "pydantic",        "pydantic>=2",        "core schemas"),
    ("pytest",       "pytest",          "pytest",             "tests"),
    ("fastapi",      "fastapi",         "fastapi",            "server"),
    ("uvicorn",      "uvicorn",         "uvicorn[standard]",  "server"),
    ("multipart",    "multipart",       "python-multipart",   "server uploads"),
    ("torch",        "torch",           "torch",              "mT5 + YOLO"),
    ("torchvision",  "torchvision",     "torchvision",        "YOLO (ultralytics needs it)"),
    ("transformers", "transformers",    "transformers",       "mT5"),
    ("sentencepiece", "sentencepiece",  "sentencepiece",      "mT5 tokenizer"),
    ("ultralytics",  "ultralytics",     "ultralytics",        "YOLO articles"),
    ("pytesseract",  "pytesseract",     "pytesseract",        "OCR"),
    ("paddleocr",    "paddleocr",       "paddleocr>=3.1.0",   "layout regions (OPTIONAL)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("SINHALA_ROOT"))
    a = ap.parse_args()

    print("=" * 68)
    print("STEP 9 ENVIRONMENT CHECK")
    print("=" * 68)
    print(f"python   {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"exe      {sys.executable}")
    venv = os.getenv("VIRTUAL_ENV") or os.getenv("CONDA_PREFIX")
    print(f"env      {venv or 'NONE — installing will hit the system python'}")

    missing, optional_missing = [], []

    print("\n--- packages " + "-" * 54)
    for label, imp, pipname, why in PKGS:
        try:
            m = importlib.import_module(imp)
            print(f"{OK}  {label:<14} {_ver(m):<12} {why}")
        except Exception as e:
            tag = "OPTIONAL" in why
            print(f"{WARN if tag else BAD}  {label:<14} {'-':<12} {why}"
                  f"   ({type(e).__name__})")
            (optional_missing if tag else missing).append(pipname)

    # ---- torch / GPU -----------------------------------------------------
    print("\n--- gpu " + "-" * 59)
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            p = torch.cuda.get_device_properties(i)
            print(f"{OK}  CUDA {torch.version.cuda}  {p.name}  "
                  f"{p.total_memory/1e9:.1f} GB")
        else:
            print(f"{WARN}  torch present but CUDA NOT available — the pipeline "
                  f"will run on CPU (much slower, still correct)")
    except Exception:
        print(f"{BAD}  torch not importable, cannot check GPU")

    # ---- tesseract binary ------------------------------------------------
    print("\n--- tesseract binary " + "-" * 46)
    exe = shutil.which("tesseract")
    if exe is None:
        print(f"{BAD}  tesseract not on PATH")
        print("       Windows: install from "
              "https://github.com/UB-Mannheim/tesseract/wiki")
        print("       then either add it to PATH, or set in your code:")
        print("       pytesseract.pytesseract.tesseract_cmd = "
              r"r'C:\Program Files\Tesseract-OCR\tesseract.exe'")
        missing.append("<tesseract binary>")
    else:
        try:
            v = subprocess.run([exe, "--version"], capture_output=True,
                               text=True, timeout=20).stdout.splitlines()[0]
        except Exception as e:
            v = f"(version check failed: {e})"
        print(f"{OK}  {exe}")
        print(f"       {v}")
        try:
            # first line is a header ("List of available languages ...")
            out = subprocess.run([exe, "--list-langs"], capture_output=True,
                                 text=True, timeout=20).stdout.splitlines()
            langs = [l.strip() for l in out[1:] if l.strip()]
        except Exception:
            langs = []
        if "sin" in langs:
            print(f"{OK}  'sin' language data present")
        else:
            print(f"{BAD}  'sin' language data NOT installed  "
                  f"(found: {', '.join(langs[:8]) or 'none'})")
            print("       download sin.traineddata from "
                  "https://github.com/tesseract-ocr/tessdata")
            print("       and put it in the tessdata folder next to tesseract.exe")
            missing.append("<sin.traineddata>")

    # ---- corpus / models -------------------------------------------------
    print("\n--- data and models " + "-" * 47)
    if not a.root:
        print(f"{WARN}  no --root and no SINHALA_ROOT set; using the config default")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from core import config
            root = Path(config.PROJECT_ROOT)
        except Exception:
            root = None
    else:
        root = Path(a.root)

    if root is None:
        print(f"{BAD}  could not determine the corpus root")
    else:
        print(f"       root: {root}")
        checks = [
            ("YOLO weights", root/"layout"/"runs"/"articles_full"/"weights"/"best.pt", 1e6),
            ("mT5 config",   root/"models"/"mt5_plain"/"config.json", 1),
            ("mT5 weights",  root/"models"/"mt5_plain"/"model.safetensors", 1e9),
            ("mT5 tokenizer", root/"models"/"mt5_plain"/"tokenizer.json", 1e6),
            ("page_diagnostics.csv", root/"layout"/"page_diagnostics.csv", 1e3),
            ("raw_pages/",   root/"layout"/"raw_pages", None),
        ]
        for label, p, minsize in checks:
            if not p.exists():
                print(f"{BAD}  {label:<22} not found at {p}")
                missing.append(f"<{label}>")
                continue
            if p.is_dir():
                n = len(list(p.glob('*.jpg')))
                print(f"{OK}  {label:<22} {n} jpg files")
            else:
                sz = p.stat().st_size
                flag = OK if (minsize is None or sz >= minsize) else BAD
                note = "" if flag == OK else "  <-- TOO SMALL, likely a Drive placeholder"
                print(f"{flag}  {label:<22} {sz/1e6:9.1f} MB{note}")
                if flag is BAD:
                    missing.append(f"<{label} truncated>")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 68)
    if not missing:
        print("READY — everything step 9 needs is present.")
        if optional_missing:
            print(f"Optional not installed: {', '.join(optional_missing)}")
            print("PaddleOCR is optional; the pipeline falls back without it.")
    else:
        print("NOT READY. Missing:")
        pip_items = [m for m in missing if not m.startswith("<")]
        manual = [m for m in missing if m.startswith("<")]
        for m in missing:
            print(f"  - {m}")
        if pip_items:
            print("\nInstall with:")
            print(f"  {Path(sys.executable).name} -m pip install " + " ".join(pip_items))
        if manual:
            print("\nManual steps are printed above for: " + ", ".join(manual))
    print("=" * 68)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())

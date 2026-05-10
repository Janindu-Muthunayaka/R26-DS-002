"""
app.py  —  Sinhala OCR  |  Frontend Backend Server
====================================================
Serves the HTML frontend and orchestrates the two-stage pipeline:
  1. MainPreProcess.py  ->  Inputs -> Processes
  2. MainRecognize.py   ->  Processes -> Outputs + Report

Run with:
    python app.py
Then open: http://localhost:5000
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
import threading
import time
import uuid
import base64
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file

# ── Directory layout ──────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent          # 3_FrontEnd/
INPUTS_DIR   = FRONTEND_DIR / "Inputs"
PROCESSES_DIR= FRONTEND_DIR / "Processes"
OUTPUTS_DIR  = FRONTEND_DIR / "Outputs"

for d in (INPUTS_DIR, PROCESSES_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Sibling pipeline scripts ───────────────────────────────────────────────────
BASE         = FRONTEND_DIR.parent                      # Work/Janindu/
PREPROCESS_DIR  = BASE / "1_Preprocess"
RECOGNIZE_DIR   = BASE / "2_Recogniton"
PREPROCESS_MAIN = PREPROCESS_DIR / "MainPreProcess.py"
RECOGNIZE_MAIN  = RECOGNIZE_DIR  / "MainRecognize.py"

# The wrappers are thin subprocess launchers — they pick the right venv themselves.
# Flask server uses whatever Python launched app.py.
PYTHON = sys.executable

# ── Job state ─────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}   # job_id -> {status, log, results}
_jobs_lock = threading.Lock()

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


# =============================================================================
# STATIC FILES
# =============================================================================

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUTS_DIR), filename)


@app.route("/processes/<path:filename>")
def serve_process(filename):
    return send_from_directory(str(PROCESSES_DIR), filename)


# =============================================================================
# UPLOAD
# =============================================================================

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

@app.route("/api/upload", methods=["POST"])
def upload():
    """Receive uploaded images and save to Inputs/."""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files received"}), 400

    saved = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        dest = INPUTS_DIR / f.filename
        f.save(str(dest))
        saved.append(f.filename)

    return jsonify({"saved": saved, "count": len(saved)})


@app.route("/api/clear_inputs", methods=["POST"])
def clear_inputs():
    """Remove all files from Inputs/ folder."""
    for p in INPUTS_DIR.iterdir():
        if p.is_file():
            p.unlink()
    return jsonify({"ok": True})


@app.route("/api/list_inputs", methods=["GET"])
def list_inputs():
    imgs = sorted(p.name for p in INPUTS_DIR.iterdir()
                  if p.suffix.lower() in ALLOWED_EXT)
    return jsonify({"images": imgs})


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def _log(job_id: str, msg: str):
    with _jobs_lock:
        _jobs[job_id]["log"].append(msg)
    print(f"[{job_id[:8]}] {msg}")


def _run_stage(job_id: str,
               script: Path,
               script_dir: Path,
               extra_args: list[str] = None) -> bool:
    """Run a pipeline script as a subprocess, streaming stdout to the job log."""
    cmd = [PYTHON, str(script)] + (extra_args or [])
    _log(job_id, f"> Running: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _log(job_id, line)
        proc.wait()
        if proc.returncode != 0:
            _log(job_id, f"FAILED Script exited with code {proc.returncode}")
            return False
        _log(job_id, "OK Stage complete")
        return True
    except Exception as exc:
        _log(job_id, f"FAILED Exception: {exc}")
        return False


def _pipeline_worker(job_id: str, image_names: list[str]):
    """
    Background thread:
      1. Copy selected images -> Inputs/  (already done by upload)
      2. Call wrapper_preprocess.py  ->  output in Processes/
      3. Call wrapper_recognize.py   ->  output in Outputs/
      4. Call frontend_reporting.py  ->  builds the master report
    """
    def update(status, **kw):
        with _jobs_lock:
            _jobs[job_id]["status"] = status
            _jobs[job_id].update(kw)

    update("running_preprocess")
    _log(job_id, "=" * 60)
    _log(job_id, f"Pipeline started  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(job_id, f"Images: {', '.join(image_names)}")
    _log(job_id, "=" * 60)

    # ── Stage 1: Preprocess (uses 1_Preprocess venv via wrapper) ───────────────
    _log(job_id, "\n[Stage 1] Preprocessing …")
    ok = _run_stage(
        job_id,
        FRONTEND_DIR / "wrapper_preprocess.py",
        FRONTEND_DIR,
        ["--inputs",  str(INPUTS_DIR),
         "--outputs", str(PROCESSES_DIR),
         "--images"] + image_names,
    )
    if not ok:
        update("failed", error="Preprocessing failed. Check logs.")
        return

    # ── Stage 2: Recognition (uses 2_Recogniton venv via wrapper) ────────────
    update("running_recognition")
    _log(job_id, "\n[Stage 2] Recognition …")
    ok = _run_stage(
        job_id,
        FRONTEND_DIR / "wrapper_recognize.py",
        FRONTEND_DIR,
        ["--processes", str(PROCESSES_DIR),
         "--outputs",   str(OUTPUTS_DIR),
         "--inputs",    str(INPUTS_DIR),
         "--images"] + image_names,
    )
    if not ok:
        update("failed", error="Recognition failed. Check logs.")
        return

    # ── Stage 3: Generate master report ──────────────────────────────────────
    _log(job_id, "\n[Stage 3] Generating report …")
    ok = _run_stage(
        job_id,
        FRONTEND_DIR / "frontend_reporting.py",
        FRONTEND_DIR,
        ["--outputs", str(OUTPUTS_DIR),
         "--images"] + image_names,
    )
    if not ok:
        _log(job_id, "⚠ Report generation had errors (results may still be partial).")

    # ── Collect results ───────────────────────────────────────────────────────
    results = _collect_results(image_names)
    update("done", results=results)
    _log(job_id, "\nOK Pipeline complete!")


def _collect_results(image_names: list[str]) -> list[dict]:
    """Read the per-image summary JSONs produced by frontend_reporting."""
    results = []
    for name in image_names:
        stem = Path(name).stem
        summary_path = OUTPUTS_DIR / stem / "frontend_summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            results.append({
                "stem": stem,
                "fname": name,
                "predicted_text": "",
                "splits": [],
                "error": "No output generated",
            })
    return results


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Start the pipeline for a list of image names already in Inputs/."""
    body = request.get_json(silent=True) or {}
    images = body.get("images", [])
    if not images:
        # Use everything in Inputs/ if not specified
        images = sorted(p.name for p in INPUTS_DIR.iterdir()
                        if p.suffix.lower() in ALLOWED_EXT)
    if not images:
        return jsonify({"error": "No images to process"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "log": [],
            "results": [],
            "images": images,
        }

    t = threading.Thread(target=_pipeline_worker, args=(job_id, images), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    # Only return recent log lines to keep response small
    log_offset = int(request.args.get("log_offset", 0))
    return jsonify({
        "status":  job["status"],
        "log":     job["log"][log_offset:],
        "results": job.get("results", []),
        "error":   job.get("error", ""),
    })


# =============================================================================
# IMAGE SERVING (for report thumbnails etc.)
# =============================================================================

@app.route("/api/image/<path:rel_path>")
def serve_image(rel_path: str):
    """Serve images from frontend directory."""
    # 1. Try directly as relative to FRONTEND_DIR
    p = FRONTEND_DIR / rel_path
    if p.exists() and p.is_file():
        return send_file(str(p))
    
    # 2. Try as relative to standard subfolders (backward compatibility)
    for base in (OUTPUTS_DIR, PROCESSES_DIR, INPUTS_DIR):
        p = base / rel_path
        if p.exists() and p.is_file():
            return send_file(str(p))
            
    return "Not found", 404


# =============================================================================
# REPORT
# =============================================================================

@app.route("/api/report")
def master_report():
    """Return all available result summaries."""
    results = []
    for stem_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not stem_dir.is_dir():
            continue
        sp = stem_dir / "frontend_summary.json"
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                results.append(json.load(f))
    return jsonify({"results": results})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Sinhala OCR  |  Frontend Server")
    print(f"  Inputs    : {INPUTS_DIR}")
    print(f"  Processes : {PROCESSES_DIR}")
    print(f"  Outputs   : {OUTPUTS_DIR}")
    print(f"  Python    : {PYTHON}")
    print("=" * 60)
    print("\n  Open in browser: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

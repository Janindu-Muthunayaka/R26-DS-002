"""
LAYER 4A — MAT skeleton drawn on the binarized title, for SHOW ONLY.

Nothing reads it: no model, no OCR, the title is unchanged. It is a picture —
the binarized strip with its medial-axis skeleton (skeletonize, as in
Work/Janindu/1_Preprocess/stage5_MAT.py) drawn over it in red — behind the
MAT button of the L4A box.

It costs the scan nothing:
  * it runs in a SEPARATE PROCESS at IDLE priority, so a scan running now
    always wins the CPU;
  * it starts only after /capture has built its reply;
  * one job at a time, from one queue;
  * nothing waits on it. When the picture is written, a thread in the server
    adds title_images['mat'] to the article and the MAT button appears.

`SINHALA_TITLE_MAT=off` disables it (core/config.py).

Worker:  python -m layers.l4a_title.mat   (JSON job on stdin, JSON lines out)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

SYSTEM_DIR = Path(__file__).resolve().parents[2]
SKELETON_RED = (0, 0, 220)       # BGR, as stage5_MAT.py
SKELETON_DILATE = 1              # 3x3 dilation so a 1 px line is visible


# ═══ WORKER SIDE (runs only in the background process) ══════════════════════

def skeleton_on_binarized(binarized_bgr):
    """The binarized strip with its skeleton drawn over it in red."""
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    grey = cv2.cvtColor(binarized_bgr, cv2.COLOR_BGR2GRAY)
    skeleton = skeletonize(grey < 128)
    skel = skeleton.astype(np.uint8) * 255
    if SKELETON_DILATE > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=SKELETON_DILATE)
    out = binarized_bgr.copy()
    out[skel > 0] = SKELETON_RED
    return out


def _worker_main():
    import cv2
    job = json.loads(sys.stdin.read() or '{}')
    for task in job.get('tasks', []):
        res = {'art': task.get('art'), 'suffix': task.get('suffix')}
        try:
            img = cv2.imread(task['bin'])
            if img is None:
                raise ValueError(f"cannot read {task['bin']}")
            cv2.imwrite(task['mat'], skeleton_on_binarized(img))
            res['mat'] = Path(task['mat']).name
        except Exception as e:
            res['error'] = f'{type(e).__name__}: {e}'
        sys.stdout.write(json.dumps(res) + '\n')
        sys.stdout.flush()


# ═══ SERVER SIDE (stdlib only; never blocks the caller) ═════════════════════

_jobs: "queue.Queue" = queue.Queue()
_thread = None
_lock = threading.Lock()


def _idle_popen_kwargs():
    if os.name == 'nt':
        return {'creationflags': 0x00000040 | 0x08000000}   # IDLE_PRIORITY_CLASS | CREATE_NO_WINDOW
    return {'preexec_fn': lambda: os.nice(19)}


def _articles(doc):
    return list(doc.articles) + list(getattr(doc, 'all_articles', []) or [])


def _tasks_for(doc, job_dir: Path):
    tasks = []
    for art in _articles(doc):
        for key, name in list((art.title_images or {}).items()):
            if key != 'bin' and not key.startswith('bin_'):
                continue
            stem = name[:-len('bin.png')] if name.endswith('bin.png') else Path(name).stem + '_'
            t = {'art': art.index, 'suffix': key[3:],             # '' or '_1', ...
                 'bin': str(job_dir / name), 'mat': str(job_dir / f'{stem}mat.png')}
            if t not in tasks:
                tasks.append(t)
    return tasks


def _apply(doc, results):
    for r in results:
        if not r.get('mat'):
            continue
        for art in _articles(doc):
            if art.index == r.get('art'):
                art.title_images[f"mat{r['suffix']}"] = r['mat']


def _loop():
    while True:
        doc, job_dir, tasks = _jobs.get()
        try:
            env = dict(os.environ, PYTHONPATH=str(SYSTEM_DIR) + os.pathsep + os.environ.get('PYTHONPATH', ''))
            p = subprocess.Popen([sys.executable, '-m', 'layers.l4a_title.mat'],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, cwd=str(SYSTEM_DIR), env=env,
                                 **_idle_popen_kwargs())
            out, _ = p.communicate(json.dumps({'tasks': tasks}).encode('utf-8'))
            _apply(doc, [json.loads(l) for l in out.decode('utf-8', 'replace').splitlines() if l.strip()])
        except Exception as e:                          # show-only: never take the server down
            print(f'[title MAT] background job failed: {type(e).__name__}: {e}')
        finally:
            _jobs.task_done()


def schedule(doc, job_dir) -> int:
    """Queue the skeleton picture for every binarized title of `doc`. Returns
    at once (the number of strips queued); the work happens in the background."""
    from core.config import TITLE_MAT_MODE
    global _thread
    if TITLE_MAT_MODE == 'off' or doc is None or job_dir is None:
        return 0
    tasks = _tasks_for(doc, Path(job_dir))
    if not tasks:
        return 0
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, name='title-mat', daemon=True)
            _thread.start()
    _jobs.put((doc, Path(job_dir), tasks))
    return len(tasks)


if __name__ == '__main__':
    _worker_main()

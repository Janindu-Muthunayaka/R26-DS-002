# delivery/laptop_player.py
# Speaks a speech plan (delivery/planner.py) on the laptop, so delivery can be
# tested without the phone (ADAPTIVE_VOICE_PLAN.md, Phase 1).
#
#   - gTTS (lang="si") makes one mp3 per segment, cached by a hash of the text
#     so replays are instant. Needs internet, like Google STT.
#   - ffplay plays each mp3 with atempo=<rate>, then waits pause_after_ms.
#     atempo time-stretches recorded audio, so rate only approximates
#     Android's setSpeechRate(), which changes the synthesis itself.
#   - The next segment is synthesized while the current one plays, so the
#     network call doesn't add a gap between segments.
#   - Pressing Enter at any moment stops playback (Windows console only).
#   - (Phase 3) play_plan(start=, end=) plays part of a plan: app.py uses it
#     to continue a stopped answer and to pause at "continue?" check-ins.
#
# Demo (from voice_interaction/):
#   venv\Scripts\python.exe -m delivery.laptop_player --rate 0.8

import argparse
import glob
import hashlib
import msvcrt
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from gtts import gTTS

try:
    from delivery.planner import plan_speech
except ImportError:
    from layers.l7_personalization.delivery.planner import plan_speech

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tts_cache"
)
PREVIEW_PAUSE_MS = 500
POLL_SEC = 0.05

# Fixed answer for the Phase 1 demo: three spoken ordinals (one segment each)
# and a decimal number that must not be split.
SAMPLE_ANSWER = (
    "මෙම ලිපියේ ප්‍රධාන කරුණු තුනක් තියෙනවා. "
    "පළමුව, ඉන්ධන මිල රුපියල් 25.50 කින් අඩු කළා. "
    "දෙවනුව, නව මිල අද රාත්‍රියේ සිට ක්‍රියාත්මක වෙනවා. "
    "තෙවනුව, බස් ගාස්තු අඩු කිරීමට සාකච්ඡා කෙරෙනවා."
)


def find_ffplay():
    """
    Locates ffplay.exe. Checks FFPLAY_PATH, then PATH, then winget's package
    folder — so a fresh `winget install Gyan.FFmpeg` works without reopening
    the terminal.
    """
    override = os.environ.get("FFPLAY_PATH")
    if override and os.path.isfile(override):
        return override
    on_path = shutil.which("ffplay")
    if on_path:
        return on_path
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet",
                           "Packages", "Gyan.FFmpeg*", "*", "bin", "ffplay.exe")
    found = sorted(glob.glob(pattern))
    if found:
        return found[-1]
    raise FileNotFoundError(
        "ffplay not found. Install FFmpeg (winget install Gyan.FFmpeg) "
        "or set FFPLAY_PATH to ffplay.exe."
    )


def synthesize(text):
    """Returns the path of an mp3 of `text` spoken in Sinhala, using the cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}.mp3")
    if not os.path.exists(path):
        tmp = path + ".part"
        gTTS(text, lang="si").save(tmp)
        os.replace(tmp, path)   # never leave a half-written file in the cache
    return path


def _enter_pressed():
    """True if Enter was pressed since the last check; other keys are ignored."""
    pressed = False
    while msvcrt.kbhit():
        if msvcrt.getwch() in ("\r", "\n"):
            pressed = True
    return pressed


def _flush_keys():
    """Drops key presses made before playback (e.g. an extra Enter while the
    microphone was listening), so they cannot stop the new answer at once."""
    while msvcrt.kbhit():
        msvcrt.getwch()


def _play_file(ffplay, path, rate):
    """Plays one mp3 at `rate`. Returns True if the user stopped it."""
    proc = subprocess.Popen(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
         "-af", f"atempo={rate}", path],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    while proc.poll() is None:
        if _enter_pressed():
            proc.kill()
            proc.wait()
            return True
        time.sleep(POLL_SEC)
    return False


def _wait(ms):
    """Silent pause of `ms` milliseconds. Returns True if the user stopped it."""
    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        if _enter_pressed():
            return True
        time.sleep(POLL_SEC)
    return False


def _new_result(plan, start):
    return {"completed": False, "stopped_at": None, "stopped_in": None,
            "segments_played": start, "start": start,
            "n_segments": len(plan["segments"]), "finished_at": None}


def _range_end(plan, end):
    n = len(plan["segments"])
    return n if end is None else max(0, min(int(end), n))


def play_plan(plan, ffplay=None, on_segment=None, start=0, end=None):
    """
    Speaks a speech plan: the preview (if any, and only when starting at the
    first segment), then each segment followed by its pause.

    start — index of the first segment to play (Phase 3: continue a stopped
            answer, or the part after a "continue?" check-in)
    end   — stop BEFORE this segment index (a checkpoint); None = to the end.
            The pause after the last segment before `end` is skipped: the
            caller asks its check-in question there.
    on_segment(index, segment) is called just before each segment plays.

    Returns:
      {
        "completed":       True if the plan's last segment was played,
        "stopped_at":      index of the segment that was playing (or had just
                           finished, if stopped during its pause); None if
                           not stopped by the user, or stopped in the preview,
        "stopped_in":      "preview" | "segment" | "pause" | "checkpoint" | None
                           ("checkpoint" = reached `end`, not a user stop),
        "segments_played": index of the first segment NOT heard to the end
                           (= number of segments heard when start is 0) — a
                           stopped answer resumes from here,
        "start":           the `start` passed in,
        "n_segments":      len(plan["segments"]),
        "finished_at":     ISO time playback ended (last_spoken.finished_at;
                           the 30-minute REPEAT expiry is measured from it)
      }
    """
    ffplay = ffplay or find_ffplay()
    rate = plan["rate"]
    segments = plan["segments"]
    stop = _range_end(plan, end)
    result = _new_result(plan, start)

    preview = plan.get("preview") if start == 0 else None
    texts = ([preview] if preview else []) + [s["text"] for s in segments[start:stop]]
    offset = 1 if preview else 0
    _flush_keys()

    # Synthesize ahead in the background; play strictly in order.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(synthesize, t) for t in texts]
        try:
            if offset:
                if _play_file(ffplay, futures[0].result(), rate) or _wait(PREVIEW_PAUSE_MS):
                    result["stopped_in"] = "preview"
                    return result

            for k, i in enumerate(range(start, stop)):
                seg = segments[i]
                if on_segment:
                    on_segment(i, seg)
                if _play_file(ffplay, futures[k + offset].result(), rate):
                    result.update(stopped_at=i, stopped_in="segment")
                    return result
                result["segments_played"] = i + 1
                if i < stop - 1 and seg["pause_after_ms"] and _wait(seg["pause_after_ms"]):
                    result.update(stopped_at=i, stopped_in="pause")
                    return result

            if stop < len(segments):
                result["stopped_in"] = "checkpoint"
            else:
                result["completed"] = True
            return result
        finally:
            for f in futures:
                f.cancel()   # skip synthesis of segments that won't be played
            result["finished_at"] = datetime.now().isoformat(timespec="seconds")


def play_plan_silent(plan, ffplay=None, on_segment=None, start=0, end=None):
    """
    Same arguments and result as play_plan(), but nothing is synthesized or
    played and nobody can stop it: every segment in the range is "heard".
    For `app.py --no-audio` (no internet / FFmpeg needed) and for tests.
    """
    stop = _range_end(plan, end)
    result = _new_result(plan, start)
    for i in range(start, stop):
        if on_segment:
            on_segment(i, plan["segments"][i])
        result["segments_played"] = i + 1
    if stop < len(plan["segments"]):
        result["stopped_in"] = "checkpoint"
    else:
        result["completed"] = True
    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Sinhala in the Windows console

    parser = argparse.ArgumentParser(description="Play a Sinhala answer as a speech plan.")
    parser.add_argument("--rate", type=float, default=1.0, help="speech rate, 1.0 = normal (0.6-1.8)")
    parser.add_argument("--style", default="StepByStep", choices=["Simple", "Detailed", "StepByStep"])
    parser.add_argument("--pause-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--text", default=SAMPLE_ANSWER, help="Sinhala text to speak")
    args = parser.parse_args()

    plan = plan_speech(args.text, args.style, {
        "rate": args.rate, "pause_scale": args.pause_scale, "chunk_size": args.chunk_size,
    })
    print(f"rate {plan['rate']} | style {args.style} | {len(plan['segments'])} segments"
          f" | checkpoints {plan['checkpoints']} (not asked in Phase 1)")
    if plan["preview"]:
        print(f"preview: {plan['preview']}")
    print("Press Enter at any time to stop.\n")

    def show(i, seg):
        print(f"  [{i}] {seg['text']}  (pause {seg['pause_after_ms']} ms)")

    try:
        result = play_plan(plan, on_segment=show)
    except FileNotFoundError as e:
        print(e)
        return 1
    except Exception as e:   # gTTS network errors etc.
        print(f"Playback failed: {e}")
        return 1

    if result["completed"]:
        print(f"\nCompleted: all {result['n_segments']} segments played.")
    else:
        where = "the preview" if result["stopped_in"] == "preview" else \
            f"segment {result['stopped_at']} ({result['stopped_in']})"
        print(f"\nStopped during {where}; {result['segments_played']} of "
              f"{result['n_segments']} segments fully played.")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

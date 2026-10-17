# history.py
# Per-session chat history, persisted as one JSON file per session:
#     chat_sessions/session_<session_id>.json
import hashlib
import json
import os
import re
import threading
from datetime import datetime

MAX_TURNS = 4  # how many recent turns to keep per session
DEFAULT_SESSION_ID = "default"
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_sessions")

_lock = threading.RLock()
_warned_missing = False
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _resolve(session_id) -> str:
    """Returns a filename-safe session id. Missing id -> 'default' (+ one warning).
    Ids with unusual characters are hashed so they can never escape HISTORY_DIR."""
    global _warned_missing
    sid = str(session_id).strip() if session_id is not None else ""
    if not sid:
        if not _warned_missing:
            print("[history] WARNING: no session_id received - using the shared "
                  f"'{DEFAULT_SESSION_ID}' session.")
            _warned_missing = True
        return DEFAULT_SESSION_ID
    if _SAFE_ID.fullmatch(sid):
        return sid
    return "s_" + hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]


def history_path(session_id=None) -> str:
    return os.path.join(HISTORY_DIR, f"session_{_resolve(session_id)}.json")


def _read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        turns = data.get("turns", []) if isinstance(data, dict) else []
        return turns if isinstance(turns, list) else []
    except (OSError, ValueError) as e:
        print(f"[history] could not read {path}: {e!r} - treating as empty")
        return []


def get_history(session_id=None) -> list[dict]:
    """Returns the kept turns of this session, oldest first."""
    with _lock:
        return _read(history_path(session_id))


def add_turn(question: str, standalone_question: str, answer: str,
             session_id=None) -> None:
    with _lock:
        sid = _resolve(session_id)
        path = history_path(sid)
        turns = _read(path)
        turns.append({
            "question": question,                        # what the user actually asked
            "standalone_question": standalone_question,  # rewritten, self-contained version
            "answer": answer,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        turns = turns[-MAX_TURNS:]  # oldest turns are dropped
        try:
            os.makedirs(HISTORY_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"session_id": sid, "turns": turns}, f,
                          ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[history] could not save {path}: {e!r}")


def clear_history(session_id=None) -> None:
    """Deletes this session's history file (other sessions are untouched)."""
    with _lock:
        path = history_path(session_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"[history] could not delete {path}: {e!r}")

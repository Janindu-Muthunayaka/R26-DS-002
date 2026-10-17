# services/rag/history.py
import hashlib
import json
import os
import re
import threading
from datetime import datetime

MAX_TURNS = 4
DEFAULT_SESSION_ID = "default"
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_sessions")

_lock = threading.RLock()
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _resolve(session_id) -> str:
    sid = str(session_id).strip() if session_id is not None else ""
    if not sid:
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
    except (OSError, ValueError):
        return []


def get_history(session_id=None) -> list[dict]:
    with _lock:
        return _read(history_path(session_id))


def add_turn(question: str, standalone_question: str, answer: str,
             session_id=None) -> None:
    with _lock:
        sid = _resolve(session_id)
        path = history_path(sid)
        turns = _read(path)
        turns.append({
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        turns = turns[-MAX_TURNS:]
        try:
            os.makedirs(HISTORY_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"session_id": sid, "turns": turns}, f,
                          ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[history] could not save {path}: {e!r}")


def format_history_for_prompt(turns: list[dict], max_turns: int = 2,
                              answer_chars: int = 400) -> str:
    refusal = "ප්‍රමාණවත් තොරතුරු නොමැත"
    usable = []
    for t in (turns or []):
        ans = (t.get("answer") or t.get("assistant_answer") or t.get("answer_si") or "").strip()
        if refusal not in ans:
            usable.append(t)
    usable = usable[-max_turns:]
    lines = []
    for t in usable:
        q = (t.get("standalone_question") or t.get("question") or
             t.get("user_query") or t.get("query") or "").strip()
        a = (t.get("answer") or t.get("assistant_answer") or t.get("answer_si") or "").strip()
        if len(a) > answer_chars:
            a = a[:answer_chars].rstrip() + "…"
        if q:
            lines.append(f"පරිශීලකයා: {q}")
        if a:
            lines.append(f"ඔබේ පිළිතුර: {a}")
    return "\n".join(lines)

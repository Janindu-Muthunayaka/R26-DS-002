# delivery/voice_profile.py
# Per-user delivery settings: speech rate, pause scale, chunk size, the
# counters the feedback rules need, and the last thing spoken (for REPEAT).
#
# Stored in the "voice_profiles" table of data/db.json — the same TinyDB file
# personalization/logger.py writes interaction logs to. Profiles are plain
# dicts so delivery/feedback.py can update them without touching storage.

from datetime import datetime, timedelta

from tinydb import TinyDB, Query

try:
    from personalization.logger import DB_PATH
except ImportError:
    from layers.l7_personalization.personalization.logger import DB_PATH

TABLE = "voice_profiles"
LOG_TABLE = "delivery_log"    # one row per user turn (delivery/turn.py)

# A last_spoken older than this is "nothing to repeat" (same as the full
# system reader's 30-minute session lifetime).
REPEAT_MAX_AGE = timedelta(minutes=30)

DEFAULTS = {
    "rate": 1.0,
    "pause_scale": 1.0,
    "chunk_size": 3,
    "clean_streak": 0,        # answers finished in a row with no delivery complaint
    "long_history": [],       # stopped? (True/False) for the last 3 long answers
    "counts": {"repeat": 0, "slower": 0, "faster": 0, "stopped": 0},
    "last_spoken": None,      # {text, kind, style, finished_at, completed, resume_at, turns_since}
}

_db_cache = {}


def _table(db_path=None):
    path = db_path or DB_PATH
    if path not in _db_cache:
        # Default TinyDB settings, same as logger.py: Sinhala is stored as
        # \u escapes, so the file stays ASCII and both readers agree on it.
        _db_cache[path] = TinyDB(path)
    return _db_cache[path].table(TABLE)


def new_profile():
    """A fresh profile with default settings (deep-copied, safe to mutate)."""
    return {**DEFAULTS, "long_history": [], "counts": dict(DEFAULTS["counts"])}


def get_profile(user_id, db_path=None):
    """The user's stored profile, or defaults for a new user. Never None."""
    row = _table(db_path).get(Query().user_id == user_id)
    profile = new_profile()
    if row:
        stored = {k: v for k, v in row.items() if k != "user_id"}
        profile.update(stored)
        profile["counts"] = {**DEFAULTS["counts"], **stored.get("counts", {})}
    return profile


def save_profile(user_id, profile, db_path=None):
    _table(db_path).upsert({**profile, "user_id": user_id}, Query().user_id == user_id)


def log_delivery(record, db_path=None):
    """Appends one turn's delivery record to the "delivery_log" table."""
    path = db_path or DB_PATH
    _table(path)                       # opens (and caches) the file
    _db_cache[path].table(LOG_TABLE).insert(record)
    return record


def set_last_spoken(profile, text, kind, style, finished_at, completed=True, resume_at=None):
    """
    Records what was just spoken, overwriting the previous one.
    kind:        "answer" | "article"
    finished_at: ISO time the player finished or was stopped (the 30-minute
                 REPEAT_MAX_AGE is measured from it)
    completed:   False if the user stopped it (or it paused at a check-in)
    resume_at:   when not completed, the index of the first segment not yet
                 heard — a later slower/faster request continues from there
    turns_since: user turns since it was spoken; 0 = the next turn is "the
                 very next turn", the only one where REPEAT means "didn't hear"
    """
    if isinstance(finished_at, datetime):
        finished_at = finished_at.isoformat(timespec="seconds")
    profile["last_spoken"] = {"text": text, "kind": kind, "style": style,
                              "finished_at": finished_at, "completed": bool(completed),
                              "resume_at": None if completed else resume_at,
                              "turns_since": 0}
    return profile


def count_turn(profile):
    """One more user turn since last_spoken. Call once per user turn
    (feedback.apply_turn does it for every turn it reads)."""
    last = profile.get("last_spoken")
    if last:
        last["turns_since"] = last.get("turns_since", 0) + 1
    return profile


def is_next_turn(profile):
    """True if this user turn is the very next one after last_spoken.
    Profiles saved before Phase 3 have no turns_since: never the next turn."""
    last = profile.get("last_spoken")
    return bool(last) and last.get("turns_since") == 0


def seconds_since_spoken(profile, now):
    """Seconds since last_spoken finished, or None if nothing was spoken."""
    last = profile.get("last_spoken")
    if not last or not last.get("finished_at"):
        return None
    return (now - datetime.fromisoformat(last["finished_at"])).total_seconds()


def repeatable(profile, now):
    """last_spoken if it can still be replayed, else None ("nothing to repeat")."""
    age = seconds_since_spoken(profile, now)
    if age is None or age > REPEAT_MAX_AGE.total_seconds():
        return None
    return profile["last_spoken"]

# personalization/users.py
# The user registry: who the users are, their base style weights, and the
# words shown about them in the UI.
#
# Stored in the "user_profiles" table of data/db.json (the same TinyDB file
# as the interaction logs), so it survives restarts.
#
# RULES
#   * Three built-in users (user_001..003), one per style, base split 50/25/25.
#     They can be reset but never deleted.
#   * Added users get the next number. Ids stay contiguous: deleting user_004
#     renumbers user_005 -> user_004 and so on, moving everything the system
#     keeps under that id (style model state, voice profile, logs).
#   * Base weights are fractions of 1 over STYLE_CLASSES. The style model
#     starts a user from them, resets to them, and never lets the user's
#     dominant style fall below its base share.

import re
import threading

from tinydb import Query

try:
    from personalization.logger import db
except ImportError:
    from layers.l7_personalization.personalization.logger import db

STYLE_CLASSES = ["Simple", "Detailed", "StepByStep"]

BUILTIN_USERS = [
    {
        "user_id": "user_001",
        "name": "User 001",
        "tag": "Simple Preference",
        "description": "For people who prefer simple, concise explanations in clear everyday language without jargon.",
        "base_weights": {"Simple": 0.50, "Detailed": 0.25, "StepByStep": 0.25},
    },
    {
        "user_id": "user_002",
        "name": "User 002",
        "tag": "Detailed Preference",
        "description": "For people interested in depth and detail, comprehensive context, and thorough background.",
        "base_weights": {"Simple": 0.25, "Detailed": 0.50, "StepByStep": 0.25},
    },
    {
        "user_id": "user_003",
        "name": "User 003",
        "tag": "Step-by-Step Preference",
        "description": "For people who follow structured, numbered steps, sequential walkthroughs, and clear progression.",
        "base_weights": {"Simple": 0.25, "Detailed": 0.25, "StepByStep": 0.50},
    },
]
BUILTIN_IDS = {u["user_id"] for u in BUILTIN_USERS}

TAG_FOR_STYLE = {
    "Simple": "Simple Preference",
    "Detailed": "Detailed Preference",
    "StepByStep": "Step-by-Step Preference",
}

_table = db.table("user_profiles")
_lock = threading.RLock()
_ID_RE = re.compile(r"^user_(\d+)$")


def user_number(user_id):
    m = _ID_RE.match(user_id or "")
    return int(m.group(1)) if m else None


def make_user_id(n):
    return f"user_{n:03d}"


def _default_name(user_id):
    return "User " + user_id.split("_", 1)[1]


def normalize_weights(weights):
    """Fractions over STYLE_CLASSES summing to 1. Accepts percentages too."""
    w = {c: float((weights or {}).get(c, 0) or 0) for c in STYLE_CLASSES}
    if any(v < 0 for v in w.values()):
        raise ValueError("weights must not be negative")
    total = sum(w.values())
    if total <= 0:
        raise ValueError("weights must not all be zero")
    return {c: round(v / total, 4) for c, v in w.items()}


def dominant_style(weights):
    return max(STYLE_CLASSES, key=lambda c: weights.get(c, 0))


def _seed():
    """Make sure the three built-ins exist with full records. Older db.json
    files hold only {user_id}; those are filled in, never duplicated."""
    with _lock:
        U = Query()
        for b in BUILTIN_USERS:
            row = _table.get(U.user_id == b["user_id"])
            if row is None:
                _table.insert({**b, "builtin": True})
            elif not row.get("base_weights") or not row.get("builtin"):
                _table.update({**b, "builtin": True, **{k: row[k] for k in ("name", "tag", "description") if row.get(k)}},
                              U.user_id == b["user_id"])


def list_users():
    """Every user, in numeric order."""
    _seed()
    rows = [dict(r) for r in _table.all() if user_number(r.get("user_id")) is not None]
    rows.sort(key=lambda r: user_number(r["user_id"]))
    for r in rows:
        r["builtin"] = r["user_id"] in BUILTIN_IDS
        r.setdefault("name", _default_name(r["user_id"]))
        r.setdefault("tag", "")
        r.setdefault("description", "")
    return rows


def get_user(user_id):
    for u in list_users():
        if u["user_id"] == user_id:
            return u
    return None


def get_base_weights(user_id):
    """Base split for a user, or None for an id that is not registered."""
    u = get_user(user_id)
    return dict(u["base_weights"]) if u and u.get("base_weights") else None


def add_user(base_weights, description="", name="", tag=""):
    """Register a new user under the next free number. Returns the record."""
    weights = normalize_weights(base_weights)
    with _lock:
        users = list_users()
        n = max((user_number(u["user_id"]) for u in users), default=0) + 1
        uid = make_user_id(n)
        record = {
            "user_id": uid,
            "name": (name or "").strip() or _default_name(uid),
            "tag": (tag or "").strip() or TAG_FOR_STYLE[dominant_style(weights)],
            "description": (description or "").strip(),
            "base_weights": weights,
            "builtin": False,
        }
        _table.insert(record)
        return record


def delete_user(user_id, on_delete, on_rename):
    """Delete an added user and close the gap in the numbering.

    on_delete(uid)        drops everything stored under uid (model state etc.)
    on_rename(old, new)   moves everything stored under old to new
    Returns the list of (old_id, new_id) renames that were applied.
    """
    if user_id in BUILTIN_IDS:
        raise PermissionError("the built-in users cannot be deleted")
    with _lock:
        U = Query()
        if _table.get(U.user_id == user_id) is None:
            raise KeyError(user_id)
        _table.remove(U.user_id == user_id)
        on_delete(user_id)

        gone = user_number(user_id)
        renames = []
        for u in list_users():
            n = user_number(u["user_id"])
            if n > gone:
                old, new = u["user_id"], make_user_id(n - 1)
                fields = {"user_id": new}
                if u.get("name") == _default_name(old):
                    fields["name"] = _default_name(new)
                _table.update(fields, U.user_id == old)
                on_rename(old, new)
                renames.append((old, new))
        return renames

# personalization/logger.py
# Step 1 — Logs each interaction into db.json (TinyDB)

import os
from datetime import datetime
from tinydb import TinyDB, Query

# data/ is a sibling of personalization/, both under voice_interaction/
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "db.json"
)

db = TinyDB(DB_PATH)
interaction_logs = db.table("interaction_logs")


def log_interaction(user_id, result, retrieved_chunk_id=None, style_class=None):
    """
    Takes the dict returned by detect_intent_approach1() and appends a full
    record to interaction_logs. Returns the record (including its timestamp,
    which acts as the record's identifier for later updates).
    """
    record = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "sinhala_input": result.get("sinhala_input"),
        "english_translation": result.get("english_translation"),
        "intent": result.get("intent"),
        "personalization_flags": result.get("personalization_flags", {}),
        "retrieved_chunk_id": retrieved_chunk_id,
        "style_class": style_class,
        "corrected": False   # flips to True if Step 4 later relabels this record
    }
    interaction_logs.insert(record)
    return record


def get_last_interaction(user_id):
    """Returns the most recent logged interaction for this user, or None."""
    User = Query()
    records = interaction_logs.search(User.user_id == user_id)
    return records[-1] if records else None


def update_last_interaction_style(user_id, style_class):
    """Updates the style_class field on the user's most recent log entry
    (normal path — this record's own predicted style)."""
    User = Query()
    matches = interaction_logs.search(User.user_id == user_id)
    if not matches:
        return None
    last = matches[-1]
    interaction_logs.update(
        {"style_class": style_class},
        (User.user_id == user_id) & (User.timestamp == last["timestamp"])
    )
    return style_class


def update_interaction_style_by_timestamp(user_id, timestamp, corrected_style_class):
    """Relabels a SPECIFIC past record (identified by its exact timestamp)
    as corrected, and marks corrected=True. Used by Step 4 when the user's
    next turn signals the previous prediction was wrong."""
    User = Query()
    interaction_logs.update(
        {"style_class": corrected_style_class, "corrected": True},
        (User.user_id == user_id) & (User.timestamp == timestamp)
    )
    return corrected_style_class


if __name__ == "__main__":
    import json
    print("=" * 60)
    print("📜 TINYDB INTERACTION LOGS HISTORY")
    print("=" * 60)
    records = interaction_logs.all()
    if not records:
        print("No logs recorded yet.")
    for idx, log in enumerate(records, 1):
        print(f"\nRecord #{idx}:")
        print(json.dumps(log, indent=2, ensure_ascii=False))
    print("-" * 60)

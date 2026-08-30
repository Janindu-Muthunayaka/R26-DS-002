# personalization/logger.py
# Step 1 — Logs each interaction into db.json (TinyDB) and interaction_logs.csv (CSV)

import os
import csv
import json
from datetime import datetime
from tinydb import TinyDB, Query

# data/ is a sibling of personalization/, both under voice_interaction/
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)
DB_PATH = os.path.join(DATA_DIR, "db.json")
CSV_PATH = os.path.join(DATA_DIR, "interaction_logs.csv")

db = TinyDB(DB_PATH)
interaction_logs = db.table("interaction_logs")

CSV_COLUMNS = [
    "id",
    "user_id",
    "timestamp",
    "sinhala_input",
    "english_translation",
    "intent",
    "personalization_flags",
    "retrieved_chunk_id",
    "style_class",
    "corrected"
]


def export_logs_to_csv(csv_path=CSV_PATH):
    """
    Exports all records from TinyDB interaction_logs to a clean, readable CSV file.
    Uses 'utf-8-sig' encoding so Sinhala Unicode text opens properly in Microsoft Excel,
    Google Sheets, and all text editors.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    records = interaction_logs.all()
    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for idx, rec in enumerate(records, 1):
            flags = rec.get("personalization_flags", {})
            flags_str = json.dumps(flags, ensure_ascii=False) if flags else "{}"
            row = {
                "id": rec.doc_id if hasattr(rec, "doc_id") else idx,
                "user_id": rec.get("user_id", ""),
                "timestamp": rec.get("timestamp", ""),
                "sinhala_input": rec.get("sinhala_input", ""),
                "english_translation": rec.get("english_translation", ""),
                "intent": rec.get("intent", ""),
                "personalization_flags": flags_str,
                "retrieved_chunk_id": rec.get("retrieved_chunk_id", "") or "",
                "style_class": rec.get("style_class", "") or "",
                "corrected": rec.get("corrected", False),
            }
            writer.writerow(row)
    return len(records)


def import_csv_to_db(csv_path=CSV_PATH):
    """
    Reads the CSV file and synchronizes the records back into db.json interaction_logs.
    Useful when manual adjustments or bulk edits were made in the CSV file.
    """
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, mode="r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            flags_raw = row.get("personalization_flags", "{}")
            try:
                flags = json.loads(flags_raw) if flags_raw else {}
            except Exception:
                flags = {}
            rec = {
                "user_id": row.get("user_id", ""),
                "timestamp": row.get("timestamp", ""),
                "sinhala_input": row.get("sinhala_input", ""),
                "english_translation": row.get("english_translation", ""),
                "intent": row.get("intent", ""),
                "personalization_flags": flags,
                "retrieved_chunk_id": row.get("retrieved_chunk_id") or None,
                "style_class": row.get("style_class") or None,
                "corrected": str(row.get("corrected", "False")).strip().lower() in ("true", "1", "yes"),
            }
            records.append(rec)

    # Re-populate interaction_logs table
    interaction_logs.truncate()
    if records:
        interaction_logs.insert_multiple(records)
    return len(records)


def log_interaction(user_id, result, style_class=None):
    """
    Takes the dict returned by detect_intent_approach1() and appends a full
    record to interaction_logs in db.json AND syncs to interaction_logs.csv.
    Returns the record.
    """
    record = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "sinhala_input": result.get("sinhala_input"),
        "english_translation": result.get("english_translation"),
        "intent": result.get("intent"),
        "personalization_flags": result.get("personalization_flags", {}),
        # Placeholder — Component 3 will populate this once integrated.
        # No longer supplied by the user; the reader knows which chunk is
        # loaded, the user should never have to say so.
        "retrieved_chunk_id": None,
        "style_class": style_class,
        "corrected": False   # flips to True if Step 4 later relabels this record
    }
    interaction_logs.insert(record)
    export_logs_to_csv()
    return record


def get_last_interaction(user_id):
    """Returns the most recent logged interaction for this user, or None."""
    User = Query()
    records = interaction_logs.search(User.user_id == user_id)
    return records[-1] if records else None


def update_last_interaction_style(user_id, style_class):
    """Updates the style_class field on the user's most recent log entry
    in db.json and updates the CSV file."""
    User = Query()
    matches = interaction_logs.search(User.user_id == user_id)
    if not matches:
        return None
    last = matches[-1]
    interaction_logs.update(
        {"style_class": style_class},
        (User.user_id == user_id) & (User.timestamp == last["timestamp"])
    )
    export_logs_to_csv()
    return style_class


def update_interaction_style_by_timestamp(user_id, timestamp, corrected_style_class):
    """Relabels a SPECIFIC past record (identified by its exact timestamp)
    as corrected, marks corrected=True, and updates both db.json and CSV."""
    User = Query()
    interaction_logs.update(
        {"style_class": corrected_style_class, "corrected": True},
        (User.user_id == user_id) & (User.timestamp == timestamp)
    )
    export_logs_to_csv()
    return corrected_style_class


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = sys.argv[1:]
    
    if "--import-csv" in args:
        count = import_csv_to_db()
        print(f"[IMPORT] Imported {count} records from CSV into {DB_PATH}")
    else:
        count = export_logs_to_csv()
        print(f"[SYNC] Synced {count} records from {DB_PATH} to {CSV_PATH}")

    print("=" * 60)
    print(f"INTERACTION LOGS ({count} records in db.json & interaction_logs.csv)")
    print("=" * 60)
    records = interaction_logs.all()
    if not records:
        print("No logs recorded yet.")
    for idx, log in enumerate(records, 1):
        print(f"Record #{idx} [{log.get('user_id')} | {log.get('timestamp')} | Style: {log.get('style_class')}]:")
        print(f"  Sinhala:     {log.get('sinhala_input')}")
        print(f"  Translation: {log.get('english_translation')}")
        print(f"  Intent:      {log.get('intent')}")
        print(f"  Flags:       {log.get('personalization_flags')}")
        print(f"  Corrected:   {log.get('corrected')}")
        print("-" * 60)



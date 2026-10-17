"""
sync_logs.py — Synchronize interaction history between TinyDB (db.json) and CSV (interaction_logs.csv)

Usage:
    python sync_logs.py                  # Exports db.json -> interaction_logs.csv
    python sync_logs.py --import-csv     # Imports interaction_logs.csv -> db.json
    python sync_logs.py --status         # Shows summary count in both files
"""

import sys
import os

# Set stdout to utf-8 safely for Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from personalization.logger import (
    DB_PATH,
    CSV_PATH,
    export_logs_to_csv,
    import_csv_to_db,
    interaction_logs
)


def main():
    args = sys.argv[1:]
    
    if "--import-csv" in args:
        print(f"Reading from CSV: {CSV_PATH}")
        count = import_csv_to_db()
        print(f"Successfully updated {DB_PATH} with {count} records from CSV.")
    elif "--status" in args:
        total = len(interaction_logs.all())
        csv_exists = os.path.exists(CSV_PATH)
        print(f"Database (db.json): {total} logged interactions")
        print(f"CSV file ({CSV_PATH}): {'Present' if csv_exists else 'Not found'}")
    else:
        print(f"Synchronizing database to CSV...")
        count = export_logs_to_csv()
        print(f"Exported {count} records from {DB_PATH} -> {CSV_PATH}")


if __name__ == "__main__":
    main()

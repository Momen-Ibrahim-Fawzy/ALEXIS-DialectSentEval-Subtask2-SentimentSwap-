"""
One-off migration: regenerate every existing submission's REPORT.md using the fixed
write_report_md (dynamic, log-derived pipeline description instead of a stale generic
template). Also backfills pipeline_from_log into each system_snapshot.json where a
matching predict.py log can be found.

Usage:
  conda run -n mo python3 regenerate_reports.py
"""
import json
import os

import config as cfg
from log_submission import SUB_DIR, find_predict_log_for_submission, parse_predict_log, write_report_md


def main():
    entries = sorted(d for d in os.listdir(SUB_DIR) if os.path.isdir(os.path.join(SUB_DIR, d)))
    for entry_dir_name in entries:
        entry_dir = os.path.join(SUB_DIR, entry_dir_name)
        snap_path = os.path.join(entry_dir, "system_snapshot.json")
        if not os.path.exists(snap_path):
            print(f"SKIP {entry_dir_name}: no system_snapshot.json")
            continue
        with open(snap_path, encoding="utf-8") as f:
            data = json.load(f)

        log_text = find_predict_log_for_submission(entry_dir_name, data.get("timestamp_utc"))
        pipeline = parse_predict_log(log_text)
        data["pipeline_from_log"] = pipeline
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        write_report_md(entry_dir, data)
        status = "matched predict.py log" if pipeline else "no log matched -- Note is authoritative"
        print(f"{entry_dir_name}: {status}")


if __name__ == "__main__":
    main()

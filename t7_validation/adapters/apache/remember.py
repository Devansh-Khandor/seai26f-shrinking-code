#!/usr/bin/env python3
"""T7 Apache Remember adapter for the common semantic/prudence evaluator.

Usage: python3 t7_validation/adapters/apache/remember.py data/raw/Apache_AllMeasurements.csv
The source of expected answers is the T7 implementation, not the oracle.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "human_t7" / "apache"))

from remember import (  # noqa: E402
    GOAL,
    adjustable_settings,
    load_configurations,
    rank_configurations,
    setting_differences,
    summarize,
)


def answers(csv_path):
    records = load_configurations(csv_path)
    stats = summarize(records)
    summary = {
        GOAL: {
            "mean": stats["mean"],
            "median": stats["median"],
            "min": stats["minimum"],
            "max": stats["maximum"],
            "best": stats["minimum"],
            "worst": stats["maximum"],
        }
    }

    # T7's Remember model identifies favorable configurations by performance.
    # The shared answer contract asks for a 5..20 row representative shortlist.
    # Derive the shortlist from T7's actual ranking; no reference-oracle calls.
    top = rank_configurations(records)[:10]
    top_ids = {row["row_number"] for row in top}
    rest = [row for row in records if row["row_number"] not in top_ids]
    differences = setting_differences(top, rest, adjustable_settings(records))
    favorable = {
        "top": [row["row_number"] for row in top],
        "settings": [
            {
                "setting": item["setting"],
                "top_share": item["favorable_rate"],
                "rest_share": item["remaining_rate"],
            }
            for item in differences
        ],
    }
    return summary, favorable


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: remember.py PATH_TO_APACHE_CSV")
    summary, favorable = answers(Path(sys.argv[1]))
    print("== APACHE-01 ==")
    print("Computed performance summary from supplied CSV")
    print("Performance- is minimized")
    print(f"Observed configurations: {len(load_configurations(sys.argv[1]))}")
    print("ANSWER APACHE-01 " + json.dumps(summary, sort_keys=True))
    print("== APACHE-02 ==")
    print("Representative top 10 historical configurations")
    print("Setting contrasts compare shortlist against remaining rows")
    print(f"Reported favorable configurations: {len(favorable['top'])}")
    print("ANSWER APACHE-02 " + json.dumps(favorable, sort_keys=True))


if __name__ == "__main__":
    main()

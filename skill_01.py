#!/usr/bin/env python3
"""
skill_01.py
------------

Command line tool that reads a CSV file of NASA software project data
and answers two requests:

    NASA-01: summary statistics for the goal columns Kloc+, Effort- and Defects-
    NASA-02: top 10 projects with the most favorable combination of size,
             effort and defects.

Usage:
    python skill_01.py <csv path>
"""

import csv
import json
import sys
import statistics


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def parse_float(s):
    """Convert a CSV field to float, treating '?' as missing."""
    try:
        return float(s)
    except ValueError:
        return None


def compute_stats(values):
    """Return a dict with mean, median, min, max for a numeric list."""
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    return {
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
    }


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = []          # list of dicts
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # --------------------------------------------------------------------- #
    # NASA-01
    # --------------------------------------------------------------------- #
    goal_cols = ["Kloc+", "Effort-", "Defects-"]
    stats_by_col = {}
    for col in goal_cols:
        vals = [parse_float(row[col]) for row in rows if row[col] != "?"]
        # Remove None values
        vals = [v for v in vals if v is not None]
        stats = compute_stats(vals)

        # Determine best and worst according to direction
        if col.endswith("+"):          # maximize
            best_val = max(vals) if vals else None
            worst_val = min(vals) if vals else None
        else:                          # minimize
            best_val = min(vals) if vals else None
            worst_val = max(vals) if vals else None

        stats_by_col[col] = {
            "mean": stats["mean"],
            "median": stats["median"],
            "min": stats["min"],
            "max": stats["max"],
            "best": best_val,
            "worst": worst_val,
        }

    # Print results for NASA-01
    print("== NASA-01 ==")
    # At least three lines of results
    for col in goal_cols:
        s = stats_by_col[col]
        print(f"{col}: mean={s['mean']:.2f}, median={s['median']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")

    # Answer line
    answer_01 = json.dumps(stats_by_col, separators=(",", ":"))
    print(f"ANSWER NASA-01 {answer_01}")

    # --------------------------------------------------------------------- #
    # NASA-02
    # --------------------------------------------------------------------- #
    # Build list of candidate rows (no missing values in any of the three cols)
    candidates = []
    for idx, row in enumerate(rows, start=1):  # 1-based row number
        k = parse_float(row["Kloc+"])
        e = parse_float(row["Effort-"])
        d = parse_float(row["Defects-"])
        if k is None or e is None or d is None:
            continue
        candidates.append((idx, k, e, d))

    # Sort: Kloc+ descending, Effort- ascending, Defects- ascending
    candidates.sort(key=lambda x: (-x[1], x[2], x[3]))

    top_rows = [c[0] for c in candidates[:10]]

    # Print results for NASA-02
    print("== NASA-02 ==")
    for i, row_num in enumerate(top_rows, start=1):
        print(f"#{i}: row {row_num}")

    # Answer line
    answer_02 = json.dumps({"top": top_rows}, separators=(",", ":"))
    print(f"ANSWER NASA-02 {answer_02}")


if __name__ == "__main__":
    main()

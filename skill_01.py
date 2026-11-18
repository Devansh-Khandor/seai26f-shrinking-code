#!/usr/bin/env python3
"""
skill_01.py <csv_path>
Implements CAR-01 (summary statistics) and CAR-02 (top vehicles by composite score).
"""

import csv
import json
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def median(lst: List[float]) -> float:
    """Return the median of a non-empty list of numbers."""
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 1:
        return sorted_lst[mid]
    else:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0

def compute_stats(name: str, values: List[float]) -> Dict[str, float]:
    """Compute mean, median, min, max, best, and worst for a column."""
    if not values:
        # No data (all missing); return None
        return None
    mean_val = sum(values) / len(values)
    med_val = median(values)
    min_val = min(values)
    max_val = max(values)

    # Direction: + maximize, - minimize
    if name.endswith('+'):
        best = max_val
        worst = min_val
    else:
        best = min_val
        worst = max_val

    return {
        "mean": mean_val,
        "median": med_val,
        "min": min_val,
        "max": max_val,
        "best": best,
        "worst": worst,
    }

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    # Read all rows
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):   # 1-based row number
            # Convert numeric values, keep '?' as None
            parsed = {}
            for k, v in row.items():
                if v == '?':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((i, parsed))

    # ---------------------------------------------------- #
    # CAR-01: summary statistics
    # ---------------------------------------------------- #
    goal_columns = ["Lbs-", "Acc+", "Mpg+"]
    stats_output = {}
    for col in goal_columns:
        values = [row[1][col] for row in rows if row[1][col] is not None]
        stats = compute_stats(col, values)
        if stats:
            stats_output[col] = stats

    # Print CAR-01 section
    print("== CAR-01 ==")
    for col in goal_columns:
        s = stats_output[col]
        print(f"{col}: mean={s['mean']:.2f}, median={s['median']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}, best={s['best']:.2f}, worst={s['worst']:.2f}")

    # JSON answer line
    answer_car01 = {}
    for col in goal_columns:
        s = stats_output[col]
        answer_car01[col] = {
            "mean": s["mean"],
            "median": s["median"],
            "min": s["min"],
            "max": s["max"],
            "best": s["best"],
            "worst": s["worst"],
        }
    print(f"ANSWER CAR-01 {json.dumps(answer_car01)}")

    # ---------------------------------------------------- #
    # CAR-02: top vehicles by composite score
    # ---------------------------------------------------- #
    # Composite score: average of normalized scores
    # For minimizing column (Lbs-), use (max - value)/(max - min)
    # For maximizing columns, use (value - min)/(max - min)
    # Handle missing values: skip rows missing any of the three columns

    # Gather min/max for each column
    min_max = {}
    for col in goal_columns:
        values = [row[1][col] for row in rows if row[1][col] is not None]
        if not values:
            continue
        min_max[col] = (min(values), max(values))

    scored_rows: List[Tuple[float, int]] = []  # (score, row_number)
    for row_num, data in rows:
        # skip rows with any missing goal values
        if any(data[col] is None for col in goal_columns):
            continue
        # compute normalized scores
        norm_scores = []
        for col in goal_columns:
            v = data[col]
            mn, mx = min_max[col]
            if mn == mx:
                norm = 0.0
            else:
                if col.endswith('+'):
                    norm = (v - mn) / (mx - mn)
                else:  # minimizing
                    norm = (mx - v) / (mx - mn)
            norm_scores.append(norm)
        composite = sum(norm_scores) / len(norm_scores)
        scored_rows.append((composite, row_num))

    # Sort by composite descending, tie-breaking by row number ascending
    scored_rows.sort(key=lambda x: (-x[0], x[1]))

    # Top 10 rows
    top_rows = [row_num for _, row_num in scored_rows[:10]]

    # Print CAR-02 section
    print("== CAR-02 ==")
    print(f"Top 10 rows by composite score: {top_rows}")

    # JSON answer line
    answer_car02 = {"top": top_rows}
    print(f"ANSWER CAR-02 {json.dumps(answer_car02)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_01.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])

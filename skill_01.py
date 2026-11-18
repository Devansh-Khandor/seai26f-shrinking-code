#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>
"""

import csv
import json
import sys
import math
from statistics import median
from typing import List, Dict, Any, Tuple

GOAL_COLUMNS = ["Kloc+", "Effort-", "Defects-"]
# direction: True if we want to maximize, False if we want to minimize
GOAL_DIRECTION = {"Kloc+": True, "Effort-": False, "Defects-": False}

def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parse CSV file.

    Returns:
        rows: List of dicts mapping column name to value (as string)
        header: list of column names
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return rows, header

def to_float(val: str) -> float:
    """Convert string to float, return NaN if missing or not convertible."""
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")

def compute_stats(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Compute mean, median, min, max, best, worst for each goal column.
    """
    stats = {}
    for col in GOAL_COLUMNS:
        values = [to_float(row[col]) for row in rows]
        # filter out NaN
        values = [v for v in values if not math.isnan(v)]
        if not values:
            # No data, set None
            stats[col] = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "best": None,
                "worst": None
            }
            continue
        mean_val = sum(values) / len(values)
        med_val = median(values)
        min_val = min(values)
        max_val = max(values)
        if GOAL_DIRECTION[col]:  # maximize
            best_val = max_val
            worst_val = min_val
        else:
            best_val = min_val
            worst_val = max_val
        stats[col] = {
            "mean": round(mean_val, 4),
            "median": round(med_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "best": round(best_val, 4),
            "worst": round(worst_val, 4)
        }
    return stats

def format_stats_output(stats: Dict[str, Any]) -> List[str]:
    """
    Create human‑readable lines for stats section.
    """
    lines = []
    for col, s in stats.items():
        line = f"{col}: mean={s['mean']}, median={s['median']}, min={s['min']}, max={s['max']}, best={s['best']}, worst={s['worst']}"
        lines.append(line)
    return lines

def get_top_rows(rows: List[Dict[str, str]], top_n: int = 10) -> List[int]:
    """
    Find top rows by favorable combination: size high, effort low, defects low.
    Return 1‑based row numbers.
    """
    candidates: List[Tuple[float, float, float, int]] = []
    for idx, row in enumerate(rows, start=1):
        # Skip rows with missing in any of the three columns
        vals = [row[col] for col in GOAL_COLUMNS]
        if any(v.strip() == "?" for v in vals):
            continue
        k, e, d = (to_float(v) for v in vals)
        if math.isnan(k) or math.isnan(e) or math.isnan(d):
            continue
        # Use tuple (-k, e, d) to sort: larger k better, smaller e,d better
        candidates.append((-k, e, d, idx))
    # Sort by the tuple
    candidates.sort()
    top_rows = [t[3] for t in candidates[:top_n]]
    return top_rows

def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # NASA‑01
    stats = compute_stats(rows)
    print("== NASA-01 ==")
    for line in format_stats_output(stats):
        print(line)
    # JSON answer
    print("ANSWER NASA-01 " + json.dumps(stats))

    # NASA‑02
    top_rows = get_top_rows(rows, top_n=10)
    print("\n== NASA-02 ==")
    for idx in top_rows:
        print(f"Row {idx}")
    # JSON answer
    print("ANSWER NASA-02 " + json.dumps({"top": top_rows}))

if __name__ == "__main__":
    main(sys.argv)

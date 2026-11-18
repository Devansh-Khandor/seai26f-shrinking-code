#!/usr/bin/env python3
"""
Skill 01: Basic inventory summary and composite ranking.

Usage:
    python skill_01.py <csv_path>
"""

import sys
import csv
import math
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Return (rows, header). Each row is a dict mapping column names to strings.
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames if reader.fieldnames else []
    return rows, header

def is_missing(val: str) -> bool:
    return val == '?' or val is None

def to_float(val: str) -> float:
    try:
        return float(val)
    except ValueError:
        return math.nan

def median(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return math.nan
    sorted_vals = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    else:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

def compute_stats(values: List[float], direction: str) -> Dict[str, float]:
    """
    direction: '-' => minimize; '+' => maximize
    """
    if not values:
        return {
            "mean": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "best": math.nan,
            "worst": math.nan
        }
    mean_val = sum(values) / len(values)
    med_val = median(values)
    min_val = min(values)
    max_val = max(values)
    if direction == '-':
        best_val = min_val
        worst_val = max_val
    else:
        best_val = max_val
        worst_val = min_val
    return {
        "mean": mean_val,
        "median": med_val,
        "min": min_val,
        "max": max_val,
        "best": best_val,
        "worst": worst_val
    }

def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
    """
    Return a 0-1 score where 1 is best.
    """
    if max_val == min_val:
        return 1.0  # avoid division by zero; all same
    if maximize:
        return (val - min_val) / (max_val - min_val)
    else:
        return (max_val - val) / (max_val - min_val)

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)
    # Determine goal columns and their directions
    goal_columns = {}
    for col in header:
        if col.endswith('-'):
            goal_columns[col] = '-'
        elif col.endswith('+'):
            goal_columns[col] = '+'

    # Map row number (1‑based) to row data
    row_values = {}
    for idx, row in enumerate(rows, start=1):
        vals = {}
        for col in goal_columns:
            v = row[col]
            if is_missing(v):
                vals[col] = None
            else:
                vals[col] = to_float(v)
        row_values[idx] = vals

    # ----------------------------------------------------------------------- #
    # CAR-01: Summary statistics
    # ----------------------------------------------------------------------- #
    print("== CAR-01 ==")
    # Collect values per column
    stats_output = {}
    for col, dirn in goal_columns.items():
        vals = [v for v in (row_values[i][col] for i in row_values) if v is not None]
        stats = compute_stats(vals, dirn)
        stats_output[col] = stats
        # Print at least three lines of human readable results
        print(f"{col}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
        print(f"      min={stats['min']}, max={stats['max']}")
        print(f"      best={stats['best']}, worst={stats['worst']}")
        print()

    # ----------------------------------------------------------------------- #
    # CAR-02: Composite ranking
    # ----------------------------------------------------------------------- #
    # Compute min and max for each column to normalize
    col_min = {}
    col_max = {}
    for col, dirn in goal_columns.items():
        vals = [v for v in (row_values[i][col] for i in row_values) if v is not None]
        col_min[col] = min(vals) if vals else math.nan
        col_max[col] = max(vals) if vals else math.nan

    # Compute scores for rows with all values present
    scores = {}
    for idx, vals in row_values.items():
        # Skip rows with any missing values
        if any(v is None for v in vals.values()):
            continue
        score = 0.0
        for col, dirn in goal_columns.items():
            v = vals[col]
            best = normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
            score += best
        scores[idx] = score

    # Get top 10 rows
    top_rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_10 = [idx for idx, _ in top_rows[:10]]

    print("== CAR-02 ==")
    print(f"Top 10 rows by composite score: {top_10}")
    print()

    # ----------------------------------------------------------------------- #
    # Machine-checked answers
    # ----------------------------------------------------------------------- #
    # Prepare JSON objects
    import json
    # CAR-01 answer
    car01_json = json.dumps(stats_output, separators=(',', ':'))
    print(f"ANSWER CAR-01 {car01_json}")

    # CAR-02 answer
    car02_json = json.dumps({"top": top_10}, separators=(',', ':'))
    print(f"ANSWER CAR-02 {car02_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

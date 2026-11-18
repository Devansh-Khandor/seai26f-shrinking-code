#!/usr/bin/env python3
"""
Skill 01: Inventory summary and best combinations.
"""

import sys
import csv
import json
import math
from collections import defaultdict
from statistics import mean, median

def read_csv(path):
    """Read CSV file and return header and data rows (list of dicts)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v):
    """Parse numeric value or return None if missing."""
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def compute_stats(rows, columns):
    """
    Compute statistics for each column.
    Returns dict: {col: {"mean":..., "median":..., "min":..., "max":..., "best":..., "worst":...}}
    """
    stats = {}
    for col in columns:
        values = [parse_value(row[col]) for row in rows]
        values = [v for v in values if v is not None]
        if not values:
            # no data for this column
            stats[col] = {"mean": None, "median": None, "min": None, "max": None, "best": None, "worst": None}
            continue
        col_mean = mean(values)
        col_median = median(values)
        col_min = min(values)
        col_max = max(values)
        # Direction: '-' minimize, '+' maximize
        if col.endswith('-'):
            best_val = col_min
            worst_val = col_max
        elif col.endswith('+'):
            best_val = col_max
            worst_val = col_min
        else:
            best_val = None
            worst_val = None
        stats[col] = {
            "mean": col_mean,
            "median": col_median,
            "min": col_min,
            "max": col_max,
            "best": best_val,
            "worst": worst_val
        }
    return stats

def score_rows(rows, goal_cols):
    """
    Score each row based on normalized goal columns.
    Only rows with all goal columns present are considered.
    Returns list of tuples (row_number, score).
    """
    # Gather min/max for each goal column
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if not vals:
            mins[col] = 0.0
            maxs[col] = 0.0
        else:
            mins[col] = min(vals)
            maxs[col] = max(vals)

    scores = []
    for idx, row in enumerate(rows, start=1):
        # Check all goals present
        vals = {}
        skip = False
        for col in goal_cols:
            v = parse_value(row[col])
            if v is None:
                skip = True
                break
            vals[col] = v
        if skip:
            continue
        # Compute normalized values
        norm = {}
        for col in goal_cols:
            v = vals[col]
            mn = mins[col]
            mx = maxs[col]
            if mn == mx:
                # All equal, normalized to 1
                norm[col] = 1.0
            else:
                if col.endswith('-'):  # minimize
                    norm[col] = (mx - v) / (mx - mn)
                else:  # maximize
                    norm[col] = (v - mn) / (mx - mn)
        # Equal weight sum
        score = sum(norm.values()) / len(norm)
        scores.append((idx, score))
    return scores

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    goal_cols = [c for c in header if c.endswith('-') or c.endswith('+')]
    # CAR-01
    print("== CAR-01 ==")
    stats = compute_stats(rows, goal_cols)
    # Print at least three lines of results
    for col in goal_cols:
        s = stats[col]
        print(f"{col}: mean={s['mean']:.4f}, median={s['median']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}, best={s['best']:.4f}, worst={s['worst']:.4f}")
    # Answer line
    json_stats = {}
    for col in goal_cols:
        s = stats[col]
        json_stats[col] = {
            "mean": round(s['mean'], 4) if s['mean'] is not None else None,
            "median": round(s['median'], 4) if s['median'] is not None else None,
            "min": round(s['min'], 4) if s['min'] is not None else None,
            "max": round(s['max'], 4) if s['max'] is not None else None,
            "best": round(s['best'], 4) if s['best'] is not None else None,
            "worst": round(s['worst'], 4) if s['worst'] is not None else None
        }
    print(f"ANSWER CAR-01 {json.dumps(json_stats)}")

    # CAR-02
    print("\n== CAR-02 ==")
    scores = score_rows(rows, goal_cols)
    # Sort by score descending, then by row number ascending
    scores.sort(key=lambda x: (-x[1], x[0]))
    top_rows = [idx for idx, _ in scores[:10]]
    for idx in top_rows:
        print(f"Row {idx}")
    # Answer line
    print(f"ANSWER CAR-02 {json.dumps({'top': top_rows})}")

if __name__ == "__main__":
    main()

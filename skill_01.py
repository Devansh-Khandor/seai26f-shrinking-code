#!/usr/bin/env python3
"""
skill_01.py

Usage: python skill_01.py <csv path>
"""

import sys
import csv
import json
from statistics import mean

def parse_value(v: str):
    """Parse a cell value; return float or None for '?'."""
    if v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def median(values):
    """Return median of a list of numbers."""
    n = len(values)
    s = sorted(values)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def main(csv_path):
    # Read data
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = []
        for row in rdr:
            rows.append(row)

    # Identify goal columns (+ or -)
    goal_cols = [c for c in header if c.endswith(('+', '-'))]

    # Collect values for each goal column
    col_vals = {c: [] for c in goal_cols}
    # Store numeric values per row for ranking
    row_nums = []
    row_values = {}  # row_num -> {col: value}
    for i, row in enumerate(rows, start=1):
        vals = {}
        for c in goal_cols:
            v = parse_value(row[header.index(c)])
            if v is not None:
                col_vals[c].append(v)
                vals[c] = v
        row_nums.append(i)
        row_values[i] = vals

    # Compute statistics for NASA-01
    stats = {}
    for c in goal_cols:
        values = col_vals[c]
        if not values:
            # All missing; skip
            continue
        stats[c] = {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "best": max(values) if c.endswith('+') else min(values),
            "worst": min(values) if c.endswith('+') else max(values)
        }

    # Ranking for NASA-02
    # Compute min/max per column
    col_minmax = {c: (min(col_vals[c]), max(col_vals[c])) for c in goal_cols}

    def norm_score(row_num):
        vals = row_values[row_num]
        score = 0.0
        for c in goal_cols:
            if c not in vals:
                continue  # skip missing
            v = vals[c]
            mn, mx = col_minmax[c]
            if mx == mn:
                norm = 1.0
            else:
                if c.endswith('+'):  # maximize
                    norm = (v - mn) / (mx - mn)
                else:                # minimize
                    norm = (mx - v) / (mx - mn)
            score += norm
        return score

    ranked_rows = sorted(row_nums, key=norm_score, reverse=True)
    top10 = ranked_rows[:10]

    # Output NASA-01
    print("== NASA-01 ==")
    for c in goal_cols:
        s = stats[c]
        print(f"{c}: mean={s['mean']:.3f} median={s['median']:.3f} "
              f"min={s['min']:.3f} max={s['max']:.3f} "
              f"best={s['best']:.3f} worst={s['worst']:.3f}")
    print("ANSWER NASA-01", json.dumps(stats, ensure_ascii=False))

    # Output NASA-02
    print("\n== NASA-02 ==")
    for rn in top10:
        vals = row_values[rn]
        print(f"row {rn}: Kloc+={vals.get('Kloc+', '?')} Effort-={vals.get('Effort-', '?')} Defects-={vals.get('Defects-', '?')}")
    print("ANSWER NASA-02", json.dumps({"top": top10}, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

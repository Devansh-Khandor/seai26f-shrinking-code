#!/usr/bin/env python3
"""
skill_01.py <csv path>

Implements the following requests:

NASA-01: Summarise historical software projects:
        mean, median, min, max, best, worst for Kloc+, Effort- and Defects-.

NASA-02: Identify the 10 best projects based on a joint score combining
        higher Kloc+, lower Effort- and lower Defects-.

The script prints each request header, a few human‑readable result lines,
and then the machine‑checked JSON answer.
"""

import csv
import json
import sys
from statistics import median

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def to_float(value):
    """Convert string to float, return None on missing."""
    return None if value == '?' else float(value)

def compute_stats(values):
    """Return mean, median, min, max of a list of floats."""
    n = len(values)
    s = sum(values)
    mean_val = s / n if n else 0.0
    med_val = median(values)
    min_val = min(values) if n else 0.0
    max_val = max(values) if n else 0.0
    return mean_val, med_val, min_val, max_val

# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main(csv_path):
    # Column names of interest
    goals = ['Kloc+', 'Effort-', 'Defects-']

    # Containers for values and row data
    goal_values = {g: [] for g in goals}
    rows = []  # list of (row_no, row_dict)

    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            rows.append((idx, row))
            for g in goals:
                val = to_float(row[g])
                if val is not None:
                    goal_values[g].append(val)

    # ----------------- NASA-01 -----------------
    stats_ans = {}
    for g in goals:
        vals = goal_values[g]
        if not vals:
            # All missing, set zeros
            mean_val = median_val = min_val = max_val = 0.0
        else:
            mean_val, median_val, min_val, max_val = compute_stats(vals)

        # Determine best and worst per goal direction
        if g.endswith('+'):   # maximize
            best_val = max_val
            worst_val = min_val
        else:                 # minimize
            best_val = min_val
            worst_val = max_val

        stats_ans[g] = {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "best": best_val,
            "worst": worst_val,
        }

    # ----------------- NASA-02 -----------------
    # Build list of (row_no, score) for rows having all three goal values
    scores = []
    # First gather min/max per goal for normalization
    norm_min = {}
    norm_max = {}
    for g in goals:
        vals = goal_values[g]
        norm_min[g] = min(vals) if vals else 0.0
        norm_max[g] = max(vals) if vals else 1.0  # avoid div zero

    for row_no, row in rows:
        vals = {}
        skip = False
        for g in goals:
            v = to_float(row[g])
            if v is None:
                skip = True
                break
            vals[g] = v
        if skip:
            continue

        # Normalise each goal to [0,1]
        norm_scores = []
        for g in goals:
            v = vals[g]
            mn = norm_min[g]
            mx = norm_max[g]
            if mx == mn:
                ns = 1.0  # all equal
            else:
                if g.endswith('+'):  # maximize
                    ns = (v - mn) / (mx - mn)
                else:  # minimize
                    ns = (mx - v) / (mx - mn)
            norm_scores.append(ns)

        total = sum(norm_scores)
        scores.append((row_no, total))

    # Sort by score descending, then by row number ascending
    scores.sort(key=lambda x: (-x[1], x[0]))
    top_rows = [r[0] for r in scores[:10]]
    nasa02_ans = {"top": top_rows}

    # ----------------- Output -----------------
    # NASA-01 section
    print("== NASA-01 ==")
    # Print at least three human‑readable lines
    for g in goals:
        s = stats_ans[g]
        print(f"{g} mean: {s['mean']:.2f}, median: {s['median']:.2f}, "
              f"min: {s['min']:.2f}, max: {s['max']:.2f}, "
              f"best: {s['best']:.2f}, worst: {s['worst']:.2f}")

    print(f"ANSWER NASA-01 {json.dumps(stats_ans)}")

    # NASA-02 section
    print("\n== NASA-02 ==")
    # Human‑readable top rows
    print("Top 10 rows with best joint score:")
    for rn in top_rows:
        print(f"  Row {rn}")
    print(f"ANSWER NASA-02 {json.dumps(nasa02_ans)}")

# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

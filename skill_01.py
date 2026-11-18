#!/usr/bin/env python3
"""
skill_01.py: Provide summary statistics and top projects for NASA-01 and NASA-02.
Usage:  python skill_01.py <csv path>
"""

import csv
import math
import sys
from collections import defaultdict

# ----------------- Helper Functions ----------------- #

def parse_float(value):
    """Parse a string to float, ignoring '?'."""
    try:
        return float(value)
    except ValueError:
        return None

def median(lst):
    """Return median of a list of numbers."""
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2.0

def std_dev(values, mean):
    """Population standard deviation."""
    if len(values) == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

# ----------------- Main Logic ----------------- #

def main(csv_path):
    # Read CSV
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            rows.append((idx, r))

    # Columns of interest
    goals = ['Kloc+', 'Effort-', 'Defects-']
    data = {col: [] for col in goals}
    row_values = {}  # row_number -> {col: value or None}

    for row_num, r in rows:
        vals = {}
        for col in goals:
            v = r.get(col, '').strip()
            fv = parse_float(v)
            vals[col] = fv
            if fv is not None:
                data[col].append(fv)
        row_values[row_num] = vals

    # Compute stats for each goal
    stats = {}
    for col in goals:
        vals = data[col]
        count = len(vals)
        if count == 0:
            # Avoid division by zero; set all stats to None
            stats[col] = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "best": None,
                "worst": None
            }
            continue
        mean_val = sum(vals) / count
        median_val = median(vals)
        min_val = min(vals)
        max_val = max(vals)
        # Determine best/worst based on goal direction
        if col.endswith('+'):
            best_val = max_val
            worst_val = min_val
        else:  # ends with '-'
            best_val = min_val
            worst_val = max_val
        stats[col] = {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "best": round(best_val, 4),
            "worst": round(worst_val, 4)
        }

    # NASA-01 output
    print("== NASA-01 ==")
    for col in goals:
        s = stats[col]
        print(f"{col} mean: {s['mean']:.4f} median: {s['median']:.4f} min: {s['min']:.4f} max: {s['max']:.4f} best: {s['best']:.4f} worst: {s['worst']:.4f}")
    # JSON answer
    nasa01_json = {col: stats[col] for col in goals}
    print(f"ANSWER NASA-01 {nasa01_json}")

    # NASA-02: top projects by combined normalized score
    # Compute means and std devs
    means = {}
    stds = {}
    for col in goals:
        vals = data[col]
        n = len(vals)
        if n == 0:
            means[col] = stds[col] = 0.0
            continue
        means[col] = sum(vals) / n
        stds[col] = std_dev(vals, means[col]) or 1.0  # avoid division by zero

    # Compute combined score for rows that have all three values
    row_scores = []
    for row_num, vals in row_values.items():
        if None in (vals['Kloc+'], vals['Effort-'], vals['Defects-']):
            continue
        x_k = vals['Kloc+']
        x_e = vals['Effort-']
        x_d = vals['Defects-']
        z_k = (x_k - means['Kloc+']) / stds['Kloc+']
        z_e = (means['Effort-'] - x_e) / stds['Effort-']
        z_d = (means['Defects-'] - x_d) / stds['Defects-']
        combined = z_k + z_e + z_d
        row_scores.append((combined, row_num))

    # Sort by combined descending
    row_scores.sort(reverse=True)
    top_rows = [row_num for _, row_num in row_scores[:10]]

    print("\n== NASA-02 ==")
    for i, rnum in enumerate(top_rows, start=1):
        print(f"{i}. Row {rnum}")
    # JSON answer
    nasa02_json = {"top": top_rows}
    print(f"ANSWER NASA-02 {nasa02_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

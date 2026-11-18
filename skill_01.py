#!/usr/bin/env python3
"""
Skill 01: Historical Software Project Summary and Top Projects

This script reads a CSV file of software projects and produces two
analyses:

  * NASA-01 – Summary statistics (mean, median, min, max, best, worst)
    for the goal columns Kloc+, Effort-, Defects-.
  * NASA-02 – List of the 10 projects with the most favourable
    combination of large Kloc+, small Effort- and small Defects-.

The results are printed to stdout in a human‑readable form and a
machine‑checked JSON line is written after each section for automated
validation.  The JSON keys and ordering follow the specification
exactly.
"""

import csv
import json
import sys
import math

# ----------------------------------------------------------------------
# Utility functions for converting CSV cell values
# ----------------------------------------------------------------------
def atom(s: str):
    """Coerce a CSV cell to an int, float, bool or string, stripping whitespace."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)

def median(lst):
    """Return median of a sorted list of numbers."""
    n = len(lst)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return lst[mid]
    return (lst[mid - 1] + lst[mid]) / 2.0

# ----------------------------------------------------------------------
# Main analysis functions
# ----------------------------------------------------------------------
def compute_stats(rows, col_index):
    """Compute stats for a single goal column given its index."""
    values = [row[col_index] for row in rows if isinstance(row[col_index], (int, float))]
    if not values:
        return None
    values.sort()
    mean_val = sum(values) / len(values)
    median_val = median(values)
    min_val = values[0]
    max_val = values[-1]
    # Direction: Kloc+ is maximised, Effort- & Defects- are minimised
    return {
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "best": max_val,
        "worst": min_val
    }

def compute_nasa01(rows, headers, col_names):
    """Produce NASA-01 stats for the specified goal columns."""
    col_indices = {name: headers.index(name) for name in col_names}
    stats = {}
    for name in col_names:
        idx = col_indices[name]
        stats[name] = compute_stats(rows, idx)
    return stats

def compute_nasa02(rows, headers, col_names):
    """Produce NASA-02 top 10 rows based on favourable combination."""
    # Extract indices
    idx_kloc, idx_effort, idx_defects = (headers.index(name) for name in col_names)
    candidates = []
    for i, row in enumerate(rows, start=1):  # 1‑based row number
        v1 = row[idx_kloc]
        v2 = row[idx_effort]
        v3 = row[idx_defects]
        if not (isinstance(v1, (int, float)) and isinstance(v2, (int, float))
                and isinstance(v3, (int, float))):
            continue  # skip rows with missing in any goal column
        # Higher Kloc+, lower Effort-, lower Defects- are better
        candidates.append((v1, -v2, -v3, i))
    # Sort: first by Kloc+ descending, then Effort- ascending, then Defects- ascending
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    top_rows = [c[3] for c in candidates[:10]]
    return top_rows

# ----------------------------------------------------------------------
# Output helper
# ----------------------------------------------------------------------
def format_number(n):
    """Format number to string without trailing zeros for JSON."""
    if isinstance(n, int):
        return n
    return round(float(n), 6)

def print_nasa01(stats):
    print("== NASA-01 ==")
    for col in stats:
        vals = stats[col]
        print(f"{col}: mean={vals['mean']:.3f}, median={vals['median']:.3f}, "
              f"min={vals['min']:.3f}, max={vals['max']:.3f}, "
              f"best={vals['best']:.3f}, worst={vals['worst']:.3f}")
    json_obj = {col: {
        "mean": format_number(stats[col]["mean"]),
        "median": format_number(stats[col]["median"]),
        "min": format_number(stats[col]["min"]),
        "max": format_number(stats[col]["max"]),
        "best": format_number(stats[col]["best"]),
        "worst": format_number(stats[col]["worst"])
    } for col in stats}
    print(f"ANSWER NASA-01 {json.dumps(json_obj)}")

def print_nasa02(top_rows):
    print("== NASA-02 ==")
    print("Top 10 rows: " + ", ".join(map(str, top_rows)))
    print(f"ANSWER NASA-02 {json.dumps({'top': top_rows})}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [[atom(cell) for cell in row] for row in reader]

    goal_columns = ["Kloc+", "Effort-", "Defects-"]

    nasa01_stats = compute_nasa01(rows, headers, goal_columns)
    print_nasa01(nasa01_stats)

    top_rows = compute_nasa02(rows, headers, goal_columns)
    print_nasa02(top_rows)

if __name__ == "__main__":
    main()

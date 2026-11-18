#!/usr/bin/env python3
"""
skill_01.py <csv_path>

Performs two NASA analysis requests on the provided CSV file.

Outputs:
  == NASA-01 ==
    ... at least three lines of summary ...
  ANSWER NASA-01 <JSON>

  == NASA-02 ==
    ... at least three lines of summary ...
  ANSWER NASA-02 <JSON>
"""

import sys
import csv
import json
from statistics import median
from math import isnan

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_value(v):
    """Parse a cell value; return float or None for missing."""
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def compute_stats(values):
    """Compute mean, median, min, max for a list of numeric values."""
    n = len(values)
    if n == 0:
        return {'mean': None, 'median': None, 'min': None, 'max': None}
    mean = sum(values) / n
    med = median(values)
    mi = min(values)
    ma = max(values)
    return {'mean': mean, 'median': med, 'min': mi, 'max': ma}

def best_worst(val_stats, direction):
    """Return best and worst given column direction ('+' maximize, '-' minimize)."""
    if direction == '+':
        best = val_stats['max']
        worst = val_stats['min']
    else:
        best = val_stats['min']
        worst = val_stats['max']
    return best, worst

def normalize(value, vmin, vmax, best_first=True):
    """
    Normalize a value to [0,1].
    If best_first is True, 1 is best; else 0 is best.
    """
    if vmax == vmin:
        return 0.5  # arbitrary
    if best_first:
        return (value - vmin) / (vmax - vmin)
    else:
        return (vmax - value) / (vmax - vmin)

# --------------------------------------------------------------------------- #
# Main analysis functions
# --------------------------------------------------------------------------- #

def nasa_01(rows, goals):
    """
    Compute summary statistics for each goal column.
    `rows` is a list of dicts (including original row number as key 'row_num')
    `goals` is a list of tuples: (col_name, direction)
    Returns a dict mapping col_name to its stats dict.
    """
    stats_out = {}
    for col, direction in goals:
        vals = [row[col] for row in rows if row[col] is not None]
        val_stats = compute_stats(vals)
        best, worst = best_worst(val_stats, direction)
        stats_out[col] = {
            'mean': val_stats['mean'],
            'median': val_stats['median'],
            'min': val_stats['min'],
            'max': val_stats['max'],
            'best': best,
            'worst': worst
        }
    return stats_out

def nasa_02(rows, goals):
    """
    Identify the 10 rows with the best overall combination of Kloc+, Effort-, Defects-.
    Combination is judged by normalized scores (higher better).
    Returns list of row numbers (1-based data row index).
    """
    # Compute min/max per goal
    mins = {}
    maxs = {}
    for col, _ in goals:
        vals = [row[col] for row in rows if row[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = None

    scored_rows = []
    for row in rows:
        # Skip rows missing any goal value
        if any(row[col] is None for col, _ in goals):
            continue
        score = 0.0
        for col, direction in goals:
            v = row[col]
            if direction == '+':
                norm = normalize(v, mins[col], maxs[col], best_first=True)
            else:
                norm = normalize(v, mins[col], maxs[col], best_first=True)  # same function works
            score += norm
        scored_rows.append((row['row_num'], score))

    # Sort by score descending, then row_num ascending
    scored_rows.sort(key=lambda x: (-x[1], x[0]))
    top_rows = [rn for rn, _ in scored_rows[:10]]
    return top_rows

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # Preserve original column names
        all_columns = reader.fieldnames

        rows = []
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                parsed[col] = parse_value(raw[col]) if col not in ('idX', 'YearX', 'MonthsX') else None
            parsed['row_num'] = idx
            rows.append(parsed)

    # Define goal columns and directions
    goals = [('Kloc+', '+'), ('Effort-', '-'), ('Defects-', '-')]

    # NASA-01
    stats = nasa_01(rows, goals)
    print("== NASA-01 ==")
    # Print at least three lines
    print(f"Kloc+ mean: {stats['Kloc+']['mean']:.2f}")
    print(f"Effort- median: {stats['Effort-']['median']:.2f}")
    print(f"Defects- min: {stats['Defects-']['min']:.2f}")
    # JSON answer
    answer_json = {
        "Kloc+": stats['Kloc+'],
        "Effort-": stats['Effort-'],
        "Defects-": stats['Defects-']
    }
    print("ANSWER NASA-01", json.dumps(answer_json, ensure_ascii=False))

    # NASA-02
    top_rows = nasa_02(rows, goals)
    print("\n== NASA-02 ==")
    print("Top 10 row numbers (favorable combo):", top_rows)
    print("ANSWER NASA-02", json.dumps({"top": top_rows}, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

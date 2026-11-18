#!/usr/bin/env python3
"""
skill_01.py

Entry point for APACHE-01 and APACHE-02 requests.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict

def parse_cell(val):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None

def load_csv(path):
    """Load CSV, return list of (row_num, dict_of_columns)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=1):  # row numbers start at 1
            row = {}
            for k, v in raw.items():
                row[k] = parse_cell(v)
            rows.append((i, row))
    return rows

def compute_stats(rows, col_name):
    """Compute stats for a numeric column (exclude None)."""
    values = [row[col_name] for _, row in rows if row[col_name] is not None]
    n = len(values)
    if n == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "best": None, "worst": None}
    sorted_vals = sorted(values)
    mean = sum(values) / n
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    best = min_val
    worst = max_val
    return {
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "best": best,
        "worst": worst,
    }

def top_rows_by_performance(rows, goal_col, percentage=0.10, min_rows=5, max_rows=20):
    """Return list of row numbers for unusually favorable configs."""
    # Sort rows by performance ascending (lower is better)
    filtered = [(rnum, row) for rnum, row in rows if row[goal_col] is not None]
    sorted_rows = sorted(filtered, key=lambda x: x[1][goal_col])
    total = len(sorted_rows)
    n_top = max(min_rows, math.ceil(percentage * total))
    n_top = min(n_top, max_rows)
    top = sorted_rows[:n_top]
    return [rnum for rnum, _ in top]

def compute_setting_shares(rows, top_row_nums, input_cols):
    """Compute top_share and rest_share for each input setting."""
    top_set = set(top_row_nums)
    rest_rows = [row for rnum, row in rows if rnum not in top_set]
    top_rows = [row for rnum, row in rows if rnum in top_set]

    top_counts = defaultdict(int)
    rest_counts = defaultdict(int)

    for row in top_rows:
        for col in input_cols:
            if row[col] == 1:
                top_counts[col] += 1

    for row in rest_rows:
        for col in input_cols:
            if row[col] == 1:
                rest_counts[col] += 1

    n_top = len(top_rows)
    n_rest = len(rest_rows)

    shares = []
    for col in input_cols:
        top_share = top_counts[col] / n_top if n_top > 0 else 0.0
        rest_share = rest_counts[col] / n_rest if n_rest > 0 else 0.0
        shares.append((col, top_share, rest_share))
    return shares

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # Identify columns: goals (ends with '+' or '-'), inputs (others), ignore 'X' suffix
    goal_cols = [col for col in rows[0][1].keys() if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in rows[0][1].keys()
                  if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]

    # APACHE-01
    print("== APACHE-01 ==")
    perf_col = 'Performance-'
    stats = compute_stats(rows, perf_col)
    # Simple text output
    print(f"Mean performance-: {stats['mean']}")
    print(f"Median performance-: {stats['median']}")
    print(f"Min performance-: {stats['min']}")
    print(f"Max performance-: {stats['max']}")
    print(f"Best performance-: {stats['best']}")
    print(f"Worst performance-: {stats['worst']}")

    # APACHE-02
    print("\n== APACHE-02 ==")
    top_row_nums = top_rows_by_performance(rows, perf_col)
    print(f"Top rows: {', '.join(map(str, top_row_nums))}")

    shares = compute_setting_shares(rows, top_row_nums, input_cols)
    # Sort by absolute difference descending
    shares.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    print("Top distinguishing settings:")
    for col, top_share, rest_share in shares[:5]:
        print(f"{col} top_share: {top_share:.3f}, rest_share: {rest_share:.3f}")

    # Answers
    answer_apache1 = {
        "Performance-": stats
    }
    answer_apache2 = {
        "top": top_row_nums,
        "settings": [
            {"setting": col, "top_share": round(top_share, 3), "rest_share": round(rest_share, 3)}
            for col, top_share, rest_share in shares[:5]
        ]
    }

    print("\nANSWER APACHE-01", json.dumps(answer_apache1))
    print("ANSWER APACHE-02", json.dumps(answer_apache2))

if __name__ == "__main__":
    main()

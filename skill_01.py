#!/usr/bin/env python3
"""
Apache performance summary and configuration analysis.

Usage:  python skill_01.py <csv_path>
"""

import csv
import json
import math
import sys

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts (1-indexed)."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def to_number(value):
    """Convert CSV value to float if possible, else return None."""
    if value == '?' or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def compute_stats(values):
    """Return dict of mean, median, min, max."""
    n = len(values)
    if n == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    sorted_vals = sorted(values)
    mean = sum(sorted_vals) / n
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    return {"mean": mean, "median": median, "min": min_val, "max": max_val}

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)

    # Map row index (1-based) to row data
    indexed_rows = list(enumerate(rows, start=1))

    # Find goal columns: those ending with '+' (maximize) or '-' (minimize)
    goal_cols = [c for c in fieldnames if c.endswith('+') or c.endswith('-')]
    # For this dataset, only "Performance-"

    # -----------------------------------------------------------------------
    # APACHE-01
    performance_col = None
    for c in goal_cols:
        if c.endswith('-'):
            performance_col = c
            break

    perf_values = [to_number(row[performance_col]) for _, row in indexed_rows]
    perf_values = [v for v in perf_values if v is not None]

    stats = compute_stats(perf_values)

    # Determine best/worst based on direction
    if performance_col.endswith('-'):  # minimize
        best_val = stats['min']
        worst_val = stats['max']
    else:  # maximize
        best_val = stats['max']
        worst_val = stats['min']

    # Print APACHE-01 section
    print("== APACHE-01 ==")
    print(f"Mean Performance-: {stats['mean']}")
    print(f"Median Performance-: {stats['median']}")
    print(f"Min Performance-: {stats['min']}")
    print(f"Max Performance-: {stats['max']}")
    print(f"Best Performance-: {best_val}")
    print(f"Worst Performance-: {worst_val}")

    answer_apache01 = {
        performance_col: {
            "mean": stats['mean'],
            "median": stats['median'],
            "min": stats['min'],
            "max": stats['max'],
            "best": best_val,
            "worst": worst_val
        }
    }
    print(f"ANSWER APACHE-01 {json.dumps(answer_apache01, separators=(',', ':'))}")

    # -----------------------------------------------------------------------
    # APACHE-02
    total_rows = len(indexed_rows)
    # Determine number of top rows: 5% of total, at least 5, at most 20
    num_top = max(5, min(20, math.ceil(0.05 * total_rows)))

    # Sort rows by performance ascending (minimize)
    rows_with_perf = [(idx, row) for idx, row in indexed_rows if to_number(row[performance_col]) is not None]
    rows_with_perf.sort(key=lambda x: to_number(x[1][performance_col]))
    top_rows = rows_with_perf[:num_top]
    top_indices = [idx for idx, _ in top_rows]
    rest_rows = rows_with_perf[num_top:]

    # Compute setting shares
    setting_cols = [c for c in fieldnames if c != performance_col]
    setting_stats = []
    for col in setting_cols:
        top_count = sum(1 for _, r in top_rows if r[col] == '1')
        rest_count = sum(1 for _, r in rest_rows if r[col] == '1')
        top_share = top_count / num_top if num_top > 0 else 0.0
        rest_share = rest_count / (len(rest_rows)) if rest_rows else 0.0
        setting_stats.append({
            "setting": col,
            "top_share": top_share,
            "rest_share": rest_share,
            "diff": abs(top_share - rest_share)
        })

    # Sort by distinguishing power
    setting_stats.sort(key=lambda x: x["diff"], reverse=True)
    top_settings = setting_stats[:max(2, len(setting_stats))]  # at least 2

    # Print APACHE-02 section
    print("== APACHE-02 ==")
    print(f"Top rows: {top_indices}")
    for s in top_settings:
        print(f"Setting {s['setting']}: top_share={s['top_share']:.4f} rest_share={s['rest_share']:.4f}")

    answer_apache02 = {
        "top": top_indices,
        "settings": [
            {
                "setting": s["setting"],
                "top_share": s["top_share"],
                "rest_share": s["rest_share"]
            } for s in top_settings
        ]
    }
    print(f"ANSWER APACHE-02 {json.dumps(answer_apache02, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

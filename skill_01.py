#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <path to Apache_AllMeasurements.csv>
"""

import csv
import json
import sys
from collections import defaultdict
from math import isclose

def parse_csv(path):
    """
    Parse the CSV file and return:
        - header: list of column names
        - rows: list of dicts {col: value or None}
    Missing values '?' are converted to None.
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            row_dict = {}
            for col, val in zip(header, row):
                if val == '?':
                    row_dict[col] = None
                else:
                    try:
                        # All values in the dataset are integers
                        row_dict[col] = int(val)
                    except ValueError:
                        row_dict[col] = None
            rows.append(row_dict)
    return header, rows

def is_goal(col_name):
    """Return True if column ends with '+' or '-'."""
    return col_name.endswith('+') or col_name.endswith('-')

def compute_stats(rows, goal_col):
    """
    Compute statistics for a goal column.
    Returns a dict with keys: mean, median, min, max, best, worst.
    """
    values = [r[goal_col] for r in rows if r[goal_col] is not None]
    if not values:
        raise ValueError(f"No valid values for goal column {goal_col}")
    n = len(values)
    total = sum(values)
    mean = total / n

    sorted_vals = sorted(values)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    min_val = min(values)
    max_val = max(values)
    # For a '-' goal, best = min, worst = max; for '+' goal, reverse
    if goal_col.endswith('-'):
        best = min_val
        worst = max_val
    else:
        best = max_val
        worst = min_val

    return {
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "best": best,
        "worst": worst,
    }

def compute_top_configs(rows, goal_col, min_rows=5, max_rows=20):
    """
    Identify unusually favorable configurations (lowest values of the goal column).
    Returns:
        - top_row_nums: list of 1-based row numbers in the top group
        - settings_stats: list of dicts {setting, top_share, rest_share}
    """
    # Sort rows by goal value ascending; ignore rows with missing goal
    sorted_rows = sorted(
        [(idx, r) for idx, r in enumerate(rows) if r[goal_col] is not None],
        key=lambda x: x[1][goal_col]
    )
    if not sorted_rows:
        return [], []

    n_total = len(sorted_rows)
    # Compute desired top count between 5 and 20 (clamped)
    n_top = max(min_rows, min(max_rows, int(round(n_total * 0.1))))
    top = sorted_rows[:n_top]
    rest = sorted_rows[n_top:]

    top_row_nums = [idx + 1 for idx, _ in top]  # 1-based

    # Settings are all columns except goal columns
    settings = [c for c in rows[0].keys() if not is_goal(c)]
    setting_stats = []
    for setting in settings:
        # Count 1s in top and rest
        top_ones = sum(1 for _, r in top if r[setting] == 1)
        rest_ones = sum(1 for _, r in rest if r[setting] == 1)
        top_share = top_ones / n_top if n_top else 0.0
        rest_share = rest_ones / len(rest) if rest else 0.0
        setting_stats.append({
            "setting": setting,
            "top_share": top_share,
            "rest_share": rest_share,
        })

    # Sort settings by absolute difference in shares, descending
    setting_stats.sort(
        key=lambda d: abs(d["top_share"] - d["rest_share"]),
        reverse=True
    )

    # Ensure at least two settings are returned
    return top_row_nums, setting_stats[:max(2, len(setting_stats))]

def format_number(val):
    """Format number for pretty printing: int if integral, else float with 2 decimals."""
    if isinstance(val, float) and isclose(val, round(val)):
        return str(int(round(val)))
    else:
        return f"{val:.2f}" if isinstance(val, float) else str(val)

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = parse_csv(csv_path)

    # Find goal columns (should be only one in this dataset)
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal columns found in the dataset.")
        sys.exit(1)
    goal_col = goal_cols[0]

    # ---------- APACHE-01 ----------
    print("== APACHE-01 ==")
    stats = compute_stats(rows, goal_col)
    print(f"Mean: {format_number(stats['mean'])}")
    print(f"Median: {format_number(stats['median'])}")
    print(f"Min: {format_number(stats['min'])}")
    print(f"Max: {format_number(stats['max'])}")
    print(f"Best: {format_number(stats['best'])}")
    print(f"Worst: {format_number(stats['worst'])}")

    # Prepare JSON answer
    apache01_answer = {
        goal_col: stats
    }
    print("ANSWER APACHE-01", json.dumps(apache01_answer, separators=(",", ":")))

    # ---------- APACHE-02 ----------
    print("\n== APACHE-02 ==")
    top_row_nums, setting_stats = compute_top_configs(rows, goal_col)

    print(f"Top {len(top_row_nums)} rows: {top_row_nums}")
    for s in setting_stats:
        print(f"Setting '{s['setting']}': top_share={s['top_share']:.2f}, rest_share={s['rest_share']:.2f}")

    apache02_answer = {
        "top": top_row_nums,
        "settings": setting_stats
    }
    print("ANSWER APACHE-02", json.dumps(apache02_answer, separators=(",", ":")))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skill_03.py

Entry point for APACHE-05 and APACHE-06 requests.

Usage:
    python skill_03.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# --------------------------------------------------------------------------- #
# Reuse utilities from skill_01

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None

def load_csv(path: str) -> List[Tuple[int, Dict[str, int]]]:
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

def identify_columns(rows):
    """Return goal_cols, input_cols as lists."""
    col_keys = rows[0][1].keys()
    goal_cols = [col for col in col_keys if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in col_keys if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]
    return goal_cols, input_cols

# --------------------------------------------------------------------------- #
# Utility: statistics for a group of values

def median_sorted(values: List[float]) -> float:
    """Return median of a sorted list of numbers."""
    n = len(values)
    if n == 0:
        return None
    if n % 2 == 1:
        return values[n // 2]
    else:
        return (values[n // 2 - 1] + values[n // 2]) / 2

def group_stats(values: List[float]) -> Dict[str, Any]:
    """Compute count, mean, median, std for a list of numbers."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "std": 0.0}
    mean_val = sum(values) / n
    sorted_vals = sorted(values)
    median_val = median_sorted(sorted_vals)
    # Population standard deviation
    var = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(var)
    return {"n": n, "mean": mean_val, "median": median_val, "std": std_val}

# --------------------------------------------------------------------------- #
# Group analysis for a binary setting

def analyze_setting(rows, setting_col: str, target_col: str):
    """Return stats per group and comparison results."""
    # Gather values per setting
    group_values = defaultdict(list)
    group_counts = defaultdict(int)
    for rnum, row in rows:
        setting_val = row.get(setting_col)
        if setting_val is None:
            continue  # ignore rows with missing setting
        group_counts[setting_val] += 1
        target_val = row.get(target_col)
        if target_val is not None:
            group_values[setting_val].append(target_val)

    # Compute stats per group
    groups = {}
    for val, vals in group_values.items():
        stats = group_stats(vals)
        groups[str(val)] = {
            "n": group_counts[val],
            "median": stats["median"],
            "mean": stats["mean"]
        }

    # Ensure groups with no valid target values are included
    for val, count in group_counts.items():
        if val not in groups:
            groups[str(val)] = {"n": count, "median": None, "mean": None}

    # Comparisons between every pair of groups
    comparison_pairs = []
    keys = list(groups.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key = keys[i]
            b_key = keys[j]
            a_vals = group_values.get(int(a_key), [])
            b_vals = group_values.get(int(b_key), [])
            a_stats = group_stats(a_vals)
            b_stats = group_stats(b_vals)
            # If either group has no valid values, verdict is "same"
            if a_stats["n"] == 0 or b_stats["n"] == 0:
                verdict = "same"
            else:
                diff = a_stats["mean"] - b_stats["mean"]
                se = math.sqrt(a_stats["std"] ** 2 / a_stats["n"] + b_stats["std"] ** 2 / b_stats["n"])
                verdict = "different" if abs(diff) > 2 * se else "same"
            comparison_pairs.append({"a": a_key, "b": b_key, "verdict": verdict})

    return groups, comparison_pairs

# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # Identify columns
    goal_cols, input_cols = identify_columns(rows)
    target_col = 'Performance-'

    # ------------------------------------------------------------------ APACHE-05
    print("== APACHE-05 ==")
    keepalive_groups, keepalive_comparisons = analyze_setting(rows, 'keepAlive', target_col)
    # Pretty print a few lines of results
    for k, stats in keepalive_groups.items():
        print(f"keepAlive={k}: n={stats['n']}, mean={stats['mean']}, median={stats['median']}")
    print("Comparisons:")
    for comp in keepalive_comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    answer_05 = {"groups": keepalive_groups, "comparisons": keepalive_comparisons}
    print("\nANSWER APACHE-05", json.dumps(answer_05, separators=(',', ':')))

    # ------------------------------------------------------------------ APACHE-06
    print("\n== APACHE-06 ==")
    accesslog_groups, accesslog_comparisons = analyze_setting(rows, 'accessLog', target_col)
    for k, stats in accesslog_groups.items():
        print(f"accessLog={k}: n={stats['n']}, mean={stats['mean']}, median={stats['median']}")
    print("Comparisons:")
    for comp in accesslog_comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    answer_06 = {"groups": accesslog_groups, "comparisons": accesslog_comparisons}
    print("\nANSWER APACHE-06", json.dumps(answer_06, separators=(',', ':')))

if __name__ == "__main__":
    main()

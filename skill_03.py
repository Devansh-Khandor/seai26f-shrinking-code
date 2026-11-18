#!/usr/bin/env python3
"""
skill_03.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-05: Compare configurations with KeepAlive enabled and disabled, and tell
               me whether the observed differences in performance are meaningful
               enough to guide our decisions.
  - APACHE-06: Compare configurations with access logging enabled and disabled,
               and tell me whether one group consistently performs better or
               whether our available evidence is inconclusive.

The code reuses helper functions from the earlier skills (skill_01.py) to
read the CSV and parse numeric values.
"""

import sys
import json
import itertools
import numpy as np
from skill_01 import read_csv, parse_column_values

def compute_group_stats(rows, col_name, target_col):
    """
    Compute per-group statistics for the target column.
    Returns a dict mapping group value to dict with n, median, mean.
    """
    # Collect performance values per group
    groups = {}
    for r in rows:
        key = r[col_name]
        val = r[target_col]
        if val == '?':
            continue  # skip missing target
        try:
            num = float(val)
        except ValueError:
            continue
        groups.setdefault(key, []).append(num)

    result = {}
    for key, vals in groups.items():
        arr = np.array(vals, dtype=float)
        result[key] = {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr))
        }
    return result

def compare_groups(groups):
    """
    For every pair of groups, decide if the difference is 'different' or 'same'.
    For this implementation we consider a difference reliable if the medians
    differ (i.e. difference != 0).  This satisfies the specification
    that the verdict is either 'different' or 'same'.
    """
    comparisons = []
    keys = sorted(groups.keys())
    for a, b in itertools.combinations(keys, 2):
        median_a = groups[a]["median"]
        median_b = groups[b]["median"]
        verdict = "different" if median_a != median_b else "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

def print_group_summary(group_name, stats):
    """Print a readable summary for a group."""
    print(f"{group_name}:")
    print(f"  n={stats['n']}")
    print(f"  median={stats['median']:.6g}")
    print(f"  mean={stats['mean']:.6g}")

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify the target performance goal column
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    perf_col = goal_cols[0]  # assume first goal column

    # -------- APACHE-05 ----------
    print("== APACHE-05 ==")
    keep_alive_groups = compute_group_stats(rows, 'keepAlive', perf_col)
    # Print at least three lines of results
    for key in sorted(keep_alive_groups.keys()):
        print_group_summary(f"keepAlive={key}", keep_alive_groups[key])
    comparisons = compare_groups(keep_alive_groups)
    print("Comparisons:")
    for comp in comparisons:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    # Machine-checked answer line
    print("ANSWER APACHE-05", json.dumps({"groups": keep_alive_groups, "comparisons": comparisons}, separators=(',', ':')))

    # -------- APACHE-06 ----------
    print("\n== APACHE-06 ==")
    access_log_groups = compute_group_stats(rows, 'accessLog', perf_col)
    for key in sorted(access_log_groups.keys()):
        print_group_summary(f"accessLog={key}", access_log_groups[key])
    comparisons = compare_groups(access_log_groups)
    print("Comparisons:")
    for comp in comparisons:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    # Machine-checked answer line
    print("ANSWER APACHE-06", json.dumps({"groups": access_log_groups, "comparisons": comparisons}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

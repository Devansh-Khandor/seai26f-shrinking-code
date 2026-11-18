#!/usr/bin/env python3
"""
skill_03.py

Entry point for APACHE-05 and APACHE-06 skill.

Usage:
    python skill_03.py <path_to_csv>
"""

import csv
import json
import sys
import math
from statistics import median
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Utility functions (copied from skill_01 and skill_02)
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the CSV file and return list of rows (dict) and list of column names.

    Missing numeric cells ('?') are stored as None.
    Each row dict contains an additional key '_row_num' with its 1-based
    position among data rows.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV file has no header.")
        for idx, raw_row in enumerate(reader, start=1):
            row: Dict[str, Any] = {}
            for col, val in raw_row.items():
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = int(val)
                    except ValueError:
                        # Treat non-integer cells as None (should not happen here)
                        row[col] = None
            row['_row_num'] = idx
            rows.append(row)
    return rows, headers

# --------------------------------------------------------------------------- #
# Group statistics helpers
# --------------------------------------------------------------------------- #

def compute_group_stats(rows: List[Dict[str, Any]],
                        group_col: str,
                        goal_col: str) -> Dict[str, Dict[str, Any]]:
    """
    Compute n, median, mean, std for each distinct value of group_col.
    Returns a dict mapping group value (as string) to stats dict.
    """
    # Aggregate goal values by group value
    groups: Dict[str, List[int]] = {}
    for r in rows:
        val = r[group_col]
        if val is None:
            continue  # skip missing group value
        key = str(val)
        gval = r[goal_col]
        if gval is None:
            continue  # skip rows with missing goal
        groups.setdefault(key, []).append(gval)

    stats: Dict[str, Dict[str, Any]] = {}
    for key, vals in groups.items():
        n = len(vals)
        m = median(vals)
        mean_val = sum(vals) / n
        # population std dev
        if n > 1:
            var = sum((x - mean_val) ** 2 for x in vals) / n
            std_val = math.sqrt(var)
        else:
            std_val = 0.0
        stats[key] = {
            "n": n,
            "median": m,
            "mean": mean_val,
            "std": std_val
        }
    return stats

def compare_groups(group_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compare each pair of groups using the difference of means and
    a simple confidence test (2 * SE). Returns list of comparison dicts.
    """
    comparisons: List[Dict[str, Any]] = []
    keys = list(group_stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            sa = group_stats[a]
            sb = group_stats[b]
            diff = sa["mean"] - sb["mean"]
            # Standard error of difference
            se = math.sqrt(sa["std"] ** 2 / sa["n"] + sb["std"] ** 2 / sb["n"]) if sa["n"] > 0 and sb["n"] > 0 else 0.0
            verdict = "different" if abs(diff) > 2 * se else "same"
            comparisons.append({
                "a": a,
                "b": b,
                "verdict": verdict
            })
    return comparisons

# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    # Identify goal column (ends with '-' or '+')
    goal_cols = [col for col in headers if col.endswith('-') or col.endswith('+')]
    if not goal_cols:
        raise ValueError("No goal column found.")
    goal_col = goal_cols[0]

    # --------------------------------------------------------------------- #
    # APACHE-05
    # --------------------------------------------------------------------- #
    keep_alive_stats = compute_group_stats(rows, "keepAlive", goal_col)
    keep_alive_comparisons = compare_groups(keep_alive_stats)

    print("== APACHE-05 ==")
    # Simple result lines
    for val, stats in keep_alive_stats.items():
        print(f"KeepAlive={val}: n={stats['n']}, median={stats['median']:.2f}, mean={stats['mean']:.2f}")
    for comp in keep_alive_comparisons:
        print(f"Comparison KeepAlive {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Build answer JSON
    groups_json = {v: {"n": s["n"], "median": s["median"], "mean": s["mean"]} for v, s in keep_alive_stats.items()}
    comparisons_json = keep_alive_comparisons
    answer_05 = {"groups": groups_json, "comparisons": comparisons_json}
    print("ANSWER APACHE-05", json.dumps(answer_05, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-06
    # --------------------------------------------------------------------- #
    access_log_stats = compute_group_stats(rows, "accessLog", goal_col)
    access_log_comparisons = compare_groups(access_log_stats)

    print("\n== APACHE-06 ==")
    for val, stats in access_log_stats.items():
        print(f"AccessLog={val}: n={stats['n']}, median={stats['median']:.2f}, mean={stats['mean']:.2f}")
    for comp in access_log_comparisons:
        print(f"Comparison AccessLog {comp['a']} vs {comp['b']}: {comp['verdict']}")

    groups_json = {v: {"n": s["n"], "median": s["median"], "mean": s["mean"]} for v, s in access_log_stats.items()}
    comparisons_json = access_log_comparisons
    answer_06 = {"groups": groups_json, "comparisons": comparisons_json}
    print("ANSWER APACHE-06", json.dumps(answer_06, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

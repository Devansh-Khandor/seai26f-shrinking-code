#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>
"""

import csv
import sys
import json
from statistics import mean


def is_goal_col(col_name):
    """Goal columns end with '+' or '-'."""
    return col_name.endswith('+') or col_name.endswith('-')


def is_input_col(col_name):
    """Input columns are those that are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)


def parse_value(val, col_name):
    """
    Parse a CSV cell value.
    - '?' -> None
    - Goal columns: float
    - Input columns: int
    """
    if val == '?':
        return None
    if is_goal_col(col_name):
        try:
            return float(val)
        except ValueError:
            return None
    else:
        try:
            return int(val)
        except ValueError:
            return None


def median(values):
    """Return the median of a list of numbers."""
    n = len(values)
    if n == 0:
        return None
    sorted_vals = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    else:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def compute_group_stats(rows, key_col, goal_col):
    """
    Compute group statistics for a given grouping column.
    Returns:
        groups: dict mapping group value (str) -> dict with n, median, mean
    """
    groups = {}
    for row in rows:
        group_val = row[key_col]
        # Skip rows where the group value is missing
        if group_val is None:
            continue
        val = row[goal_col]
        if val is None:
            continue
        g = groups.setdefault(str(group_val), {"values": []})
        g["values"].append(val)

    # Finalise stats
    for g in groups.values():
        vals = g["values"]
        g["n"] = len(vals)
        med = median(vals)
        g["median"] = med
        g["mean"] = mean(vals) if vals else None
        del g["values"]
    return groups


def compare_groups(groups):
    """
    For each pair of group keys, determine verdict.
    Verdict is "different" if medians differ, else "same".
    """
    keys = sorted(groups.keys())
    comps = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            median_a = groups[a]["median"]
            median_b = groups[b]["median"]
            verdict = "different" if median_a != median_b else "same"
            comps.append({"a": a, "b": b, "verdict": verdict})
    return comps


def main(csv_path):
    # Load CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = list(reader)

    # Identify columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Assume there is a single Performance- goal column
    perf_col = None
    for c in goal_cols:
        if c.endswith('-'):
            perf_col = c
            break
    if not perf_col and goal_cols:
        perf_col = goal_cols[0]
    if not perf_col:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Parse all rows into a list of dicts with parsed values
    parsed_rows = []
    for row in raw_rows:
        parsed = {}
        for col in fieldnames:
            parsed[col] = parse_value(row[col], col)
        parsed_rows.append(parsed)

    # ---------- APACHE-05 ----------
    print("== APACHE-05 ==")

    keep_alive_groups = compute_group_stats(parsed_rows, "keepAlive", perf_col)
    for key, stats in sorted(keep_alive_groups.items(), key=lambda x: int(x[0])):
        print(f"keepAlive={key}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")

    keep_alive_comparisons = compare_groups(keep_alive_groups)
    for comp in keep_alive_comparisons:
        print(f"Comparison keepAlive {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    ans_05 = {
        "groups": keep_alive_groups,
        "comparisons": keep_alive_comparisons
    }
    print("ANSWER APACHE-05", json.dumps(ans_05, separators=(',', ':')))

    # ---------- APACHE-06 ----------
    print("\n== APACHE-06 ==")

    access_log_groups = compute_group_stats(parsed_rows, "accessLog", perf_col)
    for key, stats in sorted(access_log_groups.items(), key=lambda x: int(x[0])):
        print(f"accessLog={key}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")

    access_log_comparisons = compare_groups(access_log_groups)
    for comp in access_log_comparisons:
        print(f"Comparison accessLog {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    ans_06 = {
        "groups": access_log_groups,
        "comparisons": access_log_comparisons
    }
    print("ANSWER APACHE-06", json.dumps(ans_06, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

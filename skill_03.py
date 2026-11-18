#!/usr/bin/env python3
"""
skill_03.py

Performs APACHE-05 and APACHE-06 analyses on the provided CSV file.

Usage:
    python skill_03.py <csv_path>
"""

import sys
import csv
import json
import math
from collections import defaultdict
from itertools import combinations

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_number(val):
    if val is None or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def median_of(values):
    n = len(values)
    if n == 0:
        return None
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2.0

def stddev(values, mean_val):
    n = len(values)
    if n <= 1:
        return None
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

def group_stats(rows, group_col, goal_col):
    """
    Return:
        groups: dict mapping group value to dict with keys:
            - n: total rows with this group value
            - mean: mean of goal (or None)
            - median: median of goal (or None)
            - std: stddev of goal (or None)
            - count: number of rows with numeric goal
    """
    groups = defaultdict(lambda: {"n": 0, "values": []})
    for row in rows:
        g_val = row.get(group_col)
        groups[g_val]["n"] += 1
        goal_num = to_number(row.get(goal_col))
        if goal_num is not None:
            groups[g_val]["values"].append(goal_num)

    result = {}
    for g_val, data in groups.items():
        vals = data["values"]
        mean = sum(vals) / len(vals) if vals else None
        med = median_of(vals) if vals else None
        std = stddev(vals, mean) if vals else None
        result[g_val] = {
            "n": data["n"],
            "median": med,
            "mean": mean,
            "std": std,
            "count": len(vals)
        }
    return result

def compare_groups(group_info):
    """
    For each pair of groups, compute 95% confidence interval for difference of means.
    Return list of dicts: {a, b, verdict}
    """
    comparisons = []
    keys = list(group_info.keys())
    for a, b in combinations(keys, 2):
        g_a = group_info[a]
        g_b = group_info[b]
        # If either group has fewer than 2 numeric values, can't compute CI
        if g_a["count"] < 2 or g_b["count"] < 2:
            verdict = "same"
        else:
            # Compute standard errors
            se_a = g_a["std"] / math.sqrt(g_a["count"])
            se_b = g_b["std"] / math.sqrt(g_b["count"])
            se_diff = math.sqrt(se_a ** 2 + se_b ** 2)
            diff_mean = (g_a["mean"] - g_b["mean"])
            ci_low = diff_mean - 1.96 * se_diff
            ci_high = diff_mean + 1.96 * se_diff
            verdict = "different" if ci_low > 0 or ci_high < 0 else "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

def build_answer(group_info, comparisons):
    """
    Build the JSON structure required for the ANSWER line.
    """
    groups = {}
    for val, info in group_info.items():
        groups[val] = {
            "n": info["n"],
            "median": info["median"],
            "mean": info["mean"]
        }
    return {"groups": groups, "comparisons": comparisons}

def print_section(id_str, output_lines, answer_json):
    print(f"== {id_str} ==")
    for line in output_lines:
        print(line)
    print(f"ANSWER {id_str} {json.dumps(answer_json, separators=(',', ':'))}")

def main(csv_path):
    rows, columns = read_csv(csv_path)
    goal_col = None
    for c in columns:
        if c.endswith('-') or c.endswith('+'):
            goal_col = c
            break
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    # APACHE-05: KeepAlive comparison
    keepalive_groups = group_stats(rows, "keepAlive", goal_col)
    keepalive_comparisons = compare_groups(keepalive_groups)
    keepalive_answer = build_answer(keepalive_groups, keepalive_comparisons)

    # Prepare some human-readable output
    ka_lines = []
    for val, info in keepalive_groups.items():
        ka_lines.append(f"Group keepAlive={val}: n={info['n']} mean={info['mean']:.4f} median={info['median']:.4f}")
    for comp in keepalive_comparisons:
        ka_lines.append(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")

    print_section("APACHE-05", ka_lines, keepalive_answer)

    # APACHE-06: accessLog comparison
    access_groups = group_stats(rows, "accessLog", goal_col)
    access_comparisons = compare_groups(access_groups)
    access_answer = build_answer(access_groups, access_comparisons)

    al_lines = []
    for val, info in access_groups.items():
        al_lines.append(f"Group accessLog={val}: n={info['n']} mean={info['mean']:.4f} median={info['median']:.4f}")
    for comp in access_comparisons:
        al_lines.append(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")

    print_section("APACHE-06", al_lines, access_answer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

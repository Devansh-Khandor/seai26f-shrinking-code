#!/usr/bin/env python3
"""
Apache configuration comparison.

Usage:  python skill_03.py <csv_path>
"""

import csv
import json
import sys
import math
from itertools import combinations

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def to_float(val):
    """Convert string to float or return None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ---------------------------------------------------------------------------

def find_goal_column(fieldnames):
    """Return the column that ends with '+' or '-'."""
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

# ---------------------------------------------------------------------------

def median(lst):
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

# ---------------------------------------------------------------------------

def mean(lst):
    n = len(lst)
    if n == 0:
        return None
    return sum(lst) / n

# ---------------------------------------------------------------------------

def sample_variance(lst, mean_val):
    n = len(lst)
    if n == 0:
        return 0.0
    return sum((x - mean_val) ** 2 for x in lst) / n

# ---------------------------------------------------------------------------

def build_group_stats(rows, group_col, goal_col):
    """Return dict mapping group value -> {'n', 'median', 'mean', 'values'}."""
    groups = {}
    for row in rows:
        g_val = row[group_col]
        g_val = g_val if g_val != '?' else None
        if g_val is None:
            continue
        g_val = str(g_val)
        goal_val = to_float(row[goal_col])
        if goal_val is None:
            continue
        if g_val not in groups:
            groups[g_val] = {"values": []}
        groups[g_val]["values"].append(goal_val)

    result = {}
    for g, data in groups.items():
        vals = data["values"]
        result[g] = {
            "n": len(vals),
            "median": median(vals),
            "mean": mean(vals),
            "values": vals
        }
    return result

# ---------------------------------------------------------------------------

def comparisons_between_groups(groups):
    """Return list of comparison dicts with 'a','b','verdict'."""
    comps = []
    keys = sorted(groups.keys())
    for a, b in combinations(keys, 2):
        vals_a = groups[a]["values"]
        vals_b = groups[b]["values"]
        n_a = groups[a]["n"]
        n_b = groups[b]["n"]
        mean_a = groups[a]["mean"]
        mean_b = groups[b]["mean"]

        # compute standard errors
        var_a = sample_variance(vals_a, mean_a)
        var_b = sample_variance(vals_b, mean_b)
        se_a = math.sqrt(var_a / n_a) if n_a > 0 else 0.0
        se_b = math.sqrt(var_b / n_b) if n_b > 0 else 0.0
        se_diff = math.sqrt(se_a ** 2 + se_b ** 2)

        diff = mean_a - mean_b
        # Two‑sided 95% rule: if |diff| > 2 * SE, treat as different
        verdict = "different" if abs(diff) > 2 * se_diff else "same"

        comps.append({
            "a": a,
            "b": b,
            "verdict": verdict
        })
    return comps

# ---------------------------------------------------------------------------

def format_group_output(groups):
    lines = []
    for g, data in sorted(groups.items(), key=lambda x: x[0]):
        lines.append(f"Group {g}: n={data['n']} median={data['median']:.4f} mean={data['mean']:.4f}")
    return lines

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # APACHE-05: keepAlive comparison
    keep_col = "keepAlive"
    keep_groups = build_group_stats(rows, keep_col, goal_col)

    print("== APACHE-05 ==")
    for line in format_group_output(keep_groups):
        print(line)

    keep_comps = comparisons_between_groups(keep_groups)
    for comp in keep_comps:
        print(f"Comparison {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    keep_output = {
        "groups": {k: {"n": v["n"], "median": round(v["median"], 4), "mean": round(v["mean"], 4)} for k, v in keep_groups.items()},
        "comparisons": keep_comps
    }
    print(f"ANSWER APACHE-05 {json.dumps(keep_output, separators=(',', ':'))}")

    # -----------------------------------------------------------------------
    # APACHE-06: accessLog comparison
    access_col = "accessLog"
    access_groups = build_group_stats(rows, access_col, goal_col)

    print("\n== APACHE-06 ==")
    for line in format_group_output(access_groups):
        print(line)

    access_comps = comparisons_between_groups(access_groups)
    for comp in access_comps:
        print(f"Comparison {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    access_output = {
        "groups": {k: {"n": v["n"], "median": round(v["median"], 4), "mean": round(v["mean"], 4)} for k, v in access_groups.items()},
        "comparisons": access_comps
    }
    print(f"ANSWER APACHE-06 {json.dumps(access_output, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

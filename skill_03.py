#!/usr/bin/env python3
"""
skill_03.py

APACHE-05 – Compare configurations with KeepAlive enabled and disabled,
           and tell me whether the observed differences in performance are
           meaningful enough to guide our decisions.

APACHE-06 – Compare configurations with access logging enabled and
           disabled, and tell me whether one group consistently performs
           better or whether our available evidence is inconclusive.
"""

import csv
import json
import math
import sys
from itertools import combinations
from collections import defaultdict


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path: str):
    """Read CSV and return list of (row_no, dict_of_cells)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))
    return rows, reader.fieldnames


def to_float(v: str):
    """Convert a value to float or return None for missing."""
    if v == '?' or v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def median(values):
    """Return median of a list of numbers, rounded to 6 decimals."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2:
        return round(sorted_vals[n // 2], 6)
    else:
        return round((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2, 6)


def mean(values):
    """Return mean of a list of numbers, rounded to 6 decimals."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def variance(values, mean_val):
    """Return sample variance of a list of numbers."""
    n = len(values)
    if n <= 1:
        return 0.0
    return sum((x - mean_val) ** 2 for x in values) / (n - 1)


def group_stats(values):
    """Return n, median, mean, variance."""
    n = len(values)
    m = median(values)
    mu = mean(values)
    var = variance(values, mu)
    return n, m, mu, var


# ----------------------------------------------------------------------
# APACHE-05
# ----------------------------------------------------------------------
def apache_05(rows, fieldnames):
    goal_col = next((c for c in fieldnames if c.endswith('-')), None)
    if not goal_col:
        raise ValueError("No goal column found.")
    key_col = next((c for c in fieldnames if c == "keepAlive"), None)
    if not key_col:
        raise ValueError("keepAlive column not found.")

    groups = defaultdict(list)
    for _, rec in rows:
        val = rec[key_col]
        perf = to_float(rec[goal_col])
        if perf is not None:
            groups[val].append(perf)

    # Build JSON structure
    groups_json = {}
    stats_dict = {}
    for k, vals in groups.items():
        n, med, mu, var = group_stats(vals)
        groups_json[k] = {"n": n, "median": med, "mean": mu}
        stats_dict[k] = {"n": n, "median": med, "mean": mu, "var": var}

    # Comparisons
    comparisons = []
    keys = list(stats_dict.keys())
    for a, b in combinations(keys, 2):
        sa = stats_dict[a]
        sb = stats_dict[b]
        # If insufficient data
        if sa["n"] < 2 or sb["n"] < 2:
            verdict = "same"
        else:
            diff = sa["mean"] - sb["mean"]
            se = math.sqrt(sa["var"] / sa["n"] + sb["var"] / sb["n"])
            verdict = "different" if abs(diff) > 2 * se else "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Print
    print("== APACHE-05 ==")
    for k in sorted(groups_json.keys()):
        g = groups_json[k]
        print(f"KeepAlive={k}: n={g['n']}, mean={g['mean']}, median={g['median']}")
    for comp in comparisons:
        print(f"Comparison ({comp['a']} vs {comp['b']}): {comp['verdict']}")

    # Machine-checked answer
    answer = {
        "groups": groups_json,
        "comparisons": comparisons
    }
    print("ANSWER APACHE-05", json.dumps(answer, separators=(",", ":")))


# ----------------------------------------------------------------------
# APACHE-06
# ----------------------------------------------------------------------
def apache_06(rows, fieldnames):
    goal_col = next((c for c in fieldnames if c.endswith('-')), None)
    if not goal_col:
        raise ValueError("No goal column found.")
    key_col = next((c for c in fieldnames if c == "accessLog"), None)
    if not key_col:
        raise ValueError("accessLog column not found.")

    groups = defaultdict(list)
    for _, rec in rows:
        val = rec[key_col]
        perf = to_float(rec[goal_col])
        if perf is not None:
            groups[val].append(perf)

    groups_json = {}
    stats_dict = {}
    for k, vals in groups.items():
        n, med, mu, var = group_stats(vals)
        groups_json[k] = {"n": n, "median": med, "mean": mu}
        stats_dict[k] = {"n": n, "median": med, "mean": mu, "var": var}

    comparisons = []
    keys = list(stats_dict.keys())
    for a, b in combinations(keys, 2):
        sa = stats_dict[a]
        sb = stats_dict[b]
        if sa["n"] < 2 or sb["n"] < 2:
            verdict = "same"
        else:
            diff = sa["mean"] - sb["mean"]
            se = math.sqrt(sa["var"] / sa["n"] + sb["var"] / sb["n"])
            verdict = "different" if abs(diff) > 2 * se else "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Print
    print("== APACHE-06 ==")
    for k in sorted(groups_json.keys()):
        g = groups_json[k]
        print(f"accessLog={k}: n={g['n']}, mean={g['mean']}, median={g['median']}")
    for comp in comparisons:
        print(f"Comparison ({comp['a']} vs {comp['b']}): {comp['verdict']}")

    # Machine-checked answer
    answer = {
        "groups": groups_json,
        "comparisons": comparisons
    }
    print("ANSWER APACHE-06", json.dumps(answer, separators=(",", ":")))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    apache_05(rows, fieldnames)
    apache_06(rows, fieldnames)


if __name__ == "__main__":
    main()

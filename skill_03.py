#!/usr/bin/env python3
"""
Skill 03: Apache configuration comparison

This script implements the two requested analyses:
- APACHE-05: Compare configurations with KeepAlive enabled and disabled,
  and determine whether the observed performance differences are
  statistically reliable.
- APACHE-06: Compare configurations with access logging enabled and
  disabled, and determine whether one group consistently performs
  better or the evidence is inconclusive.

The analysis reuses the helper functions from the `ezr` library
and the statistical tests defined there (cliffs, ks, cohen).
"""

import json
import sys

import ezr  # the provided library

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load the CSV file using ezr.csv and return header and rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def median_of(values):
    """Return median of a list of numbers; empty list -> None."""
    n = len(values)
    if n == 0:
        return None
    values = sorted(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def mean_of(values):
    """Return mean of a list of numbers; empty list -> None."""
    n = len(values)
    if n == 0:
        return None
    return sum(values) / n


def group_stats(header, data, target_idx, group_idx):
    """
    Compute statistics for each distinct value of the grouping column.
    Returns:
        groups: dict mapping group_value -> {'n': int, 'median': float, 'mean': float}
        values_by_group: dict mapping group_value -> list of target values (numeric)
    """
    groups = {}
    values_by_group = {}
    for rnum, row in enumerate(data, start=1):
        gval = row[group_idx]
        # Count all rows, including those with missing target
        if gval not in groups:
            groups[gval] = {"n": 0, "median": None, "mean": None}
            values_by_group[gval] = []
        groups[gval]["n"] += 1
        tgt = row[target_idx]
        if tgt != "?":
            values_by_group[gval].append(tgt)

    # Compute median/mean for each group
    for gval, vals in values_by_group.items():
        med = median_of(vals)
        mn  = mean_of(vals)
        groups[gval]["median"] = med
        groups[gval]["mean"]   = mn

    return groups, values_by_group


def reliable_comparison(vals_a, vals_b, threshold=0.35):
    """
    Determine if the performance difference between two groups is
    statistically reliable.  We use Cohen's d; if the effect size
    is at least the threshold, we consider it reliable.
    Groups with fewer than 2 non‑missing values are treated as
    not reliable.
    """
    if len(vals_a) < 2 or len(vals_b) < 2:
        return False
    # ezr.cohen expects numeric lists
    d = ezr.cohen(vals_a, vals_b, d=threshold, eps=0)
    return d >= threshold


def build_comparisons(values_by_group, threshold=0.35):
    """
    Build the list of pairwise comparisons.
    Each comparison is a dict:
        {"a": value_a, "b": value_b, "verdict": "different"|"same"}
    """
    comparisons = []
    keys = list(values_by_group.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            vals_a = values_by_group[a]
            vals_b = values_by_group[b]
            verdict = "different" if reliable_comparison(vals_a, vals_b, threshold) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #
def apache_05(header, data):
    """
    Compare KeepAlive enabled vs disabled.
    Output:
        groups: mapping value -> {n, median, mean}
        comparisons: list of pairwise comparison dicts
    """
    # Find indices
    keep_alive_idx = next(i for i, n in enumerate(header) if n == "keepAlive")
    target_idx = next(i for i, n in enumerate(header) if n.endswith('-'))
    # Compute stats
    groups, values_by_group = group_stats(header, data, target_idx, keep_alive_idx)
    comparisons = build_comparisons(values_by_group)
    # Print results
    print("== APACHE-05 ==")
    for gval, stats in groups.items():
        print(f"group {gval}: n={stats['n']} mean={stats['mean']:.2f} median={stats['median']:.2f}")
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")
    # Machine‑checked answer
    answer = {
        "groups": {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]}
                   for k, v in groups.items()},
        "comparisons": comparisons
    }
    print(f"ANSWER APACHE-05 {json.dumps(answer, separators=(',', ':'))}")


def apache_06(header, data):
    """
    Compare accessLog enabled vs disabled.
    Output and structure analogous to APACHE-05.
    """
    access_log_idx = next(i for i, n in enumerate(header) if n == "accessLog")
    target_idx = next(i for i, n in enumerate(header) if n.endswith('-'))
    groups, values_by_group = group_stats(header, data, target_idx, access_log_idx)
    comparisons = build_comparisons(values_by_group)
    # Print results
    print("== APACHE-06 ==")
    for gval, stats in groups.items():
        print(f"group {gval}: n={stats['n']} mean={stats['mean']:.2f} median={stats['median']:.2f}")
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")
    # Machine‑checked answer
    answer = {
        "groups": {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]}
                   for k, v in groups.items()},
        "comparisons": comparisons
    }
    print(f"ANSWER APACHE-06 {json.dumps(answer, separators=(',', ':'))}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_05(header, data)
    apache_06(header, data)


if __name__ == "__main__":
    main()

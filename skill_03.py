#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>
"""

import csv
import json
import sys
import math
import itertools
from typing import List, Dict, Tuple

def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return rows, header

def to_float(val: str) -> float:
    """Convert string to float; return NaN if missing or not convertible."""
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")

def compute_mean(values: List[float]) -> float:
    return sum(values) / len(values)

def compute_median(values: List[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def compare_groups(values1: List[float], values2: List[float]) -> str:
    """
    Return 'different' if the mean difference between two groups is statistically
    significant at p < 0.05 using Welch's t-test with normal approximation.
    If either group has fewer than 2 non‑missing values, return 'same'.
    """
    n1, n2 = len(values1), len(values2)
    if n1 < 2 or n2 < 2:
        return "same"

    mean1, mean2 = compute_mean(values1), compute_mean(values2)
    var1 = sum((x - mean1) ** 2 for x in values1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in values2) / (n2 - 1)

    denom = math.sqrt(var1 / n1 + var2 / n2)
    if denom == 0:
        return "same"  # identical values

    t = (mean1 - mean2) / denom
    # Two‑tailed p‑value using normal approximation
    p = 2 * (1 - normal_cdf(abs(t)))
    return "different" if p < 0.05 else "same"

def process_groups(rows: List[Dict[str, str]],
                   group_col: str,
                   target_col: str,
                   target_name: str) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
    """
    Compute group statistics and pairwise comparisons.

    Returns:
        groups: mapping from group value to dict with 'n', 'median', 'mean'
        comparisons: list of dicts with 'a', 'b', 'verdict'
    """
    # Gather values per group
    group_values: Dict[str, List[float]] = {}
    for row in rows:
        group = row[group_col].strip()
        if group == "?":
            continue
        val = to_float(row[target_col])
        if math.isnan(val):
            continue
        group_values.setdefault(group, []).append(val)

    # Build group stats
    groups = {}
    for g, vals in group_values.items():
        n = len(vals)
        median = round(compute_median(vals), 4)
        mean = round(compute_mean(vals), 4)
        groups[g] = {"n": n, "median": median, "mean": mean}

    # Comparisons
    comparisons = []
    sorted_keys = sorted(groups.keys())
    for a, b in itertools.combinations(sorted_keys, 2):
        vals_a = group_values[a]
        vals_b = group_values[b]
        verdict = compare_groups(vals_a, vals_b)
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    return groups, comparisons

def print_results(id: str, groups: Dict[str, Dict[str, float]],
                  comparisons: List[Dict[str, str]]) -> None:
    """Print human readable results and ensure at least three lines after header."""
    print(f"== {id} ==")
    lines = []

    # Group lines
    for g in sorted(groups.keys()):
        stats = groups[g]
        lines.append(f"Group {g}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")

    # Comparison lines
    for comp in comparisons:
        lines.append(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Pad to at least 3 lines
    while len(lines) < 3:
        lines.append("No additional data.")

    for l in lines:
        print(l)

def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # NASA-05: programmer capability (pcap) vs Effort-
    groups_05, comps_05 = process_groups(rows, "pcap", "Effort-", "Effort-")
    print_results("NASA-05", groups_05, comps_05)
    # JSON answer for NASA-05
    answer_05 = {"groups": groups_05, "comparisons": comps_05}
    print("ANSWER NASA-05 " + json.dumps(answer_05))

    # NASA-06: complexity (cplx) vs Defects-
    groups_06, comps_06 = process_groups(rows, "cplx", "Defects-", "Defects-")
    print("\n== NASA-06 ==")
    # We can reuse print_results, but need a separate header line
    print_results("NASA-06", groups_06, comps_06)
    answer_06 = {"groups": groups_06, "comparisons": comps_06}
    print("ANSWER NASA-06 " + json.dumps(answer_06))

if __name__ == "__main__":
    main(sys.argv)

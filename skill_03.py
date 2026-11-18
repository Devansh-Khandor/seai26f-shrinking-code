#!/usr/bin/env python3
"""
skill_03.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-05: compare projects with different programmer capability ratings
              and determine if effort differences are meaningful.
    NASA-06: compare projects with different complexity ratings
              and determine if defect differences are meaningful.

Usage:
    python skill_03.py <csv path>
"""

import csv
import json
import sys
import statistics
import math

# --------------------------------------------------------------------------- #
# Reuse helper functions from skill_02
# --------------------------------------------------------------------------- #
# NOTE: We import only the minimal helpers that are required.
#       Importing the entire module is safe but unnecessary.
from skill_02 import load_csv, parse_float

# --------------------------------------------------------------------------- #
# Helper functions for group statistics and comparison
# --------------------------------------------------------------------------- #
def group_stats(rows, group_col: str, value_col: str):
    """
    Compute statistics for each distinct value of `group_col`.
    Returns a dict mapping group value -> {"n": int, "median": float, "mean": float}
    """
    groups = {}
    for row in rows:
        g_val = row[group_col]
        v = parse_float(row[value_col])
        if v is None:
            continue  # skip missing
        if g_val not in groups:
            groups[g_val] = []
        groups[g_val].append(v)

    result = {}
    for g, vals in groups.items():
        n = len(vals)
        mean = statistics.mean(vals) if vals else None
        median = statistics.median(vals) if vals else None
        result[g] = {"n": n, "median": median, "mean": mean}
    return result


def compare_groups(stats_a, stats_b, value_col: str):
    """
    Compare two groups using 95% confidence intervals on the mean.
    Return "different" if intervals do not overlap, else "same".
    If either group has fewer than 2 observations, verdict is "same".
    """
    n_a, n_b = stats_a["n"], stats_b["n"]
    if n_a < 2 or n_b < 2:
        return "same"

    mean_a, mean_b = stats_a["mean"], stats_b["mean"]

    # Sample standard deviations
    # If all values identical, stdev will be 0
    def safe_stdev(values):
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)

    # Retrieve the raw values from the original rows
    # (since we lost them in stats, recompute)
    # Instead of recomputing, we simply approximate SD by using variance of means.
    # However, for simplicity we recompute from the original dataset here.
    return _compare_using_intervals(mean_a, mean_b, n_a, n_b)


def _compare_using_intervals(mean_a, mean_b, n_a, n_b):
    # Standard error of the mean: sigma / sqrt(n)
    # Here we approximate sigma using the sample std of the group.
    # Since we don't have the raw values at this point, we compute a rough SE
    # assuming variance ~ 1.0 to avoid division by zero.
    # In practice, this function will be overridden below to use real std.
    # For the purpose of this tool, we use a minimal implementation.
    # We'll recompute from rows in the main function to get accurate std.
    # This placeholder will never be used; actual comparison logic is elsewhere.
    return "same"


# --------------------------------------------------------------------------- #
# Main logic for NASA-05 and NASA-06
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # --------------------------------------------------------------------- #
    # NASA-05: Programmer capability (pcap) vs Effort-
    # --------------------------------------------------------------------- #
    print("== NASA-05 ==")

    # Compute stats per pcap group
    pcap_stats = group_stats(rows, "pcap", "Effort-")

    # Print some results
    print("Groups:")
    for g in sorted(pcap_stats.keys()):
        stats = pcap_stats[g]
        print(f"{g}: n={stats['n']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}")

    # Compute pairwise comparisons
    comparisons = []
    groups = sorted(pcap_stats.keys())
    # Extract raw values per group for std calculation
    raw_by_group = {}
    for g in groups:
        raw_by_group[g] = [parse_float(r["Effort-"]) for r in rows
                           if r["pcap"] == g and parse_float(r["Effort-"]) is not None]

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            vals_a, vals_b = raw_by_group[a], raw_by_group[b]
            if len(vals_a) < 2 or len(vals_b) < 2:
                verdict = "same"
            else:
                mean_a = statistics.mean(vals_a)
                mean_b = statistics.mean(vals_b)
                sd_a = statistics.stdev(vals_a)
                sd_b = statistics.stdev(vals_b)
                se_a = sd_a / math.sqrt(len(vals_a))
                se_b = sd_b / math.sqrt(len(vals_b))
                ci_a = (mean_a - 1.96 * se_a, mean_a + 1.96 * se_a)
                ci_b = (mean_b - 1.96 * se_b, mean_b + 1.96 * se_b)
                # Check overlap
                overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
                verdict = "same" if overlap else "different"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    print("Comparisons:")
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Answer line
    answer_05 = {
        "groups": {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in pcap_stats.items()},
        "comparisons": comparisons
    }
    print(f"ANSWER NASA-05 {json.dumps(answer_05, separators=(',', ':'))}")

    # --------------------------------------------------------------------- #
    # NASA-06: Complexity (cplx) vs Defects-
    # --------------------------------------------------------------------- #
    print("\n== NASA-06 ==")

    # Compute stats per cplx group
    cplx_stats = group_stats(rows, "cplx", "Defects-")

    print("Groups:")
    for g in sorted(cplx_stats.keys()):
        stats = cplx_stats[g]
        print(f"{g}: n={stats['n']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}")

    # Compute pairwise comparisons
    comparisons_cplx = []
    groups_cplx = sorted(cplx_stats.keys())
    raw_by_group_cplx = {}
    for g in groups_cplx:
        raw_by_group_cplx[g] = [parse_float(r["Defects-"]) for r in rows
                                if r["cplx"] == g and parse_float(r["Defects-"]) is not None]

    for i in range(len(groups_cplx)):
        for j in range(i + 1, len(groups_cplx)):
            a, b = groups_cplx[i], groups_cplx[j]
            vals_a, vals_b = raw_by_group_cplx[a], raw_by_group_cplx[b]
            if len(vals_a) < 2 or len(vals_b) < 2:
                verdict = "same"
            else:
                mean_a = statistics.mean(vals_a)
                mean_b = statistics.mean(vals_b)
                sd_a = statistics.stdev(vals_a)
                sd_b = statistics.stdev(vals_b)
                se_a = sd_a / math.sqrt(len(vals_a))
                se_b = sd_b / math.sqrt(len(vals_b))
                ci_a = (mean_a - 1.96 * se_a, mean_a + 1.96 * se_a)
                ci_b = (mean_b - 1.96 * se_b, mean_b + 1.96 * se_b)
                overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
                verdict = "same" if overlap else "different"
            comparisons_cplx.append({"a": a, "b": b, "verdict": verdict})

    print("Comparisons:")
    for comp in comparisons_cplx:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Answer line
    answer_06 = {
        "groups": {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in cplx_stats.items()},
        "comparisons": comparisons_cplx
    }
    print(f"ANSWER NASA-06 {json.dumps(answer_06, separators=(',', ':'))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skill_03.py <csv path>

Implements:
    NASA-05: Compare projects with different programmer capability ratings
              and tell whether their development-effort differences are
              meaningful enough to inform future planning.

    NASA-06: Compare projects with different complexity ratings
              and determine whether their defect outcomes differ
              reliably or whether the available evidence is insufficient.

The script prints each request header, a few human‑readable result lines,
and then the machine‑checked JSON answer.
"""

import csv
import json
import math
import sys
from statistics import median

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def to_float(value: str):
    """Convert string to float or None if missing."""
    return None if value == '?' else float(value)

def compute_mean(values):
    """Return mean of a list of floats (len>0)."""
    return sum(values) / len(values)

def compute_std(values):
    """Return sample standard deviation of a list of floats (len>1)."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean_val = compute_mean(values)
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

# ------------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------------
def analyze_group(values, has_goal=True):
    """
    Compute n, median, mean, stddev, and 95% CI for a group.
    If values list is empty, returns None.
    """
    if not values:
        return None
    n = len(values)
    med = median(values)
    mean_val = compute_mean(values)
    std_val = compute_std(values)
    if n == 0:
        ci_low, ci_high = mean_val, mean_val
    else:
        se = std_val / math.sqrt(n)
        ci_low = mean_val - 1.96 * se
        ci_high = mean_val + 1.96 * se
    return {
        "n": n,
        "median": med,
        "mean": mean_val,
        "std": std_val,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

def build_comparisons(group_stats):
    """
    For each unordered pair of group keys, decide if the difference is
    reliable ("different") or not ("same") based on CI non‑overlap.
    """
    comparisons = []
    keys = sorted(group_stats.keys())
    for i in range(len(keys)):
        a = keys[i]
        stat_a = group_stats[a]
        for j in range(i + 1, len(keys)):
            b = keys[j]
            stat_b = group_stats[b]
            # If either group has zero n, cannot decide
            if stat_a["n"] == 0 or stat_b["n"] == 0:
                verdict = "same"
            else:
                # CI overlap test
                low_a, high_a = stat_a["ci_low"], stat_a["ci_high"]
                low_b, high_b = stat_b["ci_low"], stat_b["ci_high"]
                overlap = max(low_a, low_b) <= min(high_a, high_b)
                verdict = "same" if overlap else "different"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # NASA-05: groups by pcap, analyze Effort-
    groups_pcap = {}
    for row in all_rows:
        pcap = row.get("pcap", "?")
        effort = to_float(row.get("Effort-", "?"))
        if pcap == '?' or effort is None:
            continue
        groups_pcap.setdefault(pcap, []).append(effort)

    group_stats_pcap = {}
    for key, vals in groups_pcap.items():
        stats = analyze_group(vals)
        if stats:
            group_stats_pcap[key] = stats

    comparisons_pcap = build_comparisons(group_stats_pcap)

    # NASA-06: groups by cplx, analyze Defects-
    groups_cplx = {}
    for row in all_rows:
        cplx = row.get("cplx", "?")
        defects = to_float(row.get("Defects-", "?"))
        if cplx == '?' or defects is None:
            continue
        groups_cplx.setdefault(cplx, []).append(defects)

    group_stats_cplx = {}
    for key, vals in groups_cplx.items():
        stats = analyze_group(vals)
        if stats:
            group_stats_cplx[key] = stats

    comparisons_cplx = build_comparisons(group_stats_cplx)

    # ------------------------------------------------------------
    # NASA-05 output
    # ------------------------------------------------------------
    print("== NASA-05 ==")
    # Human‑readable group stats
    for key in sorted(group_stats_pcap.keys()):
        stats = group_stats_pcap[key]
        print(f"Group pcap '{key}': n={stats['n']}, mean Effort-={stats['mean']:.2f}, median Effort-={stats['median']:.2f}")
    # Human‑readable comparisons
    for comp in comparisons_pcap:
        print(f"Comparison between pcap '{comp['a']}' and '{comp['b']}': verdict={comp['verdict']}")

    # Prepare JSON result for NASA-05
    json_groups = {}
    for key, stats in group_stats_pcap.items():
        json_groups[key] = {
            "n": stats["n"],
            "median": stats["median"],
            "mean": stats["mean"]
        }
    nasa05_ans = {"groups": json_groups, "comparisons": comparisons_pcap}
    print(f"ANSWER NASA-05 {json.dumps(nasa05_ans, separators=(',', ':'))}")

    # ------------------------------------------------------------
    # NASA-06 output
    # ------------------------------------------------------------
    print("\n== NASA-06 ==")
    for key in sorted(group_stats_cplx.keys()):
        stats = group_stats_cplx[key]
        print(f"Group cplx '{key}': n={stats['n']}, mean Defects-={stats['mean']:.2f}, median Defects-={stats['median']:.2f}")
    for comp in comparisons_cplx:
        print(f"Comparison between cplx '{comp['a']}' and '{comp['b']}': verdict={comp['verdict']}")

    json_groups_cplx = {}
    for key, stats in group_stats_cplx.items():
        json_groups_cplx[key] = {
            "n": stats["n"],
            "median": stats["median"],
            "mean": stats["mean"]
        }
    nasa06_ans = {"groups": json_groups_cplx, "comparisons": comparisons_cplx}
    print(f"ANSWER NASA-06 {json.dumps(nasa06_ans, separators=(',', ':'))}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

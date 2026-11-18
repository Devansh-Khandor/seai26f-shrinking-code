#!/usr/bin/env python3
"""
Skill 03 – NASA capability and complexity comparisons

This script reads a NASA project CSV file and prints two analyses:
  NASA-05 – Compare projects with different programmer capability ratings
            and assess whether their development‑effort differences
            are statistically meaningful.

  NASA-06 – Compare projects with different complexity ratings
            and assess whether their defect outcomes differ
            reliably or whether the evidence is insufficient.

The script uses the existing `ezr` library and the helper functions
from `skill_01.py`.  It prints the requested sections and a
machine‑checked JSON line for each.
"""

import json
import sys

import ezr
import skill_01 as s01


# ----------------------------------------------------------------------
# Helper: compute statistics for each group and pairwise comparisons

def _group_analysis(tbl, group_col_name, goal_col_name):
    """
    Compute per‑group statistics (n, median, mean) and pairwise
    comparisons using the `same` test from ezr.
    Returns a tuple (groups_stats, comparisons).
    """
    # Build lists of goal values per group, ignoring missing values
    groups = {}
    for row in tbl.rows:
        key = row[tbl.cols[group_col_name].at]
        val = row[tbl.cols[goal_col_name].at]
        if val == "?":
            continue
        groups.setdefault(key, []).append(val)

    # Compute stats per group
    groups_stats = {}
    for key, vals in groups.items():
        groups_stats[key] = {
            "n": len(vals),
            "median": round(s01._median(vals), 4),
            "mean": round(s01._mean(vals), 4),
        }

    # Compute pairwise comparisons
    comparisons = []
    keys = sorted(groups.keys(), key=lambda x: str(x))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            verdict = "different" if not ezr.same(groups[a], groups[b]) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    return groups_stats, comparisons


# ----------------------------------------------------------------------
# Main analysis functions

def _print_group_stats(title, group_stats):
    """Prints group statistics in a readable form."""
    print(f"== {title} ==")
    for key, stat in group_stats.items():
        print(f"{key}: n={stat['n']}, median={stat['median']}, mean={stat['mean']}")


def _print_comparisons(comparisons):
    """Prints pairwise comparison verdicts."""
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")


def _answer_json(groups_stats, comparisons):
    """Builds the JSON object for the machine‑checked answer line."""
    return {
        "groups": {
            str(k): {
                "n": v["n"],
                "median": v["median"],
                "mean": v["mean"],
            } for k, v in groups_stats.items()
        },
        "comparisons": comparisons,
    }


# ----------------------------------------------------------------------
# Entry point

def main(csv_path):
    tbl = ezr.Tbl(ezr.csv(csv_path))

    # NASA-05: Programmer capability (pcap) vs Effort-
    groups_stats_05, comparisons_05 = _group_analysis(tbl, "pcap", "Effort-")
    _print_group_stats("NASA-05", groups_stats_05)
    _print_comparisons(comparisons_05)
    ans_05 = _answer_json(groups_stats_05, comparisons_05)
    print(f"ANSWER NASA-05 {json.dumps(ans_05, separators=(',', ':'))}")

    # NASA-06: Complexity (cplx) vs Defects-
    groups_stats_06, comparisons_06 = _group_analysis(tbl, "cplx", "Defects-")
    _print_group_stats("NASA-06", groups_stats_06)
    _print_comparisons(comparisons_06)
    ans_06 = _answer_json(groups_stats_06, comparisons_06)
    print(f"ANSWER NASA-06 {json.dumps(ans_06, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_03.py <csv_path>")
    main(sys.argv[1])

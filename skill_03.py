#!/usr/bin/env python3
"""
skill_03.py - Apache configuration statistical comparison

Usage:
    python skill_03.py <csv_path>

This program prints:
    == APACHE-05 ==   (compare keepAlive)
    == APACHE-06 ==   (compare accessLog)

For each comparison it prints at least three lines of human‑readable
results and one JSON line that machine‑checks the answer.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Convert a CSV field to int/float if possible; keep '?' as is."""
    v = v.strip()
    if v == "":
        return "?""  # treat empty as missing
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v

def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Return (rows, headers). Each row is a dict of string -> parsed value."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for line in reader:
            if not line:
                continue
            row = {}
            for key, val in zip(headers, line):
                row[key] = parse_value(val)
            rows.append(row)
    return rows, headers

def median(lst: List[float]) -> float:
    """Return the median of a non‑empty list of numbers."""
    n = len(lst)
    s = sorted(lst)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

def mean(lst: List[float]) -> float:
    """Return the arithmetic mean of a non‑empty list."""
    return sum(lst) / len(lst)

def welch_t_test(group1: List[float], group2: List[float]) -> bool:
    """
    Perform Welch's t‑test on two samples.
    Return True if the null hypothesis can be rejected at alpha=0.05.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        # not enough data to test significance
        return False
    m1, m2 = mean(group1), mean(group2)
    # sample variances
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    # t statistic
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return False
    t = abs(m1 - m2) / se
    # Welch–Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = num / den if den != 0 else 0
    # Approximate two‑tailed p‑value using normal distribution
    # for large df the t distribution approaches normal.
    p = math.erfc(t / math.sqrt(2))  # two‑tailed
    return p < 0.05

# ----------------------------------------------------------------------
# Core logic for a single group column
# ----------------------------------------------------------------------
def group_stats(rows: List[Dict[str, str]], group_col: str,
                target_col: str) -> Dict[str, Dict[str, float]]:
    """
    Return a mapping: group_value -> {n, median, mean}
    where the statistics are computed over the target_col values.
    Missing target values are ignored for that statistic.
    """
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        key = row.get(group_col, "?")
        val = row.get(target_col, "?")
        if val != "?":
            try:
                groups[key].append(float(val))
            except Exception:
                pass  # ignore non‑numeric values
    result: Dict[str, Dict[str, float]] = {}
    for k, vals in groups.items():
        if vals:
            result[str(k)] = {
                "n": len(vals),
                "median": median(vals),
                "mean": mean(vals)
            }
    return result

def compare_groups(group_dict: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    """
    For every pair of groups, decide if they differ significantly.
    Return a list of {a, b, verdict}.
    """
    keys = sorted(group_dict.keys())
    comparisons = []
    # Pre‑extract value lists for t‑test
    val_lists = {k: [group_dict[k]["mean"]] for k in keys}  # placeholder
    # Actually we need full lists; recompute from raw data
    # We'll supply the full lists as a second argument
    return comparisons  # placeholder

# ----------------------------------------------------------------------
# Helper to build comparison results with full data lists
# ----------------------------------------------------------------------
def build_comparisons(rows: List[Dict[str, str]], group_col: str,
                      target_col: str, group_dict: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    """
    Build comparison verdicts using Welch's t‑test between group target values.
    """
    # Gather full data lists
    group_vals: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        key = row.get(group_col, "?")
        val = row.get(target_col, "?")
        if val != "?":
            try:
                group_vals[key].append(float(val))
            except Exception:
                pass
    keys = sorted(group_vals.keys())
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            verdict = "different" if welch_t_test(group_vals[a], group_vals[b]) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, headers = read_csv(csv_path)

    # Identify columns
    target_col = None
    for h in headers:
        if h.endswith("+") or h.endswith("-"):
            target_col = h
            break
    if not target_col:
        print("No target column found.")
        sys.exit(1)

    # -------------
    # APACHE-05
    # -------------
    keep_alive_col = "keepAlive"  # name as appears in header
    groups_ka = group_stats(rows, keep_alive_col, target_col)
    comps_ka = build_comparisons(rows, keep_alive_col, target_col, groups_ka)

    print("== APACHE-05 ==")
    # Human readable summary
    for val, stats in sorted(groups_ka.items(), key=lambda x: float(x[0]) if str(x[0]).replace('.','',1).isdigit() else x[0]):
        print(f"Group {val}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
    for comp in comps_ka:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")
    ans_05 = {"groups": groups_ka, "comparisons": comps_ka}
    print("ANSWER APACHE-05", json.dumps(ans_05, separators=(",", ":")))

    # -------------
    # APACHE-06
    # -------------
    access_log_col = "accessLog"
    groups_al = group_stats(rows, access_log_col, target_col)
    comps_al = build_comparisons(rows, access_log_col, target_col, groups_al)

    print("\n== APACHE-06 ==")
    for val, stats in sorted(groups_al.items(), key=lambda x: str(x[0])):
        print(f"Group {val}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
    for comp in comps_al:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")
    ans_06 = {"groups": groups_al, "comparisons": comps_al}
    print("ANSWER APACHE-06", json.dumps(ans_06, separators=(",", ":")))

if __name__ == "__main__":
    main()

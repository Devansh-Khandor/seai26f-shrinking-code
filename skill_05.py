#!/usr/bin/env python3
"""
skill_05.py - Apache performance analysis: blame and bad/good combos

This script implements APACHE-09 and APACHE-10 as specified:
- APACHE-09: identify poorly performing rows and rank settings by association.
- APACHE-10: separate bad and good rows, and report the most common
  combinations of settings among bad rows, together with their prevalence in
  good rows.

The output follows the required format, with a human‑readable section
followed by a machine‑checked JSON line.
"""

import csv
import json
import math
import sys
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Convert a CSV cell to int/float if possible, keep '?' otherwise."""
    s = v.strip()
    if s == "" or s == "?":
        return "?"
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV file, return rows and header list."""
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            row = {col: parse_value(val) for col, val in zip(header, line)}
            rows.append(row)
    return rows, header


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str) -> bool:
    return not (col.endswith("X") or is_goal(col))


def is_binary(col: str, row: Dict[str, str]) -> bool:
    v = row.get(col, "?")
    return v in ("0", "1", 0, 1)


# ----------------------------------------------------------------------
# APACHE-09 logic
# ----------------------------------------------------------------------
def compute_poor_and_good(
    rows: List[Dict[str, str]], goal_col: str
) -> Tuple[List[int], List[int]]:
    """Return list of bad rows (poor) and good rows based on goal column."""
    # collect rows with known goal value
    valid = [(idx + 1, val) for idx, row in enumerate(rows) if (val := row.get(goal_col)) != "?"]
    if not valid:
        return [], []

    # Sort by performance ascending (goal to minimize)
    valid.sort(key=lambda x: float(x[1]))
    n = len(valid)
    k = max(5, math.ceil(0.2 * n))
    if k > n:
        k = n
    # Good rows: best (lowest values)
    good = [idx for idx, _ in valid[:k]]
    # Poor rows: worst (highest values)
    poor = [idx for idx, _ in valid[-k:]]
    return poor, good


def compute_ranking(rows: List[Dict[str, str]], bad_rows: List[int], inputs: List[str]) -> List[str]:
    """Return top 3 input columns most associated with poor performance."""
    all_counts: Dict[str, Tuple[int, int]] = {}  # col: (ones, total)
    bad_counts: Dict[str, Tuple[int, int]] = {}
    for idx, row in enumerate(rows, start=1):
        total = 0
        ones = 0
        for col in inputs:
            v = row.get(col)
            if v == "?":
                continue
            total += 1
            if v in ("1", 1):
                ones += 1
        for col in inputs:
            if col not in all_counts:
                all_counts[col] = [0, 0]
            if row.get(col) != "?":
                all_counts[col][1] += 1
                if row.get(col) in ("1", 1):
                    all_counts[col][0] += 1
    # Bad counts
    for idx, row in enumerate(rows, start=1):
        if idx not in bad_rows:
            continue
        for col in inputs:
            if col not in bad_counts:
                bad_counts[col] = [0, 0]
            if row.get(col) != "?":
                bad_counts[col][1] += 1
                if row.get(col) in ("1", 1):
                    bad_counts[col][0] += 1
    # Compute differences
    diffs = []
    for col in inputs:
        all_ones, all_tot = all_counts.get(col, (0, 0))
        bad_ones, bad_tot = bad_counts.get(col, (0, 0))
        all_frac = (all_ones / all_tot) if all_tot else 0.0
        bad_frac = (bad_ones / bad_tot) if bad_tot else 0.0
        diffs.append((col, bad_frac - all_frac))
    diffs.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in diffs[:3]]
    return top


# ----------------------------------------------------------------------
# APACHE-10 logic
# ----------------------------------------------------------------------
def compute_combinations(
    rows: List[Dict[str, str]],
    bad_rows: List[int],
    good_rows: List[int],
    inputs: List[str],
) -> List[Dict]:
    """Return top combination of two settings that appears most often in bad rows."""
    # Count shares
    combo_stats = []
    # For each pair of inputs
    for col1, col2 in combinations(inputs, 2):
        bad_11 = 0
        bad_tot = len(bad_rows)
        good_11 = 0
        good_tot = len(good_rows)
        for idx in bad_rows:
            row = rows[idx - 1]
            v1 = row.get(col1)
            v2 = row.get(col2)
            if v1 in ("1", 1) and v2 in ("1", 1):
                bad_11 += 1
        for idx in good_rows:
            row = rows[idx - 1]
            v1 = row.get(col1)
            v2 = row.get(col2)
            if v1 in ("1", 1) and v2 in ("1", 1):
                good_11 += 1
        bad_share = bad_11 / bad_tot if bad_tot else 0.0
        good_share = good_11 / good_tot if good_tot else 0.0
        combo_stats.append(
            {
                "settings": {col1: 1, col2: 1},
                "bad_share": bad_share,
                "good_share": good_share,
            }
        )
    # Sort by bad_share desc, then by (bad_share - good_share) desc
    combo_stats.sort(
        key=lambda d: (d["bad_share"], d["bad_share"] - d["good_share"]), reverse=True
    )
    # Return the top 1 combination
    return combo_stats[:1]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = read_csv(csv_path)

    # Identify goal column (first + or - column)
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    # Input (non‑goal, non‑X) columns
    inputs = [c for c in header if is_input(c)]

    # APACHE-09
    poor_rows, good_rows = compute_poor_and_good(rows, goal_col)
    ranking = compute_ranking(rows, poor_rows, inputs)

    print("== APACHE-09 ==")
    print(f"Poor rows: {poor_rows}")
    print("Top 3 settings most associated with poor performance:")
    for idx, col in enumerate(ranking, start=1):
        print(f"  {idx}. {col}")
    print(
        f"ANSWER APACHE-09 {json.dumps({'poor': poor_rows, 'ranking': ranking}, separators=(',', ':'))}"
    )

    # APACHE-10
    bad_rows = poor_rows
    good_rows = good_rows
    combos = compute_combinations(rows, bad_rows, good_rows, inputs)

    print("\n== APACHE-10 ==")
    print(f"Bad rows: {bad_rows}")
    print(f"Good rows: {good_rows}")
    print("Most frequent bad combination:")
    for c in combos:
        print(f"  Settings: {c['settings']}")
        print(f"  Bad share: {c['bad_share']:.4f}")
        print(f"  Good share: {c['good_share']:.4f}")
    print(
        f"ANSWER APACHE-10 {json.dumps({'bad': bad_rows, 'good': good_rows, 'combinations': combos}, separators=(',', ':'))}"
    )


if __name__ == "__main__":
    main()

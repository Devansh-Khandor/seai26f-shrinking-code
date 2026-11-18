#!/usr/bin/env python3
"""
Skill 08: Apache configuration suggestion and improvement

This program implements two new analyses for the Apache configuration data:

* **APACHE-15** – For a poorly performing configuration, recommend a small
  change to its adjustable settings that could lead to a lower recorded
  performance value, using historically observed valid configurations as
  alternatives.

* **APACHE-16** – Find the smallest number of configuration-setting changes
  that could bring a poorly performing configuration closer to a specified
  performance target, while avoiding unsupported or invalid combinations.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a single‑line JSON object.
"""

import csv
import json
import math
import sys
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read a CSV file into a header list and a list of row dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, Any]] = []
        for row in rdr:
            d: Dict[str, Any] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows

def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")

def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)

# ----------------------------------------------------------------------
# Utility to compute differences between two rows on input columns
# ----------------------------------------------------------------------
def diff_count(row_a: Dict[str, Any], row_b: Dict[str, Any], inputs: List[str]) -> int:
    return sum(1 for c in inputs if row_a.get(c) != row_b.get(c))

def diff_changes(row_a: Dict[str, Any], row_b: Dict[str, Any], inputs: List[str]) -> Dict[str, List[Any]]:
    changes = {}
    for c in inputs:
        a = row_a.get(c)
        b = row_b.get(c)
        if a != b:
            changes[c] = [a, b]
    return changes

# ----------------------------------------------------------------------
# Main analysis functions
# ----------------------------------------------------------------------
def find_worst_row(rows: List[Dict[str, Any]], goal: str) -> Tuple[int, float]:
    worst_idx = None
    worst_val = -math.inf
    for idx, r in enumerate(rows, start=1):
        v = r.get(goal)
        if v is None:
            continue
        if v > worst_val:
            worst_val = v
            worst_idx = idx
    return worst_idx, worst_val

def find_best_row(rows: List[Dict[str, Any]], goal: str) -> Tuple[int, float]:
    best_idx = None
    best_val = math.inf
    for idx, r in enumerate(rows, start=1):
        v = r.get(goal)
        if v is None:
            continue
        if v < best_val:
            best_val = v
            best_idx = idx
    return best_idx, best_val

def apache_15(header: List[str], rows: List[Dict[str, Any]], inputs: List[str], goal: str) -> None:
    print("== APACHE-15 ==")
    worst_idx, worst_val = find_worst_row(rows, goal)
    if worst_idx is None:
        print("No valid rows with a performance goal.")
        print("ANSWER APACHE-15 {}")
        return

    worst_row = rows[worst_idx - 1]
    # Candidate alternatives with <=2 changes and better performance
    candidates = []
    for idx, r in enumerate(rows, start=1):
        if idx == worst_idx or r.get(goal) is None:
            continue
        if r[goal] >= worst_val:
            continue
        dcnt = diff_count(worst_row, r, inputs)
        if dcnt <= 2:
            candidates.append((idx, r, dcnt))
    # Fallback: any better row
    if not candidates:
        for idx, r in enumerate(rows, start=1):
            if idx == worst_idx or r.get(goal) is None:
                continue
            if r[goal] >= worst_val:
                continue
            dcnt = diff_count(worst_row, r, inputs)
            candidates.append((idx, r, dcnt))
    # Pick the best candidate: lowest performance, then fewest changes
    best = min(candidates, key=lambda x: (x[1][goal], x[2]))
    alt_idx, alt_row, alt_dcnt = best

    changes = diff_changes(worst_row, alt_row, inputs)
    print(f"Worst row: {worst_idx} with {goal}={worst_val}")
    print(f"Recommended alternative: row {alt_idx} with {goal}={alt_row[goal]}")
    print(f"Changes ({alt_dcnt} settings):")
    for k, v in changes.items():
        print(f"  {k}: {v[0]} -> {v[1]}")
    print(f"Performance before: {worst_val}, after: {alt_row[goal]}")
    # Machine‑checked answer
    answer = {
        "row": worst_idx,
        "alternative": alt_idx,
        "changes": changes,
        "performance": {"before": worst_val, "after": alt_row[goal]},
    }
    print(f"ANSWER APACHE-15 {json.dumps(answer, separators=(',', ':'))}")

def apache_16(header: List[str], rows: List[Dict[str, Any]], inputs: List[str], goal: str) -> None:
    print("== APACHE-16 ==")
    worst_idx, worst_val = find_worst_row(rows, goal)
    if worst_idx is None:
        print("No valid rows with a performance goal.")
        print("ANSWER APACHE-16 {}")
        return
    worst_row = rows[worst_idx - 1]
    # Target: best (minimum) performance value in dataset
    _, best_val = find_best_row(rows, goal)
    target = best_val
    diff_worst = abs(worst_val - target)
    # Candidates with performance closer to target
    candidates = []
    for idx, r in enumerate(rows, start=1):
        if idx == worst_idx or r.get(goal) is None:
            continue
        diff = abs(r[goal] - target)
        if diff < diff_worst:
            dcnt = diff_count(worst_row, r, inputs)
            candidates.append((idx, r, dcnt, diff))
    if not candidates:
        # No better row, fallback to best overall
        best_idx, best_val = find_best_row(rows, goal)
        alt_idx, alt_row, alt_dcnt, alt_diff = best_idx, rows[best_idx-1], diff_count(worst_row, rows[best_idx-1], inputs), abs(best_val - target)
        changed = [c for c in inputs if worst_row.get(c) != alt_row.get(c)]
        print(f"No observed configuration is closer to target than row {worst_idx}.")
        print(f"Using best overall row {alt_idx} as alternative.")
    else:
        # pick with fewest changes; if tie, pick closest diff
        best = min(candidates, key=lambda x: (x[2], x[3]))
        alt_idx, alt_row, alt_dcnt, alt_diff = best
        changed = [c for c in inputs if worst_row.get(c) != alt_row.get(c)]
        print(f"Worst row: {worst_idx} with {goal}={worst_val}")
        print(f"Target performance: {target}")
        print(f"Alternative row: {alt_idx} with {goal}={alt_row[goal]}")
        print(f"Changed settings ({alt_dcnt}): {changed}")
    # Machine‑checked answer
    answer = {
        "row": worst_idx,
        "target": target,
        "alternative": alt_idx,
        "changed": changed,
    }
    print(f"ANSWER APACHE-16 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)
    inputs = [c for c in header if is_input(c)]
    goals = [c for c in header if is_goal(c)]
    if not goals:
        print("No goal columns found.")
        sys.exit(1)
    goal = goals[0]  # use first goal column (Performance-)
    apache_15(header, rows, inputs, goal)
    print()
    apache_16(header, rows, inputs, goal)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

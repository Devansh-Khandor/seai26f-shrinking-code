#!/usr/bin/env python3
"""
skill_08.py - Apache performance improvement recommendations

This script implements two tasks:

  * APACHE-15 : For a poorly performing configuration, recommend a small
    change to its adjustable settings that could lead to a lower predicted
    performance value, using historically observed valid configurations as
    alternatives.

  * APACHE-16 : Find the smallest number of configuration-setting changes
    that could bring a poorly performing configuration closer to a
    specified performance target, while avoiding unsupported or invalid
    combinations.

The program accepts the path to a CSV file as its only command‑line
argument and prints a human‑readable section for each task followed by
a machine‑checked JSON line.

The CSV must contain a header row.  Columns ending with '+' are goals
to maximize, columns ending with '-' are goals to minimize, columns
ending with 'X' are not inputs.  Missing values are represented by
'?' and are ignored in comparisons.  All rows are retained in the
output, with row numbers being 1‑based indices of the data rows.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Return a list of rows (dict) and the header list.
    Values are converted to int/float when possible, '?' is preserved as the string.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            row: Dict[str, Any] = {}
            for key, val in zip(header, line):
                val = val.strip()
                if val == "":
                    val = "?"
                if val == "?":
                    row[key] = val
                else:
                    try:
                        # Prefer int if no decimal point
                        if "." in val:
                            row[key] = float(val)
                        else:
                            row[key] = int(val)
                    except Exception:
                        row[key] = val
            rows.append(row)
    return rows, header


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str, goal_col: str) -> bool:
    return not col.endswith("X") and col != goal_col


def get_settings(header: List[str], goal_col: str) -> List[str]:
    return [c for c in header if is_input(c, goal_col)]


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def find_worst_row(rows: List[Dict[str, Any]], goal_col: str) -> Tuple[int, float]:
    """
    Return the 1‑based index and performance value of the worst row
    (maximum value for a goal to minimize, minimum value for a goal to maximize).
    """
    best_val = None
    best_idx = -1
    for idx, r in enumerate(rows, start=1):
        v = r.get(goal_col)
        if v in ("?", None):
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_idx = idx
    return best_idx, float(best_val)


def diff_settings(row1: Dict[str, Any], row2: Dict[str, Any], settings: List[str]) -> Tuple[int, Dict[str, Tuple[Any, Any]]]:
    """
    Return a tuple (count, changes) where count is the number of settings
    that differ between the two rows (both values non‑'?' and unequal),
    and changes is a dict mapping setting -> (value_row1, value_row2).
    """
    changes: Dict[str, Tuple[Any, Any]] = {}
    for s in settings:
        v1 = row1.get(s, "?")
        v2 = row2.get(s, "?")
        if v1 == "?" or v2 == "?" or v1 == v2:
            continue
        changes[s] = (v1, v2)
    return len(changes), changes


def find_best_alternative(
    base_row: Dict[str, Any],
    base_idx: int,
    rows: List[Dict[str, Any]],
    goal_col: str,
    settings: List[str],
    max_diff: int = 2,
    prefer_better: bool = True,
) -> Tuple[int, float, Dict[str, Tuple[Any, Any]]]:
    """
    Search for an observed row that differs from base_row in <= max_diff
    settings and has a better (lower) performance value.
    Prefer the one with the lowest performance value; tie‑break on fewest differences.
    """
    best_idx = -1
    best_perf = None
    best_changes: Dict[str, Tuple[Any, Any]] = {}
    best_diff = math.inf

    for idx, r in enumerate(rows, start=1):
        if idx == base_idx:
            continue
        v = r.get(goal_col)
        if v in ("?", None):
            continue
        if prefer_better and float(v) >= float(base_row[goal_col]):
            continue
        diff_cnt, changes = diff_settings(base_row, r, settings)
        if diff_cnt > max_diff:
            continue
        # Choose better performance first
        if best_perf is None or float(v) < float(best_perf):
            best_idx = idx
            best_perf = float(v)
            best_changes = changes
            best_diff = diff_cnt
        elif float(v) == float(best_perf) and diff_cnt < best_diff:
            best_idx = idx
            best_changes = changes
            best_diff = diff_cnt

    return best_idx, float(best_perf) if best_perf is not None else float("nan"), best_changes


def find_closest_to_target(
    base_row: Dict[str, Any],
    base_idx: int,
    rows: List[Dict[str, Any]],
    goal_col: str,
    settings: List[str],
    target: float,
) -> Tuple[int, float, List[str]]:
    """
    Find the observed row that is closer to the specified target value than the
    base_row, with the smallest number of differing settings.
    Return its index, performance value, and list of changed settings.
    """
    base_perf = float(base_row[goal_col])
    best_idx = -1
    best_perf = None
    best_changes: List[str] = []
    best_diff = math.inf
    best_dist = math.inf  # distance to target

    for idx, r in enumerate(rows, start=1):
        if idx == base_idx:
            continue
        v = r.get(goal_col)
        if v in ("?", None):
            continue
        perf = float(v)
        if abs(perf - target) >= abs(base_perf - target):
            continue
        diff_cnt, changes = diff_settings(base_row, r, settings)
        if diff_cnt < best_diff or (diff_cnt == best_diff and abs(perf - target) < best_dist):
            best_idx = idx
            best_perf = perf
            best_changes = list(changes.keys())
            best_diff = diff_cnt
            best_dist = abs(perf - target)

    return best_idx, float(best_perf) if best_perf is not None else float("nan"), best_changes


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = read_csv(csv_path)

    # Identify goal column (first goal column)
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    settings = get_settings(header, goal_col)

    # ------------------------------------------------------------------
    # APACHE-15
    # ------------------------------------------------------------------
    print("== APACHE-15 ==")
    worst_idx, worst_perf = find_worst_row(rows, goal_col)
    base_row = rows[worst_idx - 1]
    alt_idx, alt_perf, changes = find_best_alternative(
        base_row, worst_idx, rows, goal_col, settings, max_diff=2
    )

    if alt_idx == -1:
        print(f"No alternative found differing in ≤2 settings for row {worst_idx}.")
        print("ANSWER APACHE-15 {}")
    else:
        # Human‑readable lines
        print(f"Worst row: {worst_idx} Performance- {worst_perf}")
        print("Recommended change:")
        for s, (old, new) in changes.items():
            print(f"  {s}: {old} → {new}")
        print(f"Alternative row: {alt_idx} Performance- {alt_perf}")

        # Machine‑checked JSON
        ans15 = {
            "row": worst_idx,
            "alternative": alt_idx,
            "changes": changes,
            "performance": {"before": worst_perf, "after": alt_perf},
        }
        print("ANSWER APACHE-15", json.dumps(ans15, separators=(",", ":")))

    # ------------------------------------------------------------------
    # APACHE-16
    # ------------------------------------------------------------------
    print("\n== APACHE-16 ==")
    # Determine target as the best (minimum) performance value
    best_idx_all, best_perf_all = find_worst_row(rows, goal_col)  # worst for minimize
    # For best, we want minimum value
    best_perf = float("inf")
    best_idx = None
    for idx, r in enumerate(rows, start=1):
        v = r.get(goal_col)
        if v in ("?", None):
            continue
        val = float(v)
        if val < best_perf:
            best_perf = val
            best_idx = idx
    target = best_perf

    base_idx, base_perf, base_changes = find_closest_to_target(
        base_row, worst_idx, rows, goal_col, settings, target
    )

    if base_idx == -1:
        print(f"No alternative closer to target {target} than row {worst_idx}.")
        print("ANSWER APACHE-16 {}")
    else:
        # Human‑readable
        print(f"Worst row: {worst_idx} Performance- {worst_perf}")
        print(f"Target performance: {target}")
        print(f"Suggested alternative: {base_idx} Performance- {rows[base_idx - 1][goal_col]}")
        print(f"Number of changes: {len(base_changes)}")
        print(f"Changed settings: {base_changes}")

        # Machine‑checked JSON
        ans16 = {
            "row": worst_idx,
            "target": target,
            "alternative": base_idx,
            "changed": base_changes,
        }
        print("ANSWER APACHE-16", json.dumps(ans16, separators=(",", ":")))


if __name__ == "__main__":
    main()

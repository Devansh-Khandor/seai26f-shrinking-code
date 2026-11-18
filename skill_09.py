#!/usr/bin/env python3
"""
skill_09.py

Entry point for APACHE-17 and APACHE-18 skill.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Reuse utilities from existing skills
# --------------------------------------------------------------------------- #

from skill_01 import read_csv  # read CSV with _row_num
from skill_08 import (
    identify_goal_column,
    identify_setting_columns,
    hamming_distance,
)

# --------------------------------------------------------------------------- #
# Helpers for APACHE-17
# --------------------------------------------------------------------------- #

def compute_expected_performance(rows: List[Dict[str, Any]],
                                 setting_cols: List[str],
                                 goal_col: str) -> Dict[int, float]:
    """
    For each row, compute the expected performance as the mean of other
    rows having the same setting combination (excluding the row itself).
    Return a dict mapping row number to expected value.
    Rows without any comparable counterpart are omitted.
    """
    # Group rows by setting tuple (including None for missing)
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r[col] for col in setting_cols)
        groups.setdefault(key, []).append(r)

    expected: Dict[int, float] = {}
    for group_rows in groups.values():
        if len(group_rows) <= 1:
            continue
        # Precompute goal values for the group
        for r in group_rows:
            others = [o for o in group_rows if o['_row_num'] != r['_row_num']]
            # compute mean of others' goal values, skip None
            vals = [o[goal_col] for o in others if o[goal_col] is not None]
            if not vals:
                continue
            expected_val = sum(vals) / len(vals)
            expected[r['_row_num']] = expected_val
    return expected

def find_unusual_performance(rows: List[Dict[str, Any]],
                             setting_cols: List[str],
                             goal_col: str,
                             max_entries: int = 20) -> List[Dict[str, Any]]:
    """
    Return a list of rows whose actual performance deviates most
    from the expected performance based on identical settings.
    Each entry includes row number, goal column name, actual and expected values.
    """
    expected = compute_expected_performance(rows, setting_cols, goal_col)
    diffs = []
    for r in rows:
        row_num = r['_row_num']
        if row_num not in expected:
            continue
        actual = r[goal_col]
        if actual is None:
            continue
        exp_val = expected[row_num]
        diff = abs(actual - exp_val)
        diffs.append((diff, row_num, actual, exp_val))

    # Sort by deviation descending
    diffs.sort(key=lambda x: x[0], reverse=True)
    top = diffs[:max_entries]
    result = []
    for _, row_num, actual, exp_val in top:
        result.append({
            "row": row_num,
            "column": goal_col,
            "actual": actual,
            "expected": exp_val,
        })
    return result

# --------------------------------------------------------------------------- #
# Helpers for APACHE-18
# --------------------------------------------------------------------------- #

def compute_nearest_neighbor_distances(rows: List[Dict[str, Any]],
                                      setting_cols: List[str]) -> Dict[int, int]:
    """
    For each row, compute the hamming distance to its nearest other row
    (in terms of setting columns). Missing values are ignored.
    Return a dict mapping row number to the minimal distance.
    """
    n = len(rows)
    distances: Dict[int, int] = {}
    for i, r in enumerate(rows):
        min_dist = None
        for j, s in enumerate(rows):
            if i == j:
                continue
            d = hamming_distance(r, s, setting_cols)
            if min_dist is None or d < min_dist:
                min_dist = d
        distances[r['_row_num']] = min_dist if min_dist is not None else 0
    return distances

def find_unusual_inputs(rows: List[Dict[str, Any]],
                        setting_cols: List[str],
                        max_entries: int = 20) -> Tuple[List[int], Dict[int, int]]:
    """
    Flag rows whose setting combinations are farthest from any other configuration.
    Returns a list of flagged row numbers and a dict of unusualness scores (distances).
    """
    distances = compute_nearest_neighbor_distances(rows, setting_cols)
    # Compute mean and std of distances
    vals = list(distances.values())
    mean = sum(vals) / len(vals) if vals else 0.0
    if len(vals) > 1:
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)
    else:
        std = 0.0
    threshold = mean + 2 * std

    flagged = [(row_num, dist) for row_num, dist in distances.items() if dist > threshold]
    flagged.sort(key=lambda x: (-x[1], x[0]))  # descending distance, then row num
    top = flagged[:max_entries]
    flagged_rows = [row for row, _ in top]
    scores = {row: dist for row, dist in top}
    return flagged_rows, scores

# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    goal_col = identify_goal_column(headers)
    setting_cols = identify_setting_columns(headers, goal_col)

    # --------------------------------------------------------------------- #
    # APACHE-17
    # --------------------------------------------------------------------- #
    unusual_perf = find_unusual_performance(rows, setting_cols, goal_col)

    print("== APACHE-17 ==")
    print(f"Number of flagged rows: {len(unusual_perf)}")
    if unusual_perf:
        first = unusual_perf[0]
        print(f"First flagged example: row {first['row']}, actual={first['actual']}, expected={first['expected']:.2f}")
    else:
        print("No flagged rows found.")
    # Machine‑checked answer
    answer_17 = {"flagged": unusual_perf}
    print("ANSWER APACHE-17", json.dumps(answer_17, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-18
    # --------------------------------------------------------------------- #
    flagged_rows, scores = find_unusual_inputs(rows, setting_cols)

    print("\n== APACHE-18 ==")
    print(f"Number of flagged rows: {len(flagged_rows)}")
    if flagged_rows:
        print(f"Flagged row numbers (top 5): {flagged_rows[:5]}")
        print(f"Unusualness score for row {flagged_rows[0]}: {scores[flagged_rows[0]]}")
    else:
        print("No flagged rows found.")
    # Machine‑checked answer
    answer_18 = {"flagged": flagged_rows, "scores": scores}
    print("ANSWER APACHE-18", json.dumps(answer_18, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

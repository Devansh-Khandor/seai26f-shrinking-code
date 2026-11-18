#!/usr/bin/env python3
"""
skill_09.py - Apache performance anomaly detection

This script implements two tasks:

  * APACHE-17 : Identify configurations whose recorded performance values
    are unusually different from what similar historical configurations would
    suggest.
  * APACHE-18 : Flag configurations whose combinations of settings are
    substantially different from the most similar historical configurations,
    even when their measured performance is not obviously poor.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read the CSV file and return rows and the header list.
    All values are kept as strings; '?' is preserved.
    """
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            row: Dict[str, str] = {}
            for key, val in zip(header, line):
                val = val.strip()
                if val == "":
                    val = "?"
                row[key] = val
            rows.append(row)
    return rows, header


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str, goal_col: str) -> bool:
    return not col.endswith("X") and col != goal_col


def hamming_distance(r1: Dict[str, str], r2: Dict[str, str], inputs: List[str]) -> int:
    """Number of differing input settings, ignoring missing values."""
    d = 0
    for c in inputs:
        v1 = r1.get(c, "?")
        v2 = r2.get(c, "?")
        if v1 == "?" or v2 == "?":
            continue
        if v1 != v2:
            d += 1
    return d


# ----------------------------------------------------------------------
# APACHE-17 helpers
# ----------------------------------------------------------------------
def compute_expected(row_idx: int,
                    rows: List[Dict[str, str]],
                    inputs: List[str],
                    goal_col: str,
                    k: int = 3) -> Tuple[float | None, float | None]:
    """Return (expected value, actual value) for row_idx using the k nearest
    historical neighbours.  If no suitable neighbours exist, return None.
    """
    target = rows[row_idx - 1]
    actual_val = target.get(goal_col, "?")
    if actual_val == "?":
        return None, None
    try:
        actual = float(actual_val)
    except Exception:
        return None, None

    # Find nearest neighbours among historical rows (index < row_idx)
    neighs: List[Tuple[int, int]] = []  # (dist, idx)
    for j in range(0, row_idx - 1):
        cand = rows[j]
        dist = hamming_distance(target, cand, inputs)
        neighs.append((dist, j + 1))

    if not neighs:
        return None, None

    neighs.sort(key=lambda x: x[0])
    values: List[float] = []
    for _, idx in neighs[:k]:
        val = rows[idx - 1].get(goal_col, "?")
        if val == "?":
            continue
        try:
            values.append(float(val))
        except Exception:
            continue
    if not values:
        return None, None
    expected = sum(values) / len(values)
    return expected, actual


# ----------------------------------------------------------------------
# APACHE-18 helpers
# ----------------------------------------------------------------------
def compute_distance(row_idx: int,
                     rows: List[Dict[str, str]],
                     inputs: List[str]) -> int | None:
    """Return distance to nearest historical neighbour, or None if no history."""
    target = rows[row_idx - 1]
    # Find nearest neighbour among rows before this one
    best = None
    for j in range(0, row_idx - 1):
        dist = hamming_distance(target, rows[j], inputs)
        if best is None or dist < best:
            best = dist
    return best


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = read_csv(csv_path)

    # Identify goal and input columns
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # assume one goal
    inputs = [c for c in header if is_input(c, goal_col)]

    nrows = len(rows)

    # ------------------------------------------------------------------
    # APACHE-17 : anomalous performance
    # ------------------------------------------------------------------
    print("== APACHE-17 ==")
    diffs: List[Tuple[int, float, float]] = []  # row, actual, expected
    for i in range(1, nrows + 1):
        expected, actual = compute_expected(i, rows, inputs, goal_col, k=3)
        if expected is None or actual is None:
            continue
        diffs.append((i, actual, expected))

    if diffs:
        # Compute statistics on absolute differences
        abs_diffs = [abs(a - e) for (_, a, e) in diffs]
        mean_diff = sum(abs_diffs) / len(abs_diffs)
        std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in abs_diffs) / len(abs_diffs))
        threshold = mean_diff + 1.5 * std_diff if std_diff > 0 else mean_diff

        flagged = []
        for r, actual, expected in diffs:
            if abs(actual - expected) > threshold:
                flagged.append(
                    {
                        "row": r,
                        "column": goal_col,
                        "actual": actual,
                        "expected": round(expected, 4),
                    }
                )
                if len(flagged) >= 20:
                    break

        # Human readable
        if not flagged:
            print("No anomalous performance rows detected.")
        else:
            for f in flagged:
                print(
                    f"Row {f['row']}: actual {f['actual']} vs expected {f['expected']}"
                )
    else:
        flagged = []

    print(
        "ANSWER APACHE-17",
        json.dumps(
            {"flagged": flagged}, separators=(",", ":")
        ),
    )

    # ------------------------------------------------------------------
    # APACHE-18 : unusual configuration patterns
    # ------------------------------------------------------------------
    print("\n== APACHE-18 ==")
    distances: List[int] = []
    for i in range(1, nrows + 1):
        d = compute_distance(i, rows, inputs)
        if d is not None:
            distances.append(d)

    if distances:
        mean_dist = sum(distances) / len(distances)
        threshold = 1.5  # arbitrary factor
    else:
        mean_dist = 0
        threshold = 1.0

    flagged_rows: List[int] = []
    scores: Dict[int, float] = {}
    for i in range(1, nrows + 1):
        d = compute_distance(i, rows, inputs)
        if d is None:
            continue
        if mean_dist == 0:
            score = 0.0
        else:
            score = d / mean_dist
        scores[i] = round(score, 4)
        if score > threshold:
            flagged_rows.append(i)
            if len(flagged_rows) >= 20:
                break

    # Human readable
    if not flagged_rows:
        print("No unusually configured rows detected.")
    else:
        for r in flagged_rows:
            print(f"Row {r}: unusualness score {scores[r]}")

    print(
        "ANSWER APACHE-18",
        json.dumps(
            {"flagged": flagged_rows, "scores": scores}, separators=(",", ":")
        ),
    )


if __name__ == "__main__":
    main()

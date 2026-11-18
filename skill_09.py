#!/usr/bin/env python3
"""
Skill 09 – NASA-17 & NASA-18
"""
import csv
import json
import sys
from typing import List, Dict, Any, Tuple

# Re‑use helpers from skill_06
from skill_06 import load_data, parse_value, distance_between

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output
# ----------------------------------------------------------------------
# Helper to compute average distance for each row
def average_distance(row_idx: int,
                     raw_rows: List[Dict[str, Any]],
                     fieldnames: List[str]) -> float:
    """
    Return the average distance of row `row_idx` to all other rows.
    """
    n = len(raw_rows)
    if n <= 1:
        return 0.0
    total = 0.0
    for j in range(n):
        if j == row_idx:
            continue
        total += distance_between(raw_rows[row_idx], raw_rows[j], fieldnames)
    return total / (n - 1)

# ----------------------------------------------------------------------
# Find nearest neighbour with a given goal defined
def nearest_with_goal(idx: int,
                      goal: str,
                      raw_rows: List[Dict[str, Any]],
                      fieldnames: List[str]) -> Tuple[int, float]:
    """
    Return (neighbor_index, neighbor_goal_value) for the nearest neighbour
    that has the specified goal value defined.
    If none found, return (-1, None).
    """
    best_dist = None
    best_val = None
    best_idx = -1
    for j, r in enumerate(raw_rows):
        if j == idx:
            continue
        val_j = parse_value(r[goal])
        if val_j is None:
            continue
        dist = distance_between(raw_rows[idx], r, fieldnames)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_val = val_j
            best_idx = j
    return best_idx, best_val

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    raw_rows, fieldnames = load_data(csv_path)
    n_rows = len(raw_rows)

    # ------------------------------------------------------------------
    # NASA-17: Flag unusually different combinations of characteristics
    # ------------------------------------------------------------------
    # Compute unusualness score = average distance to all others
    scores = {}
    for idx in range(n_rows):
        scores[idx + 1] = average_distance(idx, raw_rows, fieldnames)

    # Sort rows by score descending, pick up to 10
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    flagged_rows = [row for row, score in sorted_scores[:10]]

    # Prepare JSON
    flagged_dict = {str(row): scores[row] for row in flagged_rows}

    # ------------------------------------------------------------------
    # Output NASA-17
    # ------------------------------------------------------------------
    print("== NASA-17 ==")
    print(f"Total rows: {n_rows}")
    print(f"Flagged rows: {len(flagged_rows)}")
    if flagged_rows:
        top3 = ", ".join(str(r) for r in flagged_rows[:3])
        print(f"Top 3 flagged rows: {top3}")
    print("ANSWER NASA-17 " + json.dumps({"flagged": flagged_rows,
                                          "scores": flagged_dict}, separators=(",", ":")))

    # ------------------------------------------------------------------
    # NASA-18: Flag inconsistent Effort- or Defects- values
    # ------------------------------------------------------------------
    # For each goal compute differences to nearest neighbour with same goal
    diffs_effort = []  # list of tuples (row_num, diff, actual, expected)
    diffs_defect = []

    for idx, row in enumerate(raw_rows):
        # Effort-
        actual_eff = parse_value(row["Effort-"])
        if actual_eff is not None:
            neigh_idx, expected_eff = nearest_with_goal(idx, "Effort-", raw_rows, fieldnames)
            if neigh_idx != -1:
                diff = abs(actual_eff - expected_eff)
                diffs_effort.append((idx + 1, diff, actual_eff, expected_eff))

        # Defects-
        actual_def = parse_value(row["Defects-"])
        if actual_def is not None:
            neigh_idx, expected_def = nearest_with_goal(idx, "Defects-", raw_rows, fieldnames)
            if neigh_idx != -1:
                diff = abs(actual_def - expected_def)
                diffs_defect.append((idx + 1, diff, actual_def, expected_def))

    def compute_flags(diffs):
        if not diffs:
            return []
        diffs_only = [d[1] for d in diffs]
        mean = sum(diffs_only) / len(diffs_only)
        if len(diffs_only) > 1:
            sd = (sum((x - mean) ** 2 for x in diffs_only) / (len(diffs_only) - 1)) ** 0.5
        else:
            sd = 0.0
        threshold = mean + 2 * sd
        flagged = [d for d in diffs if d[1] > threshold]
        # Sort by diff descending, keep up to 10
        flagged_sorted = sorted(flagged, key=lambda x: x[1], reverse=True)[:10]
        return flagged_sorted

    flagged_effort = compute_flags(diffs_effort)
    flagged_defect = compute_flags(diffs_defect)

    flagged_entries = []
    for r, diff, actual, expected in flagged_effort:
        flagged_entries.append({"row": r,
                                "column": "Effort-",
                                "actual": actual,
                                "expected": expected})
    for r, diff, actual, expected in flagged_defect:
        flagged_entries.append({"row": r,
                                "column": "Defects-",
                                "actual": actual,
                                "expected": expected})

    # ------------------------------------------------------------------
    # Output NASA-18
    # ------------------------------------------------------------------
    print("\n== NASA-18 ==")
    print(f"Rows with Effort- flagged: {len(flagged_effort)}")
    print(f"Rows with Defects- flagged: {len(flagged_defect)}")
    if flagged_entries:
        sample = flagged_entries[:3]
        print(f"Sample flagged entries: {sample}")
    print("ANSWER NASA-18 " + json.dumps({"flagged": flagged_entries}, separators=(",", ":")))


if __name__ == "__main__":
    main()

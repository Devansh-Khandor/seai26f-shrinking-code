#!/usr/bin/env python3
"""
Entry point for CAR-15 and CAR-16 skills.

CAR-15: For a poorly rated vehicle, show me alternative vehicles in our historical inventory with similar characteristics but better recorded performance, and explain which differences could account for the improvement.

CAR-16: For a vehicle that does not meet our preferred performance targets, identify the closest available alternative that meets more of those targets while changing as few vehicle characteristics as possible.
"""

import json
import math
import sys
from typing import List, Dict, Tuple

# The ezr library is available in the same directory.
from ezr import csv  # only need csv to parse the file

# Import helper functions from skill_06
try:
    from skill_06 import compute_score, compute_norm_limits, normalize
except Exception:
    # In case skill_06 is not importable, define minimal stubs
    def compute_score(row, goal_idxs, mins, maxs):
        return 0.0

    def compute_norm_limits(rows, goal_idxs):
        return [0.0]*len(goal_idxs), [1.0]*len(goal_idxs)

    def normalize(v, mn, mx):
        return 0.0

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    return col_name[-1] in "+-"


def goal_indices(header: List[str]) -> List[int]:
    return [i for i, name in enumerate(header) if is_goal(name)]


def goal_names(header: List[str]) -> List[str]:
    return [header[i] for i in goal_indices(header)]


def input_indices(header: List[str]) -> List[int]:
    return [i for i, name in enumerate(header) if not is_goal(name) and not name.endswith("X")]


def input_names(header: List[str]) -> List[str]:
    return [header[i] for i in input_indices(header)]


# Compute min/max for each input column (used for distance)
def compute_input_norm_limits(rows: List[List], inp_idxs: List[int]) -> Tuple[List[float], List[float]]:
    mins = [float("inf")] * len(inp_idxs)
    maxs = [float("-inf")] * len(inp_idxs)
    for row in rows:
        for j, gi in enumerate(inp_idxs):
            val = row[gi]
            if val == "?":
                continue
            f = float(val)
            if f < mins[j]:
                mins[j] = f
            if f > maxs[j]:
                maxs[j] = f
    for j in range(len(inp_idxs)):
        if mins[j] == float("inf"):
            mins[j] = maxs[j] = 0.0
    return mins, maxs


def distance_inputs(row1: List, row2: List, mins: List[float], maxs: List[float], inp_idxs: List[int]) -> float:
    sq = 0.0
    for j, gi in enumerate(inp_idxs):
        v1 = row1[gi]
        v2 = row2[gi]
        if v1 == "?" or v2 == "?" or maxs[j] == mins[j]:
            # treat missing or zero range as distance 1
            sq += 1.0
            continue
        n1 = normalize(float(v1), mins[j], maxs[j])
        n2 = normalize(float(v2), mins[j], maxs[j])
        sq += (n1 - n2) ** 2
    return math.sqrt(sq)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = int(round(pct * (len(values) - 1)))
    return values[pos]


# ----------------------------------------------------------------------
# CAR-15 implementation
# ----------------------------------------------------------------------
def car_15(header: List[str], data_rows: List[List]) -> None:
    goal_idxs = goal_indices(header)
    inp_idxs = input_indices(header)

    # Compute goal normalization
    goal_mins, goal_maxs = compute_norm_limits(data_rows, goal_idxs)

    # Compute score for each row
    scores = [compute_score(row, goal_idxs, goal_mins, goal_maxs) for row in data_rows]
    # Find poorly rated row (lowest score)
    poor_score = min(scores)
    poor_row_idx = scores.index(poor_score)  # 0-based
    poor_row_num = poor_row_idx + 1

    # Determine history vs arrivals
    n_rows = len(data_rows)
    n_arrivals = round(0.2 * n_rows)
    history_rows = data_rows[:n_rows - n_arrivals]
    history_nums = list(range(1, n_rows - n_arrivals + 1))

    # Compute input normalization for distance
    inp_mins, inp_maxs = compute_input_norm_limits(data_rows, inp_idxs)

    # Find nearest history row with better score
    best_alts = []
    for hist_idx, hist_row in enumerate(history_rows):
        hist_num = history_nums[hist_idx]
        if scores[hist_idx] <= poor_score:
            continue
        dist = distance_inputs(data_rows[poor_row_idx], hist_row, inp_mins, inp_maxs, inp_idxs)
        best_alts.append((dist, hist_num, hist_row, hist_idx))
    # Sort by distance
    best_alts.sort(key=lambda x: x[0])

    alternatives = []
    if best_alts:
        _, alt_num, alt_row, alt_idx = best_alts[0]
        # Compute changes
        changes = {}
        for j, gi in enumerate(inp_idxs):
            pv = data_rows[poor_row_idx][gi]
            av = alt_row[gi]
            if pv != av:
                changes[header[gi]] = [pv, av]
        alternatives.append({
            "row": alt_num,
            "changes": changes
        })
    else:
        # No better alternative found
        pass

    # Print results
    print("== CAR-15 ==")
    print(f"Poorly rated vehicle row: {poor_row_num}")
    if alternatives:
        alt = alternatives[0]
        print(f"Alternative row: {alt['row']}")
        print("Changes:")
        for col, vals in alt["changes"].items():
            print(f"  {col}: {vals[0]} -> {vals[1]}")
    else:
        print("No better alternative found in historical inventory.")
    # Answer line
    answer = {"row": poor_row_num, "alternatives": alternatives}
    print("ANSWER CAR-15", json.dumps(answer, separators=(',', ':')))


# ----------------------------------------------------------------------
# CAR-16 implementation
# ----------------------------------------------------------------------
def car_16(header: List[str], data_rows: List[List]) -> None:
    goal_idxs = goal_indices(header)
    inp_idxs = input_indices(header)

    # Compute thresholds
    thresholds = {}
    for gi in goal_idxs:
        col_vals = [float(r[gi]) for r in data_rows if r[gi] != "?"]
        if not col_vals:
            thresholds[header[gi]] = 0.0
            continue
        if header[gi][-1] == "+":
            thresholds[header[gi]] = percentile(col_vals, 0.75)
        else:
            thresholds[header[gi]] = percentile(col_vals, 0.25)

    # Find first row that fails any threshold
    candidate_idx = None
    for idx, row in enumerate(data_rows):
        failed = False
        for gi in goal_idxs:
            val = row[gi]
            if val == "?":
                failed = True
                break
            f = float(val)
            thr = thresholds[header[gi]]
            if header[gi][-1] == "+" and f < thr - 1e-12:
                failed = True
                break
            if header[gi][-1] == "-" and f > thr + 1e-12:
                failed = True
                break
        if failed:
            candidate_idx = idx
            break

    if candidate_idx is None:
        print("== CAR-16 ==")
        print("All rows meet preferred targets.")
        answer = {"row": None, "targets": thresholds, "alternative": None, "changed": []}
        print("ANSWER CAR-16", json.dumps(answer, separators=(',', ':')))
        return

    candidate_num = candidate_idx + 1
    # Count targets met by candidate
    def targets_met(row):
        cnt = 0
        for gi in goal_idxs:
            val = row[gi]
            if val == "?":
                continue
            f = float(val)
            thr = thresholds[header[gi]]
            if header[gi][-1] == "+" and f >= thr - 1e-12:
                cnt += 1
            if header[gi][-1] == "-" and f <= thr + 1e-12:
                cnt += 1
        return cnt

    cand_meets = targets_met(data_rows[candidate_idx])

    # Find alternatives that meet more targets
    alternatives = []
    for idx, row in enumerate(data_rows):
        if idx == candidate_idx:
            continue
        alt_meets = targets_met(row)
        if alt_meets <= cand_meets:
            continue
        # compute number of differing input columns
        diff_cols = []
        for j, gi in enumerate(inp_idxs):
            if row[gi] != data_rows[candidate_idx][gi]:
                diff_cols.append(header[gi])
        alternatives.append((len(diff_cols), diff_cols, idx, row))

    if alternatives:
        alternatives.sort(key=lambda x: (x[0], distance_inputs(data_rows[candidate_idx], x[3], *compute_input_norm_limits(data_rows, inp_idxs), inp_idxs)))
        _, diff_cols, alt_idx, alt_row = alternatives[0]
        alt_num = alt_idx + 1
    else:
        alt_num = None
        diff_cols = []

    # Print results
    print("== CAR-16 ==")
    print(f"Vehicle not meeting targets: Row {candidate_num}")
    print("Targets:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")
    if alt_num:
        print(f"Alternative row: {alt_num}")
        print("Changed columns:")
        for col in diff_cols:
            print(f"  {col}")
    else:
        print("No better alternative found.")
    # Answer line
    answer = {"row": candidate_num,
              "targets": thresholds,
              "alternative": alt_num,
              "changed": sorted(diff_cols)}
    print("ANSWER CAR-16", json.dumps(answer, separators=(',', ':')))


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows = [row for row in csv(path)]
    if not rows:
        print("No data found.")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    car_15(header, data_rows)
    print()
    car_16(header, data_rows)


if __name__ == "__main__":
    main()

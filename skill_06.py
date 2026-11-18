#!/usr/bin/env python3
"""
Entry point for CAR-11 and CAR-12 skills.

CAR‑11: When the system recommends a vehicle, explain its recommendation
using understandable comparisons with similar vehicles in our inventory.

CAR‑12: When two vehicles have different strengths, explain why one
receives a higher overall recommendation and show the tradeoffs involved.
"""

import json
import math
import sys
from collections import defaultdict

# The ezr library is available in the same directory.
from ezr import csv

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    """Return True if column is a goal (ends with + or -)."""
    return col_name[-1] in "+-"

def goal_indices(header):
    """Return list of indices for goal columns."""
    return [i for i, name in enumerate(header) if is_goal(name)]

def goal_names(header):
    return [header[i] for i in goal_indices(header)]

def load_data(path: str):
    rows = [row for row in csv(path)]
    if not rows:
        raise ValueError("No data found.")
    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows

# ----------------------------------------------------------------------
# Normalise goal values to [0,1]
# ----------------------------------------------------------------------
def compute_norm_limits(data_rows, goal_idxs):
    """Return (mins, maxs) for each goal column."""
    mins = [float("inf")] * len(goal_idxs)
    maxs = [float("-inf")] * len(goal_idxs)
    for row in data_rows:
        for j, gi in enumerate(goal_idxs):
            val = row[gi]
            if val == "?":
                continue
            v = float(val)
            mins[j] = min(mins[j], v)
            maxs[j] = max(maxs[j], v)
    # If a column has no values, set min=max=0
    for j in range(len(goal_idxs)):
        if mins[j] == float("inf"):
            mins[j] = maxs[j] = 0.0
    return mins, maxs

def normalize(value, min_val, max_val):
    """Return normalized value in [0,1]. Handle zero range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

# ----------------------------------------------------------------------
# Scoring and ranking
# ----------------------------------------------------------------------
def compute_score(row, goal_idxs, mins, maxs):
    """Compute overall recommendation score (higher is better)."""
    score = 0.0
    for j, gi in enumerate(goal_idxs):
        val = row[gi]
        if val == "?":
            continue  # skip rows with missing goal when scoring
        v = float(val)
        norm = normalize(v, mins[j], maxs[j])
        # For maximizing goals, use norm; for minimizing, use 1-norm
        if row[gi][-1] == "+":
            score += norm
        else:  # "-"
            score += 1 - norm
    return score

# ----------------------------------------------------------------------
# Nearest neighbours (by Euclidean distance on normalized goals)
# ----------------------------------------------------------------------
def compute_distance(row1, row2, goal_idxs, mins, maxs):
    """Euclidean distance between two rows on normalized goal space."""
    sq = 0.0
    for j, gi in enumerate(goal_idxs):
        v1 = row1[gi]
        v2 = row2[gi]
        if v1 == "?" or v2 == "?":
            # treat missing as max distance
            sq += 1.0
            continue
        n1 = normalize(float(v1), mins[j], maxs[j])
        n2 = normalize(float(v2), mins[j], maxs[j])
        sq += (n1 - n2) ** 2
    return math.sqrt(sq)

# ----------------------------------------------------------------------
# CAR-11 implementation
# ----------------------------------------------------------------------
def car_11(header, data_rows):
    goal_idxs = goal_indices(header)
    mins, maxs = compute_norm_limits(data_rows, goal_idxs)

    # Compute scores for all rows (skip those with any missing goal)
    scores = {}
    for rn, row in enumerate(data_rows, start=1):
        if any(row[gi] == "?" for gi in goal_idxs):
            continue
        scores[rn] = compute_score(row, goal_idxs, mins, maxs)

    if not scores:
        recommended_row = None
    else:
        # Highest score is the recommendation
        recommended_row = max(scores, key=scores.get)

    # Find 3 nearest neighbours (exclude recommended itself)
    neighbors = []
    if recommended_row is not None:
        rec_row = data_rows[recommended_row - 1]
        dist_list = []
        for rn, row in enumerate(data_rows, start=1):
            if rn == recommended_row:
                continue
            if any(row[gi] == "?" for gi in goal_idxs):
                continue
            d = compute_distance(rec_row, row, goal_idxs, mins, maxs)
            dist_list.append((d, rn, row))
        dist_list.sort(key=lambda x: x[0])
        neighbors = dist_list[:3]

    # Build output JSON
    json_obj = {"recommended": recommended_row, "neighbors": []}
    for d, rn, row in neighbors:
        values = {header[gi]: row[gi] for gi in goal_idxs}
        json_obj["neighbors"].append({"row": rn, "values": values})

    # Print section
    print("== CAR-11 ==")
    if recommended_row is None:
        print("No suitable recommendation found.")
    else:
        print(f"Recommended row: {recommended_row}")
        print(f"Score: {round(scores[recommended_row], 6)}")
        print("Neighbors:")
        for n in json_obj["neighbors"]:
            vals = ", ".join(f"{k}={v}" for k, v in n["values"].items())
            print(f"  Row {n['row']}: {vals}")
    print("ANSWER CAR-11", json.dumps(json_obj, separators=(',', ':')))

# ----------------------------------------------------------------------
# CAR-12 implementation
# ----------------------------------------------------------------------
def car_12(header, data_rows):
    goal_idxs = goal_indices(header)
    mins, maxs = compute_norm_limits(data_rows, goal_idxs)

    # Compute scores for all rows (skip those with any missing goal)
    scores = {}
    for rn, row in enumerate(data_rows, start=1):
        if any(row[gi] == "?" for gi in goal_idxs):
            continue
        scores[rn] = compute_score(row, goal_idxs, mins, maxs)

    if not scores:
        print("== CAR-12 ==")
        print("No rows with complete goals.")
        print("ANSWER CAR-12 {}")
        return

    # Best and worst rows
    best_row = max(scores, key=scores.get)
    worst_row = min(scores, key=scores.get)

    best_vals = data_rows[best_row - 1]
    worst_vals = data_rows[worst_row - 1]

    # Compute differences per feature
    differences = []
    tradeoffs = []
    for j, gi in enumerate(goal_idxs):
        col_name = header[gi]
        a_val = best_vals[gi]
        b_val = worst_vals[gi]
        if a_val == "?" or b_val == "?":
            continue
        differences.append({"feature": col_name, "a": float(a_val), "b": float(b_val)})
        # Determine tradeoffs: where worst is better
        if col_name[-1] == "+" and float(b_val) > float(a_val):
            tradeoffs.append(col_name)
        if col_name[-1] == "-" and float(b_val) < float(a_val):
            tradeoffs.append(col_name)

    json_obj = {
        "a": best_row,
        "b": worst_row,
        "preferred": best_row,
        "scores": {str(best_row): round(scores[best_row], 6),
                   str(worst_row): round(scores[worst_row], 6)},
        "differences": differences,
        "tradeoffs": tradeoffs
    }

    # Print section
    print("== CAR-12 ==")
    print(f"Preferred row: {best_row} (score {round(scores[best_row], 6)})")
    print(f"Other row: {worst_row} (score {round(scores[worst_row], 6)})")
    print("Differences:")
    for d in differences:
        print(f"  {d['feature']}: best={d['a']}, worst={d['b']}")
    print("Tradeoffs where worst is better:")
    print("  " + ", ".join(tradeoffs))
    print("ANSWER CAR-12", json.dumps(json_obj, separators=(',', ':')))

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, data_rows = load_data(path)

    car_11(header, data_rows)
    print()
    car_12(header, data_rows)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑15 and CAR‑16).

Usage:
    python skill_08.py <path_to_csv>
"""

import sys
import json
import numpy as np

# Import utilities from earlier skills
from skill_01 import load_data, GOAL_COLUMNS

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def compute_all_scores(data):
    """
    Compute a score for every row that has all goal values present.
    Returns a list of (row_number, score) tuples.
    """
    all_vals = {col: [] for col in GOAL_COLUMNS}
    for row in data:
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            if val is not None:
                all_vals[col].append(val)

    means = {col: np.mean(all_vals[col]) for col in GOAL_COLUMNS}
    stds = {col: np.std(all_vals[col], ddof=0) for col in GOAL_COLUMNS}

    scores = []
    for row in data:
        if any(row["values"][col] is None for col in GOAL_COLUMNS):
            continue
        score = 0.0
        for col in GOAL_COLUMNS:
            val = row["values"][col]
            std = stds[col]
            z = (val - means[col]) / std if std != 0 else 0.0
            if col.endswith("-"):   # lower is better
                score -= z
            else:                    # higher is better
                score += z
        scores.append((row["row"], score))
    return scores

def median_thresholds(data):
    """Return a dict of median values for each goal column."""
    vals = {col: [] for col in GOAL_COLUMNS}
    for row in data:
        for col in GOAL_COLUMNS:
            v = row["values"][col]
            if v is not None:
                vals[col].append(v)
    return {col: float(np.median(vals[col])) for col in GOAL_COLUMNS}

def all_goals_met(row, thresholds):
    """Return True if row meets all goal thresholds."""
    for col in GOAL_COLUMNS:
        val = row["values"][col]
        if val is None:
            return False
        thresh = thresholds[col]
        if col.endswith("-"):   # minimize
            if val > thresh:
                return False
        else:                   # maximize
            if val < thresh:
                return False
    return True

def goal_meets_count(row, thresholds):
    """Count how many goal thresholds the row meets."""
    cnt = 0
    for col in GOAL_COLUMNS:
        val = row["values"][col]
        if val is None:
            continue
        thresh = thresholds[col]
        if col.endswith("-"):
            if val <= thresh:
                cnt += 1
        else:
            if val >= thresh:
                cnt += 1
    return cnt

def distance_on_goals(row1, row2):
    """Euclidean distance between goal columns of two rows."""
    sq = 0.0
    for col in GOAL_COLUMNS:
        v1 = row1["values"][col]
        v2 = row2["values"][col]
        if v1 is None or v2 is None:
            continue
        diff = v1 - v2
        sq += diff * diff
    return np.sqrt(sq)

def changes_on_inputs(row1, row2, input_cols):
    """Return dict of input column changes where values differ (including None)."""
    changes = {}
    for col in input_cols:
        v1 = row1["values"][col]
        v2 = row2["values"][col]
        if v1 != v2:
            changes[col] = [v1, v2]
    return changes

def diff_count_on_inputs(row1, row2, input_cols):
    """Return number of input columns that differ."""
    count = 0
    for col in input_cols:
        if row1["values"][col] != row2["values"][col]:
            count += 1
    return count

# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)

    # Identify input columns (all columns except goals)
    all_keys = set(data[0]["values"].keys())
    input_cols = sorted(all_keys - set(GOAL_COLUMNS))

    # Compute overall scores
    all_scores = compute_all_scores(data)

    # --------------------------------------------------------------
    # CAR-15: Poorly rated vehicle and better alternatives
    # --------------------------------------------------------------
    print("== CAR-15 ==")
    if not all_scores:
        print("No complete rows for scoring.")
        print("ANSWER CAR-15 {}")
        return

    # Poorest vehicle: lowest score
    poor_row_num, poor_score = min(all_scores, key=lambda x: x[1])
    poor_row = next(row for row in data if row["row"] == poor_row_num)

    # Find alternatives: rows with higher score, similar on goals
    best_alt = None
    best_alt_score = -float("inf")
    best_dist = None
    for rnum, score in all_scores:
        if rnum == poor_row_num:
            continue
        if score <= poor_score:
            continue
        candidate = next(row for row in data if row["row"] == rnum)
        dist = distance_on_goals(poor_row, candidate)
        # Accept if distance not too large; else still consider best
        if best_alt is None or dist < best_dist or (dist == best_dist and score > best_alt_score):
            best_alt = candidate
            best_alt_score = score
            best_dist = dist

    # Prepare changes dict
    if best_alt:
        changes = changes_on_inputs(poor_row, best_alt, input_cols)
    else:
        changes = {}

    # Print some human‑readable output
    print(f"Poorly rated row: {poor_row_num} (score={poor_score:.4f})")
    if best_alt:
        print(f"Alternative row: {best_alt['row']} (score={best_alt_score:.4f})")
        print(f"Changes in input columns: {changes}")
    else:
        print("No better alternative found.")
    # Machine‑checked answer
    alternatives = []
    if best_alt:
        alternatives.append({
            "row": best_alt["row"],
            "changes": changes
        })
    answer_json = json.dumps({
        "row": poor_row_num,
        "alternatives": alternatives
    }, separators=(",", ":"))
    print(f"ANSWER CAR-15 {answer_json}")

    # --------------------------------------------------------------
    # CAR-16: Vehicle not meeting preferred targets
    # --------------------------------------------------------------
    print("\n== CAR-16 ==")
    thresholds = median_thresholds(data)
    # Find a vehicle that does not meet all thresholds
    # Choose one with lowest score among such vehicles
    candidates = []
    for rnum, score in all_scores:
        row = next(row for row in data if row["row"] == rnum)
        if not all_goals_met(row, thresholds):
            candidates.append((score, row))
    if not candidates:
        print("All vehicles meet all median targets.")
        print("ANSWER CAR-16 {}")
        return
    _, poor_row = min(candidates, key=lambda x: x[0])  # lowest score
    poor_meets = goal_meets_count(poor_row, thresholds)

    # Find alternatives that meet more targets
    best_alt = None
    best_changes = None
    best_diff_count = None
    best_score = None
    for rnum, score in all_scores:
        if rnum == poor_row["row"]:
            continue
        row = next(row for row in data if row["row"] == rnum)
        meets = goal_meets_count(row, thresholds)
        if meets <= poor_meets:
            continue
        diff = diff_count_on_inputs(poor_row, row, input_cols)
        if best_alt is None or diff < best_diff_count or (diff == best_diff_count and score > best_score):
            best_alt = row
            best_changes = [col for col in input_cols if row["values"][col] != poor_row["values"][col]]
            best_diff_count = diff
            best_score = score

    if best_alt is None:
        print("No better alternative found.")
        print("ANSWER CAR-16 {}")
        return

    # Human readable output
    print(f"Vehicle not meeting all thresholds: row {poor_row['row']} (score={poor_row['row']})")
    print(f"Thresholds: {thresholds}")
    print(f"Alternative row: {best_alt['row']} (score={best_score:.4f})")
    print(f"Changed input columns: {best_changes}")

    # Machine‑checked answer
    answer_json = json.dumps({
        "row": poor_row["row"],
        "targets": thresholds,
        "alternative": best_alt["row"],
        "changed": best_changes
    }, separators=(",", ":"))
    print(f"ANSWER CAR-16 {answer_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

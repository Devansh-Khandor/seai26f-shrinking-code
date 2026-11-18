#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑17 and CAR‑18).

Usage:
    python skill_09.py <path_to_csv>
"""

import sys
import json
import numpy as np
from skill_01 import load_data, GOAL_COLUMNS

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def get_input_columns(data):
    """Return a sorted list of column names that are not goals."""
    all_keys = set(data[0]["values"].keys())
    return sorted(all_keys - set(GOAL_COLUMNS))

def compute_unusualness_scores(data, input_cols):
    """
    Compute an unusualness score for each row based on input columns.
    The score is the Euclidean distance from the mean in standardized
    (z‑score) space, computed only over available values.
    Returns a dict mapping row number to score (float).
    """
    # Compute mean and std for each input column
    means = {}
    stds  = {}
    for col in input_cols:
        vals = [row["values"][col] for row in data if row["values"][col] is not None]
        if not vals:
            means[col] = stds[col] = 0.0
        else:
            means[col] = float(np.mean(vals))
            stds[col]  = float(np.std(vals))

    scores = {}
    for row in data:
        row_num = row["row"]
        z_vals = []
        for col in input_cols:
            val = row["values"][col]
            if val is None:
                continue
            std = stds[col]
            z = (val - means[col]) / std if std != 0 else 0.0
            z_vals.append(z)
        # Euclidean distance in z‑space
        score = np.sqrt(np.sum(np.square(z_vals))) if z_vals else 0.0
        scores[row_num] = float(score)
    return scores

def euclidean_distance(row1, row2, cols):
    """
    Euclidean distance between two rows over the given columns,
    ignoring columns where either row has a missing value.
    Returns np.inf if there are no common columns.
    """
    sq = 0.0
    common = 0
    for col in cols:
        v1 = row1["values"][col]
        v2 = row2["values"][col]
        if v1 is None or v2 is None:
            continue
        diff = v1 - v2
        sq += diff * diff
        common += 1
    if common == 0:
        return np.inf
    return np.sqrt(sq)

def find_expected_by_neighbors(data, input_cols, target_col, k=5):
    """
    For each row, compute the expected value for target_col
    as the mean of that column among the k nearest neighbors
    in input space (using Euclidean distance over input_cols).
    Returns a dict mapping row number to expected value (float),
    excluding rows where no neighbor provides a target value.
    """
    expected = {}
    for idx, row in enumerate(data):
        # Find neighbors
        distances = []
        for j, other in enumerate(data):
            if idx == j:
                continue
            dist = euclidean_distance(row, other, input_cols)
            if np.isfinite(dist):
                distances.append((dist, j))
        distances.sort(key=lambda x: x[0])
        # Take up to k neighbors with finite distance
        neigh_vals = []
        for _, j in distances[:k]:
            val = data[j]["values"][target_col]
            if val is not None:
                neigh_vals.append(val)
        if neigh_vals:
            expected[row["row"]] = float(np.mean(neigh_vals))
    return expected

# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)
    input_cols = get_input_columns(data)

    # ----- CAR-17 --------------------------------------------------
    print("== CAR-17 ==")
    unusual_scores = compute_unusualness_scores(data, input_cols)
    # Sort by descending unusualness
    sorted_rows = sorted(unusual_scores.items(), key=lambda x: x[1], reverse=True)
    flagged_rows = [row_num for row_num, _ in sorted_rows[:40]]
    flagged_scores = {str(row): unusual_scores[row] for row in flagged_rows}

    # Print at least three lines of results
    print(f"Total rows: {len(data)}")
    print(f"Top 5 most unusual rows: {sorted_rows[:5]}")
    print(f"Number flagged: {len(flagged_rows)}")

    # Machine‑checked answer
    answer_json = json.dumps(
        {"flagged": flagged_rows, "scores": flagged_scores},
        separators=(",", ":")
    )
    print(f"ANSWER CAR-17 {answer_json}")

    # ----- CAR-18 --------------------------------------------------
    print("\n== CAR-18 ==")
    # Compute expected values for both columns
    expected_mpg = find_expected_by_neighbors(data, input_cols, "Mpg+", k=5)
    expected_lbs = find_expected_by_neighbors(data, input_cols, "Lbs-", k=5)

    # Compute absolute differences for rows where actual and expected exist
    diff_entries = []
    for row in data:
        row_num = row["row"]
        actual_mpg = row["values"]["Mpg+"]
        actual_lbs = row["values"]["Lbs-"]
        if actual_mpg is not None and row_num in expected_mpg:
            diff = abs(actual_mpg - expected_mpg[row_num])
            diff_entries.append((diff, row_num, "Mpg+", actual_mpg, expected_mpg[row_num]))
        if actual_lbs is not None and row_num in expected_lbs:
            diff = abs(actual_lbs - expected_lbs[row_num])
            diff_entries.append((diff, row_num, "Lbs-", actual_lbs, expected_lbs[row_num]))

    # Sort by descending difference
    diff_entries.sort(key=lambda x: x[0], reverse=True)
    flagged_entries = [
        {
            "row": int(row_num),
            "column": col,
            "actual": float(actual),
            "expected": float(expected),
        }
        for _, row_num, col, actual, expected in diff_entries[:40]
    ]

    # Print at least three lines of results
    print(f"Total rows considered: {len(data)}")
    print(f"Top 3 flagged entries: {flagged_entries[:3]}")
    print(f"Number flagged: {len(flagged_entries)}")

    # Machine‑checked answer
    answer_json = json.dumps(
        {"flagged": flagged_entries},
        separators=(",", ":")
    )
    print(f"ANSWER CAR-18 {answer_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

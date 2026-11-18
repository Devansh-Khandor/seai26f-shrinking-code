#!/usr/bin/env python3
"""
skill_09.py – CAR-17 and CAR-18

CAR‑17: Flag vehicle records whose combinations of characteristics are unusually different from the rest of our inventory.
CAR‑18: Identify vehicles whose reported fuel efficiency or weight differs substantially from what similar vehicles would lead us to expect.

The implementation uses only the standard library and re‑uses parsing helpers from previous skills.
"""

import csv
import json
import sys
import math

# ------------------------------------------------------------------
# Parsing helpers (copied from skill_01/skill_02)
# ------------------------------------------------------------------
def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ------------------------------------------------------------------
# Helper: Euclidean distance over given columns
# ------------------------------------------------------------------
def euclidean(row_a, row_b, cols):
    """Return Euclidean distance between two rows over the supplied columns.
    If a required value is missing in either row, return None.
    """
    sum_sq = 0.0
    for col in cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va is None or vb is None:
            return None
        diff = va - vb
        sum_sq += diff * diff
    return math.sqrt(sum_sq)

# ------------------------------------------------------------------
# CAR-17 implementation
# ------------------------------------------------------------------
def car_17(rows, input_cols):
    """Identify rows whose input‑space average distance to others is unusually large."""
    n = len(rows)
    # Pre‑compute distances for all pairs (symmetric)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(rows[i], rows[j], input_cols)
            if d is None:
                # Treat missing input as infinite distance; skip in average
                d = float('inf')
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    # Compute average distance for each row
    avg_dists = []
    for i in range(n):
        distances = [dist_matrix[i][j] for j in range(n) if j != i]
        # ignore infinities
        finite = [d for d in distances if d != float('inf')]
        if not finite:
            avg = 0.0
        else:
            avg = sum(finite) / len(finite)
        avg_dists.append(avg)

    # Determine threshold: mean + 2*std
    mean_avg = sum(avg_dists) / n
    std_avg = math.sqrt(sum((a - mean_avg) ** 2 for a in avg_dists) / n) if n > 1 else 0.0
    threshold = mean_avg + 2 * std_avg

    flagged = []
    scores = {}
    for idx, avg in enumerate(avg_dists, start=1):
        if avg > threshold:
            flagged.append(idx)
            scores[str(idx)] = round(avg, 3)

    return flagged, scores

# ------------------------------------------------------------------
# CAR-18 implementation
# ------------------------------------------------------------------
def car_18(rows, input_cols, goal_col, k=5):
    """Flag rows where a target value deviates substantially from the mean of k nearest neighbours."""
    n = len(rows)
    # Pre‑compute distances (as above but store only needed)
    dist_matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = euclidean(rows[i], rows[j], input_cols)
            if d is not None:
                dist_matrix[i][j] = d

    flagged_entries = []

    for idx in range(n):
        row = rows[idx]
        actual = row.get(goal_col)
        if actual is None:
            continue
        # Find k nearest neighbours that also have the target value and full inputs
        neighbors = []
        for j in range(n):
            if j == idx:
                continue
            if dist_matrix[idx][j] == float('inf'):
                continue
            if rows[j].get(goal_col) is None:
                continue
            neighbors.append((dist_matrix[idx][j], j))
        if len(neighbors) < k:
            continue
        neighbors.sort(key=lambda x: x[0])
        neighbor_indices = [j for _, j in neighbors[:k]]
        expected = sum(rows[j][goal_col] for j in neighbor_indices) / k
        diff = abs(actual - expected)
        flagged_entries.append((diff, idx + 1, goal_col, actual, expected))

    # Sort by descending difference, take up to 40
    flagged_entries.sort(key=lambda x: -x[0])
    top_entries = flagged_entries[:40]

    result = []
    for diff, row_num, col, actual, expected in top_entries:
        result.append({
            "row": row_num,
            "column": col,
            "actual": round(actual, 3) if isinstance(actual, float) else actual,
            "expected": round(expected, 3) if isinstance(expected, float) else expected
        })
    return result

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    headers = reader.fieldnames
    goal_cols = ['Lbs-', 'Acc+', 'Mpg+']

    # Parse numeric values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for k, v in r.items():
            parsed[k] = parse_value(v)
        parsed_rows.append(parsed)

    # Identify input columns: not goal, not ending with 'X'
    input_cols = [c for c in headers if c not in goal_cols and not c.endswith('X')]

    # CAR-17
    print("== CAR-17 ==")
    flagged_17, scores_17 = car_17(parsed_rows, input_cols)
    # Provide at least three lines of results
    print(f"Flagged rows (unusual input combinations): {flagged_17[:5]}")
    print(f"Unusualness scores (example): { {k: v for k, v in list(scores_17.items())[:3]} }")
    answer_17 = {"flagged": flagged_17, "scores": scores_17}
    print(f"ANSWER CAR-17 {json.dumps(answer_17, separators=(',', ':'))}")

    # CAR-18
    print("\n== CAR-18 ==")
    flagged_18_mpg = car_18(parsed_rows, input_cols, 'Mpg+')
    flagged_18_lbs = car_18(parsed_rows, input_cols, 'Lbs-')
    # Merge lists
    flagged_18 = flagged_18_mpg + flagged_18_lbs
    # Provide at least three lines of results
    print(f"Top flagged entries (examples): {flagged_18[:3]}")
    answer_18 = {"flagged": flagged_18}
    print(f"ANSWER CAR-18 {json.dumps(answer_18, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

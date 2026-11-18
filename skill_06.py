#!/usr/bin/env python3
"""
skill_06.py – CAR-11 and CAR-12

CAR-11: For the recommended vehicle, provide the three most similar vehicles
        in the inventory.  Similarity is measured by Euclidean distance
        over all numeric columns (excluding columns ending in 'X').
CAR-12: Compare two vehicles with different strengths and explain
        why the higher-scoring one is preferred, showing the trade‑offs
        in goal columns.
"""

import csv
import json
import math
import sys

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# Goal columns: maximize except those ending with '-'
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']
DIRECTION = {
    'Lbs-': 'min',
    'Acc+': 'max',
    'Mpg+': 'max',
}

# ------------------------------------------------------------------
# Performance scoring
# ------------------------------------------------------------------
def performance_score(row):
    """Return overall performance score if all goal columns present."""
    try:
        return -row['Lbs-'] + row['Acc+'] + row['Mpg+']
    except (TypeError, KeyError):
        return None

# ------------------------------------------------------------------
# Similarity (Euclidean distance over numeric columns)
# ------------------------------------------------------------------
def compute_distance(row_a, row_b):
    """Euclidean distance between two rows over columns that exist in both."""
    sum_sq = 0.0
    common = 0
    for col, va in row_a.items():
        if col.endswith('X'):   # ignore X columns
            continue
        vb = row_b.get(col)
        if vb is None:
            continue
        if va is None or vb is None:
            continue
        diff = va - vb
        sum_sq += diff * diff
        common += 1
    if common == 0:
        return float('inf')
    return math.sqrt(sum_sq)

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    # Parse numeric values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for k, v in r.items():
            parsed[k] = parse_value(v)
        parsed_rows.append(parsed)

    n_rows = len(parsed_rows)

    # Compute performance scores for rows with complete goal data
    scored_rows = []
    for idx, row in enumerate(parsed_rows, start=1):
        score = performance_score(row)
        if score is not None:
            scored_rows.append((idx, score, row))

    if not scored_rows:
        print("== CAR-11 ==")
        print("ANSWER CAR-11 {}")
        print("\n== CAR-12 ==")
        print("ANSWER CAR-12 {}")
        return

    # ---------- CAR-11 ----------
    # Recommended vehicle: highest score (tie: lowest row number)
    scored_rows.sort(key=lambda x: (-x[1], x[0]))
    recommended_row_num, recommended_score, recommended_row = scored_rows[0]

    # Compute distances to recommended row
    distances = []
    for idx, score, row in scored_rows[1:]:
        dist = compute_distance(recommended_row, row)
        distances.append((dist, idx, row))

    distances.sort(key=lambda x: x[0])
    neighbors = []
    for dist, idx, row in distances[:3]:
        # Gather values for goal columns
        values = {col: row[col] for col in GOAL_COLUMNS}
        neighbors.append({
            "row": idx,
            "values": values
        })

    print("== CAR-11 ==")
    print(f"Recommended row: {recommended_row_num} (score={recommended_score:.3f})")
    print(f"Neighbors: {neighbors}")
    answer_11 = {
        "recommended": recommended_row_num,
        "neighbors": neighbors
    }
    print(f"ANSWER CAR-11 {json.dumps(answer_11, separators=(',', ':'))}")

    # ---------- CAR-12 ----------
    # Take second best as comparison partner
    if len(scored_rows) >= 2:
        second_row_num, second_score, second_row = scored_rows[1]
    else:
        second_row_num, second_score, second_row = scored_rows[0]

    a_num, a_score, a_row = recommended_row_num, recommended_score, recommended_row
    b_num, b_score, b_row = second_row_num, second_score, second_row

    # Determine preferred
    preferred = a_num if a_score >= b_score else b_num

    # Differences per goal column
    differences = []
    tradeoffs = []
    for col in GOAL_COLUMNS:
        a_val = a_row[col]
        b_val = b_row[col]
        if a_val is None or b_val is None:
            continue
        diff_dict = {"feature": col, "a": a_val, "b": b_val}
        differences.append(diff_dict)
        # Identify tradeoff: column where non-preferred is better
        if preferred == a_num:
            # b is non-preferred
            if DIRECTION[col] == 'max' and b_val > a_val:
                tradeoffs.append(col)
            if DIRECTION[col] == 'min' and b_val < a_val:
                tradeoffs.append(col)
        else:
            # a is non-preferred
            if DIRECTION[col] == 'max' and a_val > b_val:
                tradeoffs.append(col)
            if DIRECTION[col] == 'min' and a_val < b_val:
                tradeoffs.append(col)

    print("\n== CAR-12 ==")
    print(f"Row a: {a_num}, Row b: {b_num}")
    print(f"Preferred: {preferred}")
    print(f"Scores: a={a_score:.3f}, b={b_score:.3f}")
    print(f"Differences: {differences}")
    print(f"Tradeoffs (non-preferred better): {tradeoffs}")

    answer_12 = {
        "a": a_num,
        "b": b_num,
        "preferred": preferred,
        "scores": {"a": round(a_score, 3), "b": round(b_score, 3)},
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER CAR-12 {json.dumps(answer_12, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_08.py – CAR-15 and CAR-16

CAR-15: For a poorly rated vehicle, show alternative vehicles in the historical
inventory with similar characteristics but better recorded performance, and
explain which differences could account for the improvement.

CAR-16: For a vehicle that does not meet our preferred performance targets,
identify the closest available alternative that meets more of those targets
while changing as few vehicle characteristics as possible.
"""

import csv
import json
import sys
import math

# ------------------------------------------------------------------
# Utility functions – copied from previous skills
# ------------------------------------------------------------------
def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']

def performance_score(row):
    """Return overall performance score if all goal columns present."""
    try:
        return -row['Lbs-'] + row['Acc+'] + row['Mpg+']
    except (TypeError, KeyError):
        return None

def compute_mean(values):
    """Return mean of a non‑empty numeric list."""
    return sum(values) / len(values)

# ------------------------------------------------------------------
# Helper to find input columns
# ------------------------------------------------------------------
def get_input_columns(all_columns):
    """Return the list of columns that are inputs (not goals, not ending in 'X')."""
    return [c for c in all_columns if c not in GOAL_COLUMNS and not c.endswith('X')]

# ------------------------------------------------------------------
# CAR-15 implementation
# ------------------------------------------------------------------
def car_15(rows):
    """
    Find the poorly rated vehicle (lowest performance score).
    Find alternative vehicles with similar input characteristics and better
    recorded performance. Return a JSON object with the poor row and a list of
    alternatives, each listing the input columns that differ.
    """
    # Compute performance scores for all rows that have complete goal data
    scored = []
    for idx, row in enumerate(rows, start=1):
        score = performance_score(row)
        if score is not None:
            scored.append((idx, score, row))

    if not scored:
        return {"row": None, "alternatives": []}

    # Poor row: lowest score (first if tie)
    scored.sort(key=lambda x: (x[1], x[0]))  # ascending score
    poor_idx, poor_score, poor_row = scored[0]

    # Identify alternatives
    input_cols = get_input_columns(all_columns)
    alternatives = []

    for idx, score, alt_row in scored[1:]:
        if score <= poor_score:
            continue  # need better performance

        # Compute Euclidean distance over input columns; skip if any missing
        diff_sum = 0.0
        valid = True
        for col in input_cols:
            v1 = poor_row.get(col)
            v2 = alt_row.get(col)
            if v1 is None or v2 is None:
                valid = False
                break
            diff = v1 - v2
            diff_sum += diff * diff
        if not valid:
            continue
        distance = math.sqrt(diff_sum)

        alternatives.append((distance, idx, alt_row))

    # Sort by similarity (distance)
    alternatives.sort(key=lambda x: x[0])

    # Keep top 3 alternatives
    top_alts = alternatives[:3]
    alt_list = []
    for _, alt_idx, alt_row in top_alts:
        changes = {}
        for col in input_cols:
            v1 = poor_row.get(col)
            v2 = alt_row.get(col)
            if v1 is None or v2 is None:
                continue
            if v1 != v2:
                changes[col] = [v1, v2]
        alt_list.append({"row": alt_idx, "changes": changes})

    return {"row": poor_idx, "alternatives": alt_list}

# ------------------------------------------------------------------
# CAR-16 implementation
# ------------------------------------------------------------------
def car_16(rows):
    """
    Identify a vehicle that does not meet all preferred performance targets
    (thresholds defined as the mean of each goal column). Find the closest
    alternative that satisfies more targets while changing as few input columns
    as possible. Return JSON with details.
    """
    # Compute mean thresholds for each goal
    goal_values = {col: [] for col in GOAL_COLUMNS}
    for row in rows:
        for col in GOAL_COLUMNS:
            v = row.get(col)
            if v is not None:
                goal_values[col].append(v)

    thresholds = {col: compute_mean(goal_values[col]) for col in GOAL_COLUMNS}

    # Identify first row that fails at least one target
    input_cols = get_input_columns(all_columns)
    def met_targets(row):
        met = 0
        if row['Lbs-'] is not None and row['Lbs-'] <= thresholds['Lbs-']:
            met += 1
        if row['Acc+'] is not None and row['Acc+'] >= thresholds['Acc+']:
            met += 1
        if row['Mpg+'] is not None and row['Mpg+'] >= thresholds['Mpg+']:
            met += 1
        return met

    target_row = None
    target_row_idx = None
    target_met = 0

    for idx, row in enumerate(rows, start=1):
        # Skip rows missing any goal value
        if any(row.get(col) is None for col in GOAL_COLUMNS):
            continue
        met = met_targets(row)
        if met < 3:
            target_row = row
            target_row_idx = idx
            target_met = met
            break

    if target_row is None:
        return {"row": None, "targets": thresholds, "alternative": None, "changed": []}

    # Find alternatives that meet more targets
    candidates = []
    for idx, row in enumerate(rows, start=1):
        if idx == target_row_idx:
            continue
        # Need all goal columns present
        if any(row.get(col) is None for col in GOAL_COLUMNS):
            continue
        met = met_targets(row)
        if met <= target_met:
            continue
        # Compute number of differing input columns
        diff_cnt = 0
        diff_list = []
        for col in input_cols:
            v1 = target_row.get(col)
            v2 = row.get(col)
            if v1 is None or v2 is None:
                continue
            if v1 != v2:
                diff_cnt += 1
                diff_list.append(col)
        candidates.append((diff_cnt, idx, diff_list, row))

    if not candidates:
        return {"row": target_row_idx, "targets": thresholds, "alternative": None, "changed": []}

    # Choose candidate with minimal diff count, tie by lowest row number
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_diff, best_idx, best_changed, best_row = candidates[0]

    return {"row": target_row_idx,
            "targets": thresholds,
            "alternative": best_idx,
            "changed": best_changed}

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    all_columns = reader.fieldnames  # preserve column order

    # Parse numeric values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for k, v in r.items():
            parsed[k] = parse_value(v)
        parsed_rows.append(parsed)

    # ---- CAR-15 ----
    print("== CAR-15 ==")
    result_15 = car_15(parsed_rows)
    # Provide at least three lines of output
    print(f"Poorly rated vehicle: Row {result_15['row']}")
    print("Similar alternatives with better performance:")
    for alt in result_15['alternatives']:
        changes = {k: [round(v[0],3) if isinstance(v[0], float) else v[0],
                      round(v[1],3) if isinstance(v[1], float) else v[1]}
                  for k, v in alt['changes'].items()}
        print(f"  Row {alt['row']} changes: {changes}")
    print(f"ANSWER CAR-15 {json.dumps(result_15, separators=(',', ':'))}")

    # ---- CAR-16 ----
    print("\n== CAR-16 ==")
    result_16 = car_16(parsed_rows)
    print(f"Vehicle not meeting all targets: Row {result_16['row']}")
    print(f"Target thresholds: { {k: round(v,3) for k,v in result_16['targets'].items()} }")
    if result_16['alternative'] is None:
        print("No better alternative found.")
    else:
        print(f"Recommended alternative: Row {result_16['alternative']}")
        print(f"Changed input columns: {result_16['changed']}")
    print(f"ANSWER CAR-16 {json.dumps(result_16, separators=(',', ':'))}")

#!/usr/bin/env python3
"""
skill_07.py – CAR-13 and CAR-14

CAR‑13: identify a vehicle with a strong balance of fuel efficiency,
weight and acceleration while evaluating as few candidate vehicles as
reasonably possible.
CAR‑14: when we have a limited inspection budget, recommend which
vehicles should be evaluated first so that we can discover promising
inventory options early.
"""

import csv
import json
import sys
import math

# ------------------------------------------------------------------
# Utility functions – reused from skill_01
# ------------------------------------------------------------------
def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# Goal columns and direction
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']
DIRECTION = {
    'Lbs-': 'min',   # minimize
    'Acc+': 'max',   # maximize
    'Mpg+': 'max',   # maximize
}

def performance_score(row):
    """Return overall performance score if all goal columns present."""
    try:
        return -row['Lbs-'] + row['Acc+'] + row['Mpg+']
    except (TypeError, KeyError):
        return None

# ------------------------------------------------------------------
# CAR-13 implementation
# ------------------------------------------------------------------
def car_13(rows):
    """
    Evaluate at most 30 rows in ascending order of row number.
    Return a dict with 'evaluated' (list of row numbers evaluated) and
    'best' (row number of the best balanced vehicle among evaluated).
    """
    n_rows = len(rows)
    budget = min(30, n_rows)

    evaluated = list(range(1, budget + 1))

    # Compute score for each evaluated row that has full goal data
    best_row = None
    best_score = None

    for idx in evaluated:
        r = rows[idx - 1]
        score = performance_score(r)
        if score is None:
            continue
        if best_score is None or score > best_score or (score == best_score and idx < best_row):
            best_score = score
            best_row = idx

    return {
        "evaluated": evaluated,
        "best": best_row
    }

# ------------------------------------------------------------------
# CAR-14 implementation
# ------------------------------------------------------------------
def car_14(rows):
    """
    Recommend an evaluation order for up to 30 vehicles so that promising
    inventory is discovered early.  We sort all rows that have complete
    goal data by decreasing performance score and pick the top ones.
    If fewer than 30 rows with complete goal data exist, we append
    the remaining rows with missing data in ascending row number.
    """
    n_rows = len(rows)
    budget = min(30, n_rows)

    # List of rows with full goal data
    scored = []
    for idx, r in enumerate(rows, start=1):
        score = performance_score(r)
        if score is not None:
            scored.append((score, idx))

    # Sort by score descending, then row number ascending
    scored.sort(key=lambda x: (-x[0], x[1]))
    order = [idx for _, idx in scored[:budget]]

    # If still not enough, fill with rows that have missing data
    if len(order) < budget:
        # Determine which rows are already used
        used = set(order)
        for idx in range(1, n_rows + 1):
            if idx not in used:
                order.append(idx)
                if len(order) == budget:
                    break

    return {
        "order": order
    }

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

    # CAR-13
    print("== CAR-13 ==")
    ans_13 = car_13(parsed_rows)
    print(f"ANSWER CAR-13 {json.dumps(ans_13, separators=(',', ':'))}")

    # CAR-14
    print("\n== CAR-14 ==")
    ans_14 = car_14(parsed_rows)
    print(f"ANSWER CAR-14 {json.dumps(ans_14, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

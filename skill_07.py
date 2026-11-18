#!/usr/bin/env python3
"""
Skill 07: Vehicle evaluation strategies

Usage:
    python skill_07.py <path to auto93.csv>
"""

import json
import sys

from skill_01 import read_csv
from skill_04 import GOAL_COLUMNS, compute_means_stds, compute_score

# ----------------------------------------------------------------------
# Helper functions

def composite_score(row, means, stds):
    """Compute the composite z‑score for a single row."""
    return compute_score(row, means, stds)

def best_row(rows, indices, means, stds):
    """Return the row number with the best composite score among given indices."""
    best_num = None
    best_val = None
    for idx in indices:
        row = rows[idx]
        # skip rows with missing goal values
        if any(row.get(col) is None for col in GOAL_COLUMNS):
            continue
        score = composite_score(row, means, stds)
        if best_val is None or score > best_val:
            best_val = score
            best_num = idx + 1  # 1‑based
    return best_num

# ----------------------------------------------------------------------
# CAR‑13 implementation

def car_13(rows, means, stds):
    """
    Evaluate as few rows as possible to find a vehicle with a strong balance
    of fuel efficiency, weight, and acceleration.
    """
    # Budget: up to 30 rows
    budget = min(30, len(rows))
    evaluated = list(range(budget))  # indices 0‑based
    best_num = best_row(rows, evaluated, means, stds)

    # Output
    print("== CAR-13 ==")
    for i, idx in enumerate(evaluated, start=1):
        if i > 3:  # at least three lines of results
            break
        row_no = idx + 1
        print(f"Evaluated Row {row_no}")
    print(f"Best row: {best_num}")
    print("ANSWER CAR-13 " + json.dumps({
        "evaluated": [i + 1 for i in evaluated],
        "best": best_num
    }))

# ----------------------------------------------------------------------
# CAR‑14 implementation

def car_14(rows, means, stds):
    """
    Recommend an evaluation order that discovers promising inventory early.
    We use the overall composite score to rank all rows and take the top 30.
    """
    scored = []
    for idx, row in enumerate(rows):
        if any(row.get(col) is None for col in GOAL_COLUMNS):
            continue
        score = composite_score(row, means, stds)
        scored.append((score, idx + 1))  # 1‑based row number

    # Sort descending by score
    scored.sort(key=lambda x: -x[0])

    order = [row_no for _, row_no in scored[:30]]  # up to 30 rows

    # Output
    print("\n== CAR-14 ==")
    for i, row_no in enumerate(order, start=1):
        if i > 3:  # at least three lines of results
            break
        print(f"Next to evaluate: Row {row_no}")
    print(f"Total order length: {len(order)}")
    print("ANSWER CAR-14 " + json.dumps({
        "order": order
    }))

# ----------------------------------------------------------------------
# Main entry point

def main(csv_path):
    rows = read_csv(csv_path)

    # Compute overall means and stds for goal columns
    n_rows = len(rows)
    means, stds = compute_means_stds(rows, list(range(n_rows)))

    # CAR‑13
    car_13(rows, means, stds)

    # CAR‑14
    car_14(rows, means, stds)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

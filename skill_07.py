#!/usr/bin/env python3
"""
Skill 07: Apache configuration discovery and budgeting

This script implements the following analyses:
- APACHE-13: Find a configuration with a low performance value while
  evaluating as few candidate configurations as possible.
- APACHE-14: Recommend an evaluation order for the first 30 rows
  so that promising server settings can be discovered early.

The implementation re‑uses the existing `ezr` library and
the helper functions from earlier skill modules.
"""

import sys
import json
import ezr

# --------------------------------------------------------------------------- #
# Helpers – reuse from previous skills
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load the CSV with ezr and return header and data rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data

def goal_indices(header):
    """Return list of indices of goal columns (+/-)."""
    return [i for i, n in enumerate(header) if n.endswith(('+', '-'))]

def setting_indices(header):
    """Return list of indices of setting columns (non‑goal, non‑X)."""
    return [i for i, n in enumerate(header)
            if not n.endswith(('+', '-')) and not n.endswith('X')]

def is_goal(name: str) -> bool:
    """Return True if column name ends with '+' or '-'."""
    return name.endswith('+') or name.endswith('-')

# --------------------------------------------------------------------------- #
# Core active‑learning style procedure
# --------------------------------------------------------------------------- #
def build_order(header, data, budget):
    """
    Construct an evaluation order of row numbers.
    The strategy:
      1. Start with the first `initial` rows that have a known goal.
      2. Iteratively train a tree‑based ranker on the evaluated rows
         and select the remaining row with the lowest predicted performance.
      3. Continue until the budget is reached or all rows have been ordered.
    Returns a list of row numbers (1‑based), in the order they should be
    evaluated.  Rows with a missing goal are still included in the order
    but are skipped when actually evaluated.
    """
    # Identify known‑goal rows
    goal_idx = goal_indices(header)[0]  # assume at least one goal
    known_rows = [i+1 for i, r in enumerate(data) if r[goal_idx] != "?"]

    # Initial set: first few known rows (deterministic)
    initial = min(5, len(known_rows))
    evaluated = known_rows[:initial]
    remaining = set(known_rows[initial:])  # known rows still not evaluated

    # All rows (for ordering), keep order of appearance
    all_rows = list(range(1, len(data)+1))

    # We will keep a set of rows already placed in the order
    ordered_set = set(evaluated)

    # Add remaining rows to the order (without evaluation yet)
    for r in all_rows:
        if r not in ordered_set:
            ordered_set.add(r)

    # Iterative selection based on predictions
    while len(evaluated) < budget and remaining:
        # Build ranker on evaluated rows
        evaluated_rows = [data[i-1] for i in evaluated]
        tbl = ezr.Tbl([header] + evaluated_rows)
        rank = ezr.ranker(tbl)

        # Predict for all remaining known rows
        best_candidate = None
        best_score = None
        for r in sorted(remaining):
            row = data[r-1]
            score = rank(row)
            if best_candidate is None or score < best_score or (score == best_score and r < best_candidate):
                best_candidate = r
                best_score = score

        # Evaluate that candidate
        evaluated.append(best_candidate)
        remaining.remove(best_candidate)

    return evaluated

def evaluate_best(header, data, evaluated):
    """Return the evaluated row number with the best (lowest) true goal value."""
    goal_idx = goal_indices(header)[0]
    best_row = None
    best_val = None
    for r in evaluated:
        val = data[r-1][goal_idx]
        if val == "?":
            continue
        if best_val is None or val < best_val or (val == best_val and r < best_row):
            best_row = r
            best_val = val
    return best_row

# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #
def apache_13(header, data):
    """APACHE-13: Discover a low‑performance configuration with few evaluations."""
    # Budget: at most min(30, number_of_rows // 5)
    budget = min(30, len(data) // 5)

    evaluated = build_order(header, data, budget)
    best = evaluate_best(header, data, evaluated)

    print("== APACHE-13 ==")
    # Show results – at least three lines
    for r in evaluated:
        val = data[r-1][goal_indices(header)[0]]
        print(f"Row {r}: goal={val}")
    print(f"Best row: {best}")

    answer = {"evaluated": evaluated, "best": best}
    print(f"ANSWER APACHE-13 {json.dumps(answer, separators=(',', ':'))}")

def apache_14(header, data):
    """APACHE-14: Recommend evaluation order for the first 30 rows."""
    # Budget: at most min(30, number_of_rows // 5)
    budget = min(30, len(data) // 5)

    order = build_order(header, data, budget)

    print("== APACHE-14 ==")
    # Show first few rows of the order (at least three lines)
    for r in order[:5]:
        print(f"Order: row {r}")
    print(f"Full order length: {len(order)}")

    answer = {"order": order}
    print(f"ANSWER APACHE-14 {json.dumps(answer, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_13(header, data)
    apache_14(header, data)

if __name__ == "__main__":
    main()

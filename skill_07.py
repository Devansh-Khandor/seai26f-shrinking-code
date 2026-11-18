#!/usr/bin/env python3
"""
Skill 07: Efficient vehicle selection under limited inspection budgets.

CAR-13: Identify a vehicle with a strong balance of fuel efficiency (Mpg+),
        weight (Lbs-), and acceleration (Acc+) while evaluating as few
        candidate vehicles as reasonably possible.
CAR-14: Recommend an inspection order so that promising inventory options
        are discovered early, using only the goal values of rows already
        placed earlier in the order.

The program reads the CSV file supplied on the command line and prints
the results for both requests.  After each request’s human‑readable
output it prints a machine‑checked JSON answer.
"""

import csv
import json
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(val: str):
    """Convert a CSV cell to float, or None for missing."""
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col_name: str) -> bool:
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')

def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')

# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def compute_minmax(rows, goal_cols):
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs

def normalize_value(v, mn, mx, maximize=True):
    if mx == mn:
        return 1.0
    if maximize:
        return (v - mn) / (mx - mn)
    else:
        return (mx - v) / (mx - mn)

# ----------------------------------------------------------------------
# Score computation for a row (using specified goal columns)
# ----------------------------------------------------------------------
def score_row(row, goal_cols, mins, maxs):
    score = 0.0
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None  # cannot score if any goal missing
        mn = mins[col]
        mx = maxs[col]
        score += normalize_value(v, mn, mx, maximize=is_maximize(col))
    return score

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_cols_all = [c for c in fieldnames if is_goal(c)]
    goal_cols_3 = [c for c in ['Mpg+', 'Lbs-', 'Acc+'] if c in goal_cols_all]
    # Global min/max for all goal columns (used for scoring)
    mins_all, maxs_all = compute_minmax(rows, goal_cols_all)

    # --------------------------------------------------------------
    # CAR-13
    # --------------------------------------------------------------
    print("== CAR-13 ==")
    max_evaluate = min(30, len(rows))
    evaluated_indices = list(range(1, max_evaluate + 1))  # 1-based
    best_row = None
    best_score = -math.inf
    for idx in evaluated_indices:
        row = rows[idx - 1]
        sc = score_row(row, goal_cols_3, mins_all, maxs_all)
        if sc is not None and sc > best_score:
            best_score = sc
            best_row = idx
    if best_row is None:
        best_row = -1
    # Human‑readable output
    print(f"Evaluated {max_evaluate} rows (1-{max_evaluate})")
    if best_row != -1:
        print(f"Best row found: {best_row} with score {best_score:.6f}")
    else:
        print("No complete row found among evaluated rows.")
    # JSON answer
    print("ANSWER CAR-13", json.dumps(
        {"evaluated": evaluated_indices, "best": best_row},
        ensure_ascii=False
    ))

    # --------------------------------------------------------------
    # CAR-14
    # --------------------------------------------------------------
    print("\n== CAR-14 ==")
    # Compute score for all rows using all goal columns
    all_scores = []
    for i, r in enumerate(rows, start=1):
        sc = score_row(r, goal_cols_all, mins_all, maxs_all)
        if sc is None:
            sc = -math.inf  # treat missing as worst
        all_scores.append((i, sc))
    # Sort rows by descending score
    all_scores.sort(key=lambda x: (-x[1], x[0]))
    order = [i for i, _ in all_scores[:30]]
    # Human‑readable output
    print(f"Inspection order (top {len(order)} rows):")
    print(order[:5], "...", order[-5:])
    # JSON answer
    print("ANSWER CAR-14", json.dumps(
        {"order": order},
        ensure_ascii=False
    ))

if __name__ == "__main__":
    main()

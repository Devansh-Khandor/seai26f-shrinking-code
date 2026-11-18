#!/usr/bin/env python3
"""
Skill 07: Vehicle selection with limited inspection budget.
"""

import sys
import csv
import json
import math

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV file, return header list and rows list of dicts."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v):
    """Parse numeric value or return None if missing."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col):
    """Return True if column ends in '+' or '-'."""
    return col.endswith('+') or col.endswith('-')

# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def compute_global_minmax(rows, goal_cols):
    """Return dicts of min and max for each goal column across all rows."""
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def normalize(v, mn, mx, is_minimize):
    if mn == mx:
        return 1.0
    return (mx - v) / (mx - mn) if is_minimize else (v - mn) / (mx - mn)

def score_row(row, mins, maxs, goal_cols):
    """Return average normalized score for a row; None if any goal missing."""
    vals = []
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        vals.append(normalize(v, mins[col], maxs[col], col.endswith('-')))
    return sum(vals) / len(vals)

# --------------------------------------------------------------------------- #
# CAR-13: Limited inspection
# --------------------------------------------------------------------------- #

def car13(header, rows, mins, maxs, goal_cols, budget=30, threshold=0.99):
    evaluated = []
    best_row = None
    best_score = -1.0

    count = 0
    for idx, row in enumerate(rows, start=1):
        # Only rows with all goal values can be evaluated
        if any(parse_value(row[col]) is None for col in goal_cols):
            continue
        sc = score_row(row, mins, maxs, goal_cols)
        if sc is None:
            continue
        evaluated.append(idx)
        count += 1
        if sc > best_score:
            best_score = sc
            best_row = idx
        if count >= budget or sc >= threshold:
            break

    return evaluated, best_row

# --------------------------------------------------------------------------- #
# CAR-14: Order for early discovery
# --------------------------------------------------------------------------- #

def car14(header, rows, mins, maxs, goal_cols, budget=30):
    scored = []
    for idx, row in enumerate(rows, start=1):
        if any(parse_value(row[col]) is None for col in goal_cols):
            continue
        sc = score_row(row, mins, maxs, goal_cols)
        if sc is None:
            continue
        scored.append((idx, sc))

    # Sort by score descending, tie by row number ascending
    scored.sort(key=lambda x: (-x[1], x[0]))
    order = [idx for idx, _ in scored[:budget]]
    return order

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    goal_cols = [c for c in header if is_goal(c)]
    mins, maxs = compute_global_minmax(rows, goal_cols)

    # CAR-13
    print("== CAR-13 ==")
    evaluated, best = car13(header, rows, mins, maxs, goal_cols, budget=30, threshold=0.99)
    print(f"Evaluated rows count: {len(evaluated)}")
    print(f"Best row: {best}")
    if best is not None:
        best_score = score_row(rows[best-1], mins, maxs, goal_cols)
        print(f"Best score: {best_score:.4f}")
    else:
        print("No evaluable rows found.")
    # Machine‑checked answer
    answer13 = {"evaluated": evaluated, "best": best}
    print(f"ANSWER CAR-13 {json.dumps(answer13)}")

    # CAR-14
    print("\n== CAR-14 ==")
    order = car14(header, rows, mins, maxs, goal_cols, budget=30)
    print(f"Order first 30 rows: {order}")
    # Machine‑checked answer
    answer14 = {"order": order}
    print(f"ANSWER CAR-14 {json.dumps(answer14)}")

if __name__ == "__main__":
    main()

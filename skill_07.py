#!/usr/bin/env python3
"""
Skill 07: Vehicle selection and inspection ordering (CAR‑13 & CAR‑14).

The script follows the specification for the two CAR tasks:
  * CAR‑13 – Identify a vehicle with a strong balance of fuel‑efficiency,
    weight and acceleration by evaluating as few candidate vehicles as
    reasonably possible (up to the evaluation budget).
  * CAR‑14 – Determine an order in which to evaluate vehicles that
    maximises the chance of finding promising options early.

The implementation uses only the goal values (columns ending in '+' or '-')
and no external libraries beyond the Python standard library.

The script is run as:
    python skill_07.py <csv_path>
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read the CSV and return a list of (row_index, row_dict) tuples.
    Missing values are represented by None.  Row indices are 1‑based.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((idx, parsed))
    return rows

# ----------------------------------------------------------------------
# Goal utilities
# ----------------------------------------------------------------------
GOAL_COLS = ["Lbs-", "Acc+", "Mpg+"]

def compute_global_mins_maxs(rows):
    """
    Return two dicts mapping each goal column to its global minimum and maximum
    values over all rows that have a non‑missing value for that column.
    """
    mins = {}
    maxs = {}
    for col in GOAL_COLS:
        vals = [rec[col] for _, rec in rows if rec[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def composite_score(rec, mins, maxs):
    """
    Compute the composite score for a single record.
    If any goal value is missing, return None.
    """
    if any(rec[col] is None for col in GOAL_COLS):
        return None
    score = 0.0
    for col in GOAL_COLS:
        denom = maxs[col] - mins[col]
        if denom == 0:
            continue
        if col.endswith('-'):  # minimize
            score += (maxs[col] - rec[col]) / denom
        else:                 # maximize
            score += (rec[col] - mins[col]) / denom
    return score

# ----------------------------------------------------------------------
# CAR‑13: Balanced vehicle selection
# ----------------------------------------------------------------------
def car13(rows, mins, maxs, budget=30):
    """
    Evaluate rows one by one in the natural order until the budget is reached.
    Return the list of evaluated row numbers and the row with the highest
    composite score among those evaluated.
    """
    evaluated = []
    best_row = None
    best_score = -math.inf

    for idx, rec in rows:
        if len(evaluated) >= budget:
            break
        evaluated.append(idx)
        score = composite_score(rec, mins, maxs)
        if score is not None and score > best_score:
            best_score = score
            best_row = idx

    return {"evaluated": evaluated, "best": best_row}

# ----------------------------------------------------------------------
# CAR‑14: Inspection ordering
# ----------------------------------------------------------------------
def car14(rows, mins, maxs, budget=30):
    """
    Determine an order to inspect vehicles by sorting all rows by their
    composite score (descending).  The order includes the top `budget`
    rows.  This ordering uses only goal values and is deterministic.
    """
    scored = []
    for idx, rec in rows:
        score = composite_score(rec, mins, maxs)
        if score is not None:
            scored.append((score, idx))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    order = [idx for _, idx in scored[:budget]]
    return {"order": order}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    mins, maxs = compute_global_mins_maxs(rows)

    # CAR‑13
    print("== CAR-13 ==")
    res13 = car13(rows, mins, maxs)
    print(f"Evaluated rows (up to budget): {res13['evaluated'][:5]} ...")
    print(f"Best evaluated row: {res13['best']}")
    print("ANSWER CAR-13 " + json.dumps(res13, separators=(',', ':')))

    # CAR‑14
    print("\n== CAR-14 ==")
    res14 = car14(rows, mins, maxs)
    print(f"Top {len(res14['order'])} rows to evaluate first: {res14['order'][:5]} ...")
    print("ANSWER CAR-14 " + json.dumps(res14, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_07.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])

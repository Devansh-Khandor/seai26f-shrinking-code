#!/usr/bin/env python3
"""
skill_07.py - Apache configuration discovery and ordering

The script implements two tasks:

  * APACHE-13 : Discover a low‑performance configuration while evaluating as few
                configurations as possible (budgeted to min(30, n//5)).
  * APACHE-14 : Produce an evaluation order for the first 30 configurations so that
                the earliest evaluated rows are likely to contain promising settings.
"""

import csv
import json
import math
import sys
from typing import List, Dict, Tuple, Set

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Read the CSV file and return a list of dictionaries (row number 1-based)
    and the header list. Missing values are kept as the string '?'.
    """
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            row = {}
            for key, val in zip(header, line):
                val = val.strip()
                if val == "":
                    val = "?"
                row[key] = val
            rows.append(row)
    return rows, header

# ----------------------------------------------------------------------
# Column helpers
# ----------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")

def is_input(col: str, goal_col: str) -> bool:
    return not col.endswith("X") and col != goal_col

# ----------------------------------------------------------------------
# Distance in input space
# ----------------------------------------------------------------------
def distance(r1: Dict[str, str], r2: Dict[str, str], inputs: List[str]) -> int:
    """
    Number of differing input settings, treating missing values as unknown
    and not counting them as mismatches.
    """
    d = 0
    for c in inputs:
        v1, v2 = r1[c], r2[c]
        if v1 == "?" or v2 == "?":
            continue
        if v1 != v2:
            d += 1
    return d

# ----------------------------------------------------------------------
# Active‑learning selection
# ----------------------------------------------------------------------
def select_next(
    evaluated: Set[int],
    rows: List[Dict[str, str]],
    goal_col: str,
    inputs: List[str],
) -> int:
    """
    Choose the next row index (0‑based) to evaluate.
    The estimate for each candidate is the goal value of the nearest
    evaluated row (in input distance).  The candidate with the
    lowest estimated goal is returned.  Ties are broken by lowest index.
    """
    best_idx = None
    best_est = math.inf

    for i, row in enumerate(rows):
        if i in evaluated:
            continue
        # Find nearest evaluated row
        min_dist = math.inf
        min_goal = math.inf
        for ev in evaluated:
            d = distance(row, rows[ev], inputs)
            if d < min_dist:
                min_dist = d
                # goal value may be missing; treat missing as very large
                gv = rows[ev][goal_col]
                min_goal = float(gv) if gv != "?" else math.inf
            elif d == min_dist:
                gv = rows[ev][goal_col]
                gv_val = float(gv) if gv != "?" else math.inf
                if gv_val < min_goal:
                    min_goal = gv_val
        if min_goal < best_est or (min_goal == best_est and (best_idx is None or i < best_idx)):
            best_est = min_goal
            best_idx = i
    return best_idx

# ----------------------------------------------------------------------
# Main logic for APACHE‑13 and APACHE‑14
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, header = read_csv(path)

    # Identify goal and input columns
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]
    inputs = [c for c in header if is_input(c, goal_col)]

    n = len(rows)
    budget = min(30, n // 5)

    # APACHE‑13: sequential discovery
    evaluated13: List[int] = []   # 1‑based indices
    evaluated_set13: Set[int] = set()

    # Start by evaluating first row that has a goal value (or any if all missing)
    # Find first row with non‑? goal
    for i in range(n):
        if rows[i][goal_col] != "?":
            start = i
            break
    else:
        start = 0  # all missing, just pick 0

    evaluated13.append(start + 1)
    evaluated_set13.add(start)

    while len(evaluated13) < budget:
        nxt = select_next(evaluated_set13, rows, goal_col, inputs)
        evaluated13.append(nxt + 1)
        evaluated_set13.add(nxt)

    # Determine best row among those evaluated
    best_val = math.inf
    best_idx = None
    for idx in evaluated13:
        gv = rows[idx - 1][goal_col]
        gv_val = float(gv) if gv != "?" else math.inf
        if gv_val < best_val:
            best_val = gv_val
            best_idx = idx

    # Output APACHE‑13
    print("== APACHE-13 ==")
    print(f"Evaluated rows: {evaluated13}")
    print(f"Best row: {best_idx} (value {best_val})")
    print(f"Number evaluated: {len(evaluated13)}")

    answer13 = {"evaluated": evaluated13, "best": best_idx}
    print("ANSWER APACHE-13", json.dumps(answer13, separators=(",", ":")))

    # APACHE‑14: produce an order for the first 30 rows
    evaluated14: List[int] = []
    evaluated_set14: Set[int] = set()

    # Again start with same first row as before
    evaluated14.append(start + 1)
    evaluated_set14.add(start)

    while len(evaluated14) < budget:
        nxt = select_next(evaluated_set14, rows, goal_col, inputs)
        evaluated14.append(nxt + 1)
        evaluated_set14.add(nxt)

    print("\n== APACHE-14 ==")
    print(f"Evaluation order: {evaluated14}")
    print(f"Number in order: {len(evaluated14)}")

    answer14 = {"order": evaluated14}
    print("ANSWER APACHE-14", json.dumps(answer14, separators=(",", ":")))

if __name__ == "__main__":
    main()

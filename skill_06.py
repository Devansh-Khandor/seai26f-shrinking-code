#!/usr/bin/env python3
"""
skill_06.py

Entry point for APACHE-11 and APACHE-12 requests.

Usage:
    python skill_06.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Shared utilities (same as in previous skills)

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_csv(path: str):
    """Load CSV, return list of (row_num, dict_of_columns)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=1):  # row numbers start at 1
            row = {}
            for k, v in raw.items():
                row[k] = parse_cell(v)
            rows.append((i, row))
    return rows


def identify_columns(rows):
    """Return goal_cols, input_cols as lists."""
    col_keys = rows[0][1].keys()
    goal_cols = [col for col in col_keys if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in col_keys if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]
    return goal_cols, input_cols


# --------------------------------------------------------------------------- #
# Helper functions

def goal_better(a, b, goal_col):
    """Return True if a is better than b for this goal."""
    if a is None or b is None:
        return False  # cannot compare
    if goal_col.endswith('+'):
        return a > b
    else:  # goal_col endswith '-'
        return a < b


def row_score(row, goal_cols):
    """
    Compute a simple score for a row: sum over goal columns
    where higher is better for '+', lower is better for '-'.
    """
    score = 0.0
    for g in goal_cols:
        val = row.get(g)
        if val is None:
            continue
        if g.endswith('+'):
            score += val
        else:  # '-'
            score -= val
    return score


def hamming_distance(row1, row2, cols):
    """Count mismatched values over specified columns."""
    dist = 0
    for c in cols:
        v1 = row1.get(c)
        v2 = row2.get(c)
        if v1 != v2:
            dist += 1
    return dist


# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)

    # ------------------------------------------------------------------ APACHE-11
    print("== APACHE-11 ==")

    # Find best row (minimize Performance-; generally lower better for '-' goals)
    # We'll choose the row with smallest value for any '-' goal, but primary is Performance-
    target_goal = None
    for g in goal_cols:
        if g == 'Performance-':
            target_goal = g
            break
    if target_goal is None:
        target_goal = goal_cols[0]

    best_row_num, best_row = min(rows, key=lambda r: (r[1].get(target_goal) is None, r[1].get(target_goal)))
    print(f"Recommended row: {best_row_num}")

    # Find 3 closest rows by Hamming distance over input columns
    neighbors = []
    for rnum, row in rows:
        if rnum == best_row_num:
            continue
        dist = hamming_distance(best_row, row, input_cols)
        neighbors.append((dist, rnum, row))
    neighbors.sort(key=lambda x: x[0])
    selected = neighbors[:3]

    print("Nearest neighbors (by Hamming distance on input cols):")
    for dist, rnum, row in selected:
        print(f"Row {rnum} (dist {dist}):")
        # Show goal column values
        for g in goal_cols:
            print(f"  {g}: {row[g]}")

    # Prepare JSON answer
    neighbors_json = []
    for _, rnum, row in selected:
        val_dict = {g: row[g] for g in goal_cols}
        neighbors_json.append({"row": rnum, "values": val_dict})

    answer_11 = {
        "recommended": best_row_num,
        "neighbors": neighbors_json
    }
    print("\nANSWER APACHE-11", json.dumps(answer_11))

    # ------------------------------------------------------------------ APACHE-12
    print("\n== APACHE-12 ==")

    # Identify worst row (maximize Performance-; highest is worse for '-' goal)
    worst_row_num, worst_row = max(rows, key=lambda r: (r[1].get(target_goal) is None, r[1].get(target_goal)))
    print(f"Best row: {best_row_num}")
    print(f"Worst row: {worst_row_num}")

    # Determine preferred (better) row
    preferred = best_row_num
    other = worst_row_num
    # compute scores
    score_a = row_score(best_row, goal_cols)
    score_b = row_score(worst_row, goal_cols)
    scores = {"a": score_a, "b": score_b}

    # Differences between features
    differences = []
    for col in best_row.keys():
        va = best_row.get(col)
        vb = worst_row.get(col)
        if va != vb:
            differences.append({"feature": col, "a": va, "b": vb})
    if not differences:
        # ensure at least one difference
        differences.append({"feature": "unknown", "a": None, "b": None})

    # Tradeoffs: goal columns where non-preferred row is better
    tradeoffs = []
    for g in goal_cols:
        if not goal_better(best_row.get(g), worst_row.get(g), g):
            if goal_better(worst_row.get(g), best_row.get(g), g):
                tradeoffs.append(g)

    answer_12 = {
        "a": best_row_num,
        "b": worst_row_num,
        "preferred": "a",
        "scores": scores,
        "differences": differences,
        "tradeoffs": tradeoffs
    }

    print("\nANSWER APACHE-12", json.dumps(answer_12, separators=(',', ':')))


if __name__ == "__main__":
    main()

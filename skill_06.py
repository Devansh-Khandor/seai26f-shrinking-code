#!/usr/bin/env python3
"""
Skill 06: Recommendations and trade‑off explanations for the auto93 dataset.

- CAR-11: Recommend the best vehicle and provide its most similar neighbours.
- CAR-12: Compare the top and bottom vehicles, explain why the top one is
          preferred, and list trade‑offs where the bottom one actually performs
          better on individual goal metrics.

The program reads a CSV file passed on the command line and outputs the
requested information in human‑readable form followed by a machine‑checked JSON
answer for each task.
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
    """Convert a CSV cell into float or None."""
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col_name: str) -> bool:
    """Return True for columns that end with '+' or '-' and not end with 'X'."""
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
# Normalized statistics
# ----------------------------------------------------------------------
def compute_minmax(rows, goal_cols):
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs

def normalize_row(row, mins, maxs, goal_cols):
    norm = {}
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm[col] = 1.0
        else:
            if is_maximize(col):
                norm[col] = (v - mn) / (mx - mn)
            else:
                norm[col] = (mx - v) / (mx - mn)
    return norm

def score_from_norm(norm_dict):
    return sum(norm_dict.values())

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_cols = [c for c in fieldnames if is_goal(c)]
    non_goal_cols = [c for c in fieldnames if not is_goal(c)]

    # Compute min/max per goal
    mins, maxs = compute_minmax(rows, goal_cols)

    # Build list of valid rows with all goal values
    valid = []  # list of dicts: idx, raw, norm, score
    for idx, row in enumerate(rows, start=1):
        norm = normalize_row(row, mins, maxs, goal_cols)
        if norm is None:
            continue
        raw = {col: parse_value(row[col]) for col in goal_cols}
        sc = score_from_norm(norm)
        valid.append({
            'idx': idx,
            'raw': raw,
            'norm': norm,
            'score': sc
        })

    if not valid:
        print("No complete rows with all goal values.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-11
    # ------------------------------------------------------------------
    print("== CAR-11 ==")

    # Best row (max score)
    best = max(valid, key=lambda v: v['score'])
    recommended_row = best['idx']
    recommended_score = best['score']

    print(f"Recommended row: {recommended_row} (score {recommended_score:.6f})")

    # Find nearest neighbours in normalized goal space
    def distance(a, b):
        return math.sqrt(sum((a[c] - b[c]) ** 2 for c in goal_cols))

    neighbours = sorted(
        ((v, distance(best['norm'], v['norm'])) for v in valid if v['idx'] != best['idx']),
        key=lambda x: x[1]
    )[:3]

    print("Nearest neighbours:")
    neighbor_list = []
    for neighbour, dist in neighbours:
        row_no = neighbour['idx']
        values = neighbour['raw']  # dict of goal cols
        neighbor_list.append({
            'row': row_no,
            'values': values
        })
        print(f"Row {row_no}: {values} (dist {dist:.6f})")

    # JSON answer
    answer11 = {
        "recommended": recommended_row,
        "neighbors": neighbor_list
    }
    print("ANSWER CAR-11", json.dumps(answer11, ensure_ascii=False))

    # ------------------------------------------------------------------
    # CAR-12
    # ------------------------------------------------------------------
    print("\n== CAR-12 ==")

    # Best and worst
    a = best  # preferred
    b = min(valid, key=lambda v: v['score'])
    preferred = a['idx']
    a_score = a['score']
    b_score = b['score']

    print(f"Row a (preferred): {a['idx']} (score {a_score:.6f})")
    print(f"Row b: {b['idx']} (score {b_score:.6f})")
    print(f"Preferred: {preferred}")

    # Differences in goal columns
    differences = []
    for col in goal_cols:
        diff_entry = {
            "feature": col,
            "a": a['raw'][col],
            "b": b['raw'][col]
        }
        differences.append(diff_entry)

    # Trade‑offs where non‑preferred (b) is better
    tradeoffs = []
    for col in goal_cols:
        a_val = a['raw'][col]
        b_val = b['raw'][col]
        if a_val is None or b_val is None:
            continue
        if is_maximize(col):
            if b_val > a_val:
                tradeoffs.append(col)
        else:
            if b_val < a_val:
                tradeoffs.append(col)

    print("Differences (goal columns):")
    for d in differences[:5]:
        print(f"  {d['feature']}: a={d['a']}, b={d['b']}")

    print("Trade‑offs where the non‑preferred row is better:")
    for col in tradeoffs[:5]:
        print(f"  {col}")

    # JSON answer
    answer12 = {
        "a": a['idx'],
        "b": b['idx'],
        "preferred": preferred,
        "scores": {"a": a_score, "b": b_score},
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print("ANSWER CAR-12", json.dumps(answer12, ensure_ascii=False))

if __name__ == "__main__":
    main()

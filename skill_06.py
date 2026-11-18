#!/usr/bin/env python3
"""
skill_06.py: Explain vehicle recommendations (CAR-11) and compare strengths (CAR-12).
"""

import csv
import json
import sys
from collections import defaultdict, Counter
from statistics import mean as _mean
from math import isfinite

# --------------------------------------------------------------------------- #
# Basic utilities
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Convert a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s

def read_csv(path: str):
    """Return header list and rows list (1‑based index)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, line in enumerate(reader, 1):
            rows.append((i, [parse_cell(v) for v in line]))
    return header, rows

# --------------------------------------------------------------------------- #
# Column metadata
# --------------------------------------------------------------------------- #

def is_goal(col: str):
    return col.endswith(("+", "-"))

def is_input(col: str):
    return not col.endswith("X") and not is_goal(col)

def column_types(header, rows):
    """Return dict of column -> 'num' or 'sym'."""
    col_types = {}
    for idx, name in enumerate(header):
        values = [row[idx] for _, row in rows if row[idx] is not None]
        if not values:
            # default to numeric to avoid divide-by-zero later
            col_types[name] = "num"
            continue
        if all(isinstance(v, (int, float)) for v in values):
            col_types[name] = "num"
        else:
            col_types[name] = "sym"
    return col_types

def numeric_bounds(header, rows):
    """Return dict of column -> (min, max). Only numeric columns."""
    bounds = {}
    for idx, name in enumerate(header):
        vals = [row[idx] for _, row in rows if isinstance(row[idx], (int, float))]
        if vals:
            bounds[name] = (min(vals), max(vals))
    return bounds

# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def compute_score(row, goal_names, bounds, col_types):
    """Return composite score 0-1 for a row with complete goal values."""
    scores = []
    for goal in goal_names:
        val = row[header.index(goal)]
        if val is None:
            return None
        lo, hi = bounds[goal]
        if hi == lo:
            norm = 0.5
        else:
            if goal.endswith("-"):          # minimize
                norm = (hi - val) / (hi - lo)
            else:                            # maximize
                norm = (val - lo) / (hi - lo)
        scores.append(norm)
    return sum(scores) / len(scores)

# --------------------------------------------------------------------------- #
# Distance between rows (for similarity)
# --------------------------------------------------------------------------- #

def row_distance(r1, r2, header, bounds, col_types):
    """Euclidean-like distance over all columns, ignoring missing values."""
    total = 0.0
    for idx, name in enumerate(header):
        v1, v2 = r1[idx], r2[idx]
        if v1 is None or v2 is None:
            continue
        if col_types[name] == "num":
            lo, hi = bounds.get(name, (0, 1))
            rng = hi - lo
            if rng == 0:
                diff = 0.0
            else:
                diff = abs(v1 - v2) / rng
        else:
            diff = 0.0 if v1 == v2 else 1.0
        total += diff
    return total

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(path):
    global header
    header, rows = read_csv(path)
    goal_names = [c for c in header if is_goal(c)]
    if not goal_names:
        print("== CAR-11 ==")
        print("No goal columns found.")
        print("ANSWER CAR-11 {}")
        print("\n== CAR-12 ==")
        print("No goal columns found.")
        print("ANSWER CAR-12 {}")
        return

    # Prepare metadata
    col_types = column_types(header, rows)
    bounds = numeric_bounds(header, rows)

    # Evaluation budget
    budget = min(30, len(rows) // 5)

    # Evaluate rows
    evaluated = []
    for row_num, row in rows:
        if len(evaluated) >= budget:
            break
        # Row must have all goals present
        if any(row[header.index(g)] is None for g in goal_names):
            continue
        score = compute_score(row, goal_names, bounds, col_types)
        if score is None:
            continue
        evaluated.append((row_num, row, score))

    if not evaluated:
        print("== CAR-11 ==")
        print("No rows available for evaluation.")
        print("ANSWER CAR-11 {}")
        print("\n== CAR-12 ==")
        print("No rows available for comparison.")
        print("ANSWER CAR-12 {}")
        return

    # CAR-11: Recommendation
    # Pick best score (higher better). Tie: lower row number
    recommended = max(evaluated, key=lambda x: (x[2], -x[0]))
    rec_num, rec_row, rec_score = recommended

    # Find 3 nearest neighbors (excluding itself)
    neighbors = []
    for rnum, r, _ in evaluated:
        if rnum == rec_num:
            continue
        d = row_distance(rec_row, r, header, bounds, col_types)
        neighbors.append((d, rnum, r))
    neighbors.sort(key=lambda x: x[0])
    neighbors = neighbors[:3]

    # Build JSON for CAR-11
    neighbor_json = []
    for _, nnum, nrow in neighbors:
        val_dict = {c: nrow[header.index(c)] for c in goal_names}
        neighbor_json.append({"row": nnum, "values": val_dict})

    car11_ans = {"recommended": rec_num, "neighbors": neighbor_json}

    # CAR-12: Compare two vehicles
    # Use highest and lowest score rows
    a = max(evaluated, key=lambda x: (x[2], -x[0]))
    b = min(evaluated, key=lambda x: (x[2], -x[0]))
    a_num, a_row, a_score = a
    b_num, b_row, b_score = b
    preferred = a_num  # higher score

    # Differences between a and b
    diff_list = []
    for idx, name in enumerate(header):
        v_a = a_row[idx]
        v_b = b_row[idx]
        if v_a == v_b:
            continue
        if v_a is None or v_b is None:
            continue
        diff_list.append({"feature": name, "a": v_a, "b": v_b})
    # Ensure at least one
    if not diff_list:
        diff_list = [{"feature": goal_names[0], "a": a_row[header.index(goal_names[0])],
                     "b": b_row[header.index(goal_names[0])] }]

    # Tradeoffs: goals where b is better
    tradeoffs = []
    for g in goal_names:
        idx = header.index(g)
        va = a_row[idx]
        vb = b_row[idx]
        if va is None or vb is None:
            continue
        if g.endswith("-"):  # minimize
            better = vb < va
        else:
            better = vb > va
        if better:
            tradeoffs.append(g)

    car12_ans = {
        "a": a_num,
        "b": b_num,
        "preferred": preferred,
        "scores": {str(a_num): round(a_score, 4), str(b_num): round(b_score, 4)},
        "differences": diff_list[:5],  # limit to 5 for brevity
        "tradeoffs": tradeoffs[:5]
    }

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    print("== CAR-11 ==")
    print(f"Recommended row: {rec_num} (score={round(rec_score,4)})")
    print("Neighbors:")
    for n in neighbor_json:
        vals = ", ".join(f"{k}={v}" for k, v in n["values"].items())
        print(f"  Row {n['row']}: {vals}")
    print(f"ANSWER CAR-11 {json.dumps(car11_ans, separators=(',', ':'))}")

    print("\n== CAR-12 ==")
    print(f"Comparing row {a_num} (score={round(a_score,4)}) vs row {b_num} (score={round(b_score,4)})")
    print(f"Preferred: row {preferred}")
    print("Differences:")
    for d in diff_list[:3]:
        print(f"  {d['feature']}: a={d['a']} b={d['b']}")
    print("Tradeoffs where second is better:")
    for t in tradeoffs[:3]:
        print(f"  {t}")
    print(f"ANSWER CAR-12 {json.dumps(car12_ans, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

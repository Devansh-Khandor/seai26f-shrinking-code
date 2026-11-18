#!/usr/bin/env python3
"""
Skill 07: Inventory evaluation strategy.

This script implements CAR‑13 and CAR‑14.

CAR‑13: Identify a vehicle with a strong balance of fuel efficiency (Mpg+), weight (Lbs-),
and acceleration (Acc+).  We evaluate as few candidate vehicles as reasonably possible
under the inspection budget (at most min(30, n//5) rows).  The chosen strategy
evaluates the rows with the highest composite score until the budget is exhausted;
the best vehicle among those evaluated is returned.

CAR‑14: Recommend an order for evaluating vehicles to discover promising options early.
Again we use the composite score but report the top 30 rows as the inspection order.
"""

import sys
import json
from typing import List, Dict, Tuple
import math

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, normalize
except Exception:
    # Minimal fall‑back implementations
    import csv

    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except Exception:
            return math.nan

    def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
        if max_val == min_val:
            return 1.0
        if maximize:
            return (val - min_val) / (max_val - min_val)
        return (max_val - val) / (max_val - min_val)

# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def compute_goal_min_max(rows: List[Dict[str, str]],
                         goal_cols: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per‑goal min and max across all rows that have the value."""
    col_min = {}
    col_max = {}
    for col in goal_cols:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        if vals:
            col_min[col] = min(vals)
            col_max[col] = max(vals)
        else:
            col_min[col] = col_max[col] = 0.0
    return col_min, col_max

def composite_scores(rows: List[Dict[str, str]],
                     goal_cols: Dict[str, str],
                     col_min: Dict[str, float],
                     col_max: Dict[str, float]) -> List[Tuple[int, float]]:
    """Return list of (row_idx, score) for rows with complete goal data."""
    scores = []
    for idx, row in enumerate(rows, start=1):
        if any(is_missing(row[col]) for col in goal_cols):
            continue
        score = 0.0
        for col, dirn in goal_cols.items():
            val = to_float(row[col])
            score += normalize(val, col_min[col], col_max[col], maximize=(dirn == '+'))
        scores.append((idx, score))
    return scores

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify goal columns (ending in '+' or '-')
    goal_columns = {col: col[-1] for col in header if col.endswith('+') or col.endswith('-')}

    # Compute min/max per goal column
    col_min, col_max = compute_goal_min_max(rows, goal_columns)

    # Compute composite scores for all evaluable rows
    scores = composite_scores(rows, goal_columns, col_min, col_max)

    # Sort by score descending, then by row number ascending for stable tie‑break
    scores.sort(key=lambda kv: (-kv[1], kv[0]))

    # Budget: at most min(30, number of rows // 5)
    budget = min(30, len(rows) // 5)

    # Evaluate rows in this order until budget is exhausted
    evaluated = [row_idx for row_idx, _ in scores[:budget]]
    best_row = evaluated[0] if evaluated else None

    # CAR-13 output
    print("== CAR-13 ==")
    print(f"Evaluated rows (first {len(evaluated)}): {evaluated}")
    print(f"Best row: {best_row}")
    print()

    # Machine‑checked answer for CAR‑13
    car13_json = {"evaluated": evaluated, "best": best_row}
    print(f"ANSWER CAR-13 {json.dumps(car13_json, separators=(',', ':'))}")

    # CAR-14 output
    print("== CAR-14 ==")
    print(f"Recommended inspection order (top {len(evaluated)}): {evaluated}")
    print()

    # Machine‑checked answer for CAR‑14
    car14_json = {"order": evaluated}
    print(f"ANSWER CAR-14 {json.dumps(car14_json, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

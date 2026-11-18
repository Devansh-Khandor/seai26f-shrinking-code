#!/usr/bin/env python3
"""
Skill 08: Alternative vehicle recommendation for poorly rated or sub‑optimal inventory.
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple, Any

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, normalize
except Exception:
    # Minimal fall‑back implementations
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
# Utility functions
# --------------------------------------------------------------------------- #

def compute_goal_min_max(rows: List[Dict[str, str]], goal_cols: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per‑goal min and max across rows that have the value."""
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

def compute_composite(row: Dict[str, str], goal_cols: Dict[str, str],
                      col_min: Dict[str, float], col_max: Dict[str, float]) -> float:
    score = 0.0
    for col, dirn in goal_cols.items():
        v = to_float(row[col])
        score += normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
    return score

def input_distance(row1: Dict[str, str], row2: Dict[str, str], input_cols: List[str]) -> float:
    dist = 0.0
    for col in input_cols:
        v1 = to_float(row1[col])
        v2 = to_float(row2[col])
        if math.isnan(v1) or math.isnan(v2):
            return math.inf
        dist += (v1 - v2) ** 2
    return dist

def mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify goal columns and input columns
    goal_columns = {col: col[-1] for col in header if col.endswith('+') or col.endswith('-')}
    input_columns = [col for col in header if col not in goal_columns and not col.endswith('X')]

    # Compute min/max for each goal column
    col_min, col_max = compute_goal_min_max(rows, goal_columns)

    # Compute composite scores for rows that have all goal columns
    scored_rows: List[Tuple[int, float, Dict[str, str]]] = []
    for idx, row in enumerate(rows, start=1):
        if any(is_missing(row[col]) for col in goal_columns):
            continue
        score = compute_composite(row, goal_columns, col_min, col_max)
        scored_rows.append((idx, score, row))

    if not scored_rows:
        print("== CAR-15 ==")
        print("No rows with complete goal data.")
        print()
        print("ANSWER CAR-15 {}")
        print("\n== CAR-16 ==")
        print("No rows with complete goal data.")
        print()
        print("ANSWER CAR-16 {}")
        return

    # ------------------------------------------------------- #
    # CAR-15 : Poorly rated vehicle and alternatives
    # ------------------------------------------------------- #

    # Find the worst row (lowest composite score)
    worst = min(scored_rows, key=lambda kv: kv[1])
    worst_idx, worst_score, worst_row = worst

    # Find alternatives: rows with higher score, all input cols present, and minimal distance
    alternatives = []
    for idx, score, row in scored_rows:
        if idx == worst_idx:
            continue
        # require higher score
        if score <= worst_score:
            continue
        # ensure all input cols present
        if any(is_missing(row[col]) for col in input_columns):
            continue
        dist = input_distance(worst_row, row, input_columns)
        alternatives.append((dist, -score, idx, row))  # negative score for sorting desc

    # Choose best alternative
    if alternatives:
        alternatives.sort()
        _, _, alt_idx, alt_row = alternatives[0]
        # Build changes dict (only differing input columns)
        changes = {}
        for col in input_columns:
            v1 = to_float(worst_row[col])
            v2 = to_float(alt_row[col])
            if v1 != v2:
                changes[col] = [v1, v2]
    else:
        alt_idx = None
        changes = {}

    # Print CAR-15 results
    print("== CAR-15 ==")
    print(f"Poorly rated vehicle: row {worst_idx}")
    if alt_idx is not None:
        print(f"Best alternative: row {alt_idx} (score higher, similar characteristics)")
        print(f"Number of differing input columns: {len(changes)}")
    else:
        print("No suitable alternative found.")
    print()

    # Machine‑checked answer
    car15_json = {"row": worst_idx, "alternatives": [
        {"row": alt_idx, "changes": changes} if alt_idx is not None else {}
    ]}
    print(f"ANSWER CAR-15 {json.dumps(car15_json, separators=(',', ':'))}")

    # ------------------------------------------------------- #
    # CAR-16 : Vehicle not meeting preferred targets
    # ------------------------------------------------------- #

    # Compute threshold for each goal column (mean)
    thresholds = {}
    for col in goal_columns:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        thresholds[col] = mean(vals)

    # Evaluate target satisfaction for the worst row
    def satisfies(row: Dict[str, str]) -> int:
        cnt = 0
        for col, dirn in goal_columns.items():
            v = to_float(row[col])
            thr = thresholds[col]
            if dirn == '+':
                if v >= thr:
                    cnt += 1
            else:
                if v <= thr:
                    cnt += 1
        return cnt

    worst_targets = satisfies(worst_row)

    # Find alternative that meets more targets and has minimal input differences
    candidate_rows = []
    for idx, score, row in scored_rows:
        if idx == worst_idx:
            continue
        # skip if missing any input column
        if any(is_missing(row[col]) for col in input_columns):
            continue
        sat = satisfies(row)
        if sat <= worst_targets:
            continue
        diff_cnt = sum(1 for col in input_columns
                       if to_float(row[col]) != to_float(worst_row[col]))
        candidate_rows.append((diff_cnt, -sat, idx, row))

    if candidate_rows:
        candidate_rows.sort()
        diff_cnt, _, alt_idx, alt_row = candidate_rows[0]
        changed_cols = [col for col in input_columns
                        if to_float(alt_row[col]) != to_float(worst_row[col])]
    else:
        alt_idx = None
        changed_cols = []

    # Print CAR-16 results
    print("\n== CAR-16 ==")
    print(f"Vehicle not meeting preferred targets: row {worst_idx}")
    print(f"Targets thresholds: {thresholds}")
    if alt_idx is not None:
        print(f"Alternative row: {alt_idx} (meets {satisfies(alt_row)} targets)")
        print(f"Changed input columns: {changed_cols}")
    else:
        print("No alternative meeting more targets found.")
    print()

    # Machine‑checked answer
    car16_json = {"row": worst_idx,
                  "targets": thresholds,
                  "alternative": alt_idx,
                  "changed": changed_cols}
    print(f"ANSWER CAR-16 {json.dumps(car16_json, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

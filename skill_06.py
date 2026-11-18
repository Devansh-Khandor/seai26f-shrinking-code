#!/usr/bin/env python3
"""
Skill 06: Explain recommendation and trade‑offs.

Usage:
    python skill_06.py <csv_path>
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Import helper functions from skill_01
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
# Core logic
# --------------------------------------------------------------------------- #

def compute_goal_stats(rows: List[Dict[str, str]], goal_cols: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return per‑goal min and max over rows that have a value."""
    col_min, col_max = {}, {}
    for col in goal_cols:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        if vals:
            col_min[col] = min(vals)
            col_max[col] = max(vals)
        else:
            col_min[col] = col_max[col] = 0.0
    return col_min, col_max

def compute_scores(rows: List[Dict[str, str]],
                   goal_cols: Dict[str, str],
                   col_min: Dict[str, float],
                   col_max: Dict[str, float]) -> List[Tuple[int, float, Dict[str, str]]]:
    """Return list of (row_idx, score, row_dict) for rows with complete goal data."""
    scored = []
    for idx, row in enumerate(rows, start=1):
        if any(is_missing(row[col]) for col in goal_cols):
            continue
        score = 0.0
        for col, dirn in goal_cols.items():
            v = to_float(row[col])
            best = normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
            score += best
        scored.append((idx, score, row))
    return scored

def euclidean_sq(row_a: Dict[str, str], row_b: Dict[str, str], goal_cols: Dict[str, str]) -> float:
    s = 0.0
    for col in goal_cols:
        va = to_float(row_a[col])
        vb = to_float(row_b[col])
        s += (va - vb) ** 2
    return s

def find_neighbors(recommended_row: Tuple[int, float, Dict[str, str]],
                   scored_rows: List[Tuple[int, float, Dict[str, str]]],
                   goal_cols: Dict[str, str],
                   k: int = 3) -> List[Dict]:
    """Return k nearest neighbors (excluding recommended) based on Euclidean distance in goal space."""
    idx, _, row_a = recommended_row
    distances = []
    for j, (_, _, row_b) in enumerate(scored_rows):
        if j == scored_rows.index(recommended_row):
            continue
        dist = euclidean_sq(row_a, row_b, goal_cols)
        distances.append((dist, row_b))
    distances.sort(key=lambda x: x[0])
    neighbors = []
    for _, row in distances[:k]:
        val_dict = {col: row[col] for col in goal_cols}
        neighbors.append({"row": next(i for i, (_, _, r) in enumerate(scored_rows, start=1) if r is row),
                         "values": val_dict})
    return neighbors

def find_tradeoff_pair(scored_rows: List[Tuple[int, float, Dict[str, str]]],
                       goal_cols: Dict[str, str]) -> Tuple[int, int, str, List[Dict], List[str]]:
    """
    Find a pair (a,b) where a is preferred (higher score) and there is at least one goal
    where b is better. Return (a_idx, b_idx, preferred, differences, tradeoffs).
    If none found, return second best as b anyway.
    """
    # sort descending
    sorted_rows = sorted(scored_rows, key=lambda kv: kv[1], reverse=True)
    a_idx, a_score, a_row = sorted_rows[0]
    for cand in sorted_rows[1:]:
        b_idx, b_score, b_row = cand
        # Compute tradeoffs
        tradeoffs = []
        for col, dirn in goal_cols.items():
            va = to_float(a_row[col])
            vb = to_float(b_row[col])
            if dirn == '+':
                if vb > va:
                    tradeoffs.append(col)
            else:
                if vb < va:
                    tradeoffs.append(col)
        if tradeoffs:
            # Differences: list all goal columns where values differ
            differences = []
            for col in goal_cols:
                if a_row[col] != b_row[col]:
                    differences.append({"feature": col,
                                        "a": a_row[col],
                                        "b": b_row[col]})
            # pick first diff if list empty
            if not differences:
                differences.append({"feature": list(goal_cols)[0],
                                    "a": a_row[list(goal_cols)[0]],
                                    "b": b_row[list(goal_cols)[0]]})
            preferred = "a" if a_score >= b_score else "b"
            return a_idx, b_idx, preferred, differences, tradeoffs
    # fallback to second best
    b_idx, b_score, b_row = sorted_rows[1]
    differences = []
    for col in goal_cols:
        if a_row[col] != b_row[col]:
            differences.append({"feature": col,
                                "a": a_row[col],
                                "b": b_row[col]})
    if not differences:
        differences.append({"feature": list(goal_cols)[0],
                            "a": a_row[list(goal_cols)[0]],
                            "b": b_row[list(goal_cols)[0]]})
    tradeoffs = []
    for col, dirn in goal_cols.items():
        va = to_float(a_row[col])
        vb = to_float(b_row[col])
        if dirn == '+':
            if vb > va:
                tradeoffs.append(col)
        else:
            if vb < va:
                tradeoffs.append(col)
    preferred = "a" if a_score >= b_score else "b"
    return a_idx, b_idx, preferred, differences, tradeoffs

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify goal columns and directions
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}

    # Compute min/max for scoring
    col_min, col_max = compute_goal_stats(rows, goal_columns)

    # Compute composite scores
    scored_rows = compute_scores(rows, goal_columns, col_min, col_max)

    if not scored_rows:
        print("== CAR-11 ==")
        print("No rows with complete goal data.")
        print()
        print("ANSWER CAR-11 {}")
        print()
        print("== CAR-12 ==")
        print("No rows with complete goal data.")
        print()
        print("ANSWER CAR-12 {}")
        return

    # Find recommended vehicle (top score)
    recommended_row = max(scored_rows, key=lambda kv: kv[1])

    # ----- CAR-11 -----
    rec_idx, rec_score, rec_row = recommended_row
    neighbors = find_neighbors(recommended_row, scored_rows, goal_columns, k=3)

    # Human readable
    print("== CAR-11 ==")
    print(f"Recommended vehicle: row {rec_idx} (score={rec_score:.4f})")
    print("Neighbors (closest in goal space):")
    for n in neighbors:
        vals = n["values"]
        vals_str = ", ".join(f"{c}={vals[c]}" for c in sorted(vals))
        print(f"  Row {n['row']}: {vals_str}")
    print()

    # Machine‑checked answer
    neighbors_out = []
    for n in neighbors:
        neighbors_out.append({"row": n["row"], "values": {c: n["values"][c] for c in goal_columns}})
    car11_json = {"recommended": rec_idx, "neighbors": neighbors_out}
    print(f"ANSWER CAR-11 {json.dumps(car11_json, separators=(',', ':'))}")

    # ----- CAR-12 -----
    a_idx, b_idx, preferred, differences, tradeoffs = find_tradeoff_pair(scored_rows, goal_columns)
    a_score = next(score for idx, score, _ in scored_rows if idx == a_idx)
    b_score = next(score for idx, score, _ in scored_rows if idx == b_idx)

    print("\n== CAR-12 ==")
    print(f"Comparing row {a_idx} (score={a_score:.4f}) and row {b_idx} (score={b_score:.4f})")
    print(f"Preferred: {'a' if preferred == 'a' else 'b'}")
    print("Differences in goal values:")
    for diff in differences[:3]:
        print(f"  {diff['feature']}: a={diff['a']}, b={diff['b']}")
    print("Trade‑offs where non‑preferred row is better:")
    print("  " + ", ".join(tradeoffs))
    print()

    car12_json = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {str(a_idx): a_score, str(b_idx): b_score},
        "differences": differences[:3],
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER CAR-12 {json.dumps(car12_json, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

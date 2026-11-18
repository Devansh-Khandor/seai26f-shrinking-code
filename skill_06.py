#!/usr/bin/env python3
"""
Skill 06: NASA-11 and NASA-12

This script answers two requests:

- NASA-11: explain the effort estimate for a held‑out project using
  similar historical projects.
- NASA-12: compare two held‑out proposals, show why one is preferred,
  and list the trade‑offs.

The implementation re‑uses helper functions from the previous skills
(`skill_01.py`, `skill_04.py`, `skill_05.py`).  Only the Python
standard library is used.
"""

import csv
import json
import math
import sys
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Common helper functions (copied from skill_01.py)

GOAL_COLUMNS = {
    "Kloc+": "max",      # maximize
    "Effort-": "min",    # minimize
    "Defects-": "min",   # minimize
}

def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    """
    Read CSV file and return a list of row dictionaries, a list of
    1‑based row numbers, and a list of header columns.
    """
    rows: List[Dict[str, Any]] = []
    row_nums: List[int] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for rn, rec in enumerate(reader, start=1):
            row_nums.append(rn)
            parsed: Dict[str, Any] = {}
            for key, val in rec.items():
                val = val.strip()
                if val == "?":
                    parsed[key] = None
                else:
                    # try numeric conversion
                    try:
                        if '.' in val or 'e' in val.lower():
                            parsed[key] = float(val)
                        else:
                            parsed[key] = int(val)
                    except ValueError:
                        parsed[key] = val
            rows.append(parsed)
    return rows, row_nums, header

# ----------------------------------------------------------------------
# Utility functions

def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')

# ----------------------------------------------------------------------
# NASA-11 logic

def find_first_held_out(rows: List[Dict[str, Any]], row_nums: List[int]) -> int:
    """Return the 1‑based index (0‑based) of the first held‑out row."""
    for idx, rn in enumerate(row_nums):
        if rn % 5 == 0:
            return idx
    raise ValueError("No held‑out rows found")

def similarity_score(row_a: Dict[str, Any], row_b: Dict[str, Any],
                     non_goal_cols: List[str]) -> int:
    """Count matching non‑goal cells between two rows."""
    score = 0
    for col in non_goal_cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va is None or vb is None:
            continue
        if va == vb:
            score += 1
    return score

def find_top_neighbors(project_idx: int,
                       rows: List[Dict[str, Any]],
                       non_goal_cols: List[str],
                       n: int = 3) -> List[int]:
    """Return indices of top n neighbors of the project (excluding held‑out)."""
    scores: List[Tuple[int, int]] = []  # (score, index)
    for idx, row in enumerate(rows):
        if idx == project_idx:
            continue
        # Skip other held‑out rows
        if (idx + 1) % 5 == 0:
            continue
        score = similarity_score(rows[project_idx], row, non_goal_cols)
        scores.append((score, idx))
    # Sort descending by score, then ascending by row number
    scores.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in scores[:n]]

def estimate_effort(neighbor_indices: List[int], rows: List[Dict[str, Any]]) -> float:
    """Mean Effort- of the given neighbors."""
    effs = [rows[i]["Effort-"] for i in neighbor_indices
            if is_numeric(rows[i].get("Effort-"))]
    return mean(effs)

# ----------------------------------------------------------------------
# NASA-12 logic

def risk_score(row: Dict[str, Any]) -> float:
    """
    Simple risk score: higher Effort- and Defects- increase risk,
    higher Kloc+ decreases risk.
    """
    e = row.get("Effort-")
    d = row.get("Defects-")
    k = row.get("Kloc+")
    if not (is_numeric(e) and is_numeric(d) and is_numeric(k)):
        return float('nan')
    return e + d - k

def better_goal(a: Dict[str, Any], b: Dict[str, Any], goal: str) -> bool:
    """
    Return True if row a is better than row b on the given goal.
    For "max" goals, higher is better; for "min", lower is better.
    """
    a_val = a.get(goal)
    b_val = b.get(goal)
    if not (is_numeric(a_val) and is_numeric(b_val)):
        return False
    direction = GOAL_COLUMNS.get(goal)
    if direction == "max":
        return a_val > b_val
    else:  # "min"
        return a_val < b_val

# ----------------------------------------------------------------------
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, row_nums, header = read_data(csv_path)

    # Identify non‑goal columns (exclude goals and columns ending with 'X')
    non_goal_cols = [c for c in header
                     if c not in GOAL_COLUMNS and not c.endswith('X')]

    # ------------------------------------------------------------------
    # NASA‑11

    # Find first held‑out row
    proj_idx = find_first_held_out(rows, row_nums)
    proj_row_num = row_nums[proj_idx]
    proj_row = rows[proj_idx]

    neighbor_indices = find_top_neighbors(proj_idx, rows, non_goal_cols, n=3)
    neighbors = []
    for ni in neighbor_indices:
        r = rows[ni]
        neighbors.append({
            "row": row_nums[ni],
            "values": {col: r[col] for col in header if r[col] is not None}
        })

    eff_estimate = estimate_effort(neighbor_indices, rows)

    # Human‑readable output
    print("== NASA-11 ==")
    print(f"Project row: {proj_row_num}")
    print(f"Estimated Effort-: {eff_estimate:.2f}")
    print(f"Top 3 neighbors: {[n['row'] for n in neighbors]}")

    # JSON answer
    answer_n11 = {
        "project": proj_row_num,
        "estimate": round(eff_estimate, 6),
        "neighbors": neighbors
    }
    print(f"ANSWER NASA-11 {json.dumps(answer_n11)}")

    # ------------------------------------------------------------------
    # NASA‑12

    # Pick two held‑out rows (first two)
    held_out_indices = [i for i, rn in enumerate(row_nums) if rn % 5 == 0]
    if len(held_out_indices) < 2:
        print("Not enough held‑out rows for NASA-12")
        sys.exit(1)
    a_idx, b_idx = held_out_indices[0], held_out_indices[1]
    a_row_num, b_row_num = row_nums[a_idx], row_nums[b_idx]
    a_row, b_row = rows[a_idx], rows[b_idx]

    a_score, b_score = risk_score(a_row), risk_score(b_row)
    scores_dict = {str(a_row_num): a_score, str(b_row_num): b_score}

    # Determine preferred (lower risk)
    if a_score <= b_score:
        preferred = a_row_num
    else:
        preferred = b_row_num

    # Differences on non‑goal columns
    differences = []
    for col in non_goal_cols:
        va = a_row.get(col)
        vb = b_row.get(col)
        if va is None or vb is None:
            continue
        differences.append({
            "feature": col,
            "a": va,
            "b": vb
        })

    # Trade‑offs: goal columns where non‑preferred is better
    non_preferred = b_row if preferred == a_row_num else a_row
    tradeoffs = []
    for g in GOAL_COLUMNS:
        if better_goal(non_preferred, a_row if preferred == a_row_num else b_row, g):
            tradeoffs.append(g)

    # Human‑readable output
    print("\n== NASA-12 ==")
    print(f"Rows compared: {a_row_num} vs {b_row_num}")
    print(f"Preferred: {preferred}")
    print(f"Scores: {scores_dict}")
    print(f"Differences sample: {differences[:2]}")

    # JSON answer
    answer_n12 = {
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": scores_dict,
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER NASA-12 {json.dumps(answer_n12)}")

if __name__ == "__main__":
    main()

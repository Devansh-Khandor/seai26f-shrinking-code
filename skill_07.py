#!/usr/bin/env python3
"""
skill_07.py

Implements CAR-13 and CAR-14 for the auto93 dataset.

CAR-13:
    Identify a vehicle with a strong balance of fuel efficiency, weight, and
    acceleration while evaluating as few candidate vehicles as reasonably
    possible.

CAR-14:
    When we have a limited inspection budget, recommend which vehicles
    should be evaluated first so that we can discover promising inventory
    options early.

The script is self‑contained and uses only the Python standard library.
"""

import csv
import json
import sys
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def read_data(path: str) -> List[Dict[str, Any]]:
    """
    Read CSV into list of dictionaries.
    '?' or '' -> None.
    numeric strings -> float.
    others kept as str.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v == '?' or v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def goal_columns(rows: List[Dict[str, Any]]) -> Set[str]:
    """Return set of column names that are goals (end with '-' or '+')."""
    if not rows:
        return set()
    return {k for k in rows[0] if k.endswith('-') or k.endswith('+')}


# ------------------------------------------------------------------
# Normalisation utilities
# ------------------------------------------------------------------
def compute_min_max(rows: List[Dict[str, Any]], goals: Set[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for col in goals:
        vals = [r[col] for r in rows if r[col] is not None]
        mins[col] = min(vals) if vals else 0.0
        maxs[col] = max(vals) if vals else 0.0
    return mins, maxs


def normalized_value(val: float, mn: float, mx: float, minimize: bool) -> float:
    if mx == mn:
        return 0.0
    if minimize:
        return (mx - val) / (mx - mn)
    return (val - mn) / (mx - mn)


def compute_row_score(row: Dict[str, Any],
                      goals: Set[str],
                      mins: Dict[str, float],
                      maxs: Dict[str, float]) -> float | None:
    """Compute a composite score for the row using Lbs-, Acc+, Mpg+."""
    if any(row[col] is None for col in goals):
        return None
    # Only consider the three specified goals
    needed = {"Lbs-": True, "Acc+": True, "Mpg+": True}
    score = 0.0
    for col in needed:
        val = row[col]
        mn, mx = mins[col], maxs[col]
        score += normalized_value(val, mn, mx, minimize=col.endswith('-'))
    return score


# ------------------------------------------------------------------
# CAR-13
# ------------------------------------------------------------------
def car_13(rows: List[Dict[str, Any]]) -> Tuple[List[int], int]:
    goals = goal_columns(rows)
    mins, maxs = compute_min_max(rows, goals)

    # Evaluate at most 30 rows, only those with all goal values.
    limit = 30
    evaluated: List[int] = []
    best_idx = None
    best_score = -1.0

    for idx, row in enumerate(rows, start=1):
        if idx in evaluated:
            continue
        score = compute_row_score(row, goals, mins, maxs)
        if score is None:
            continue
        evaluated.append(idx)
        if score > best_score or (score == best_score and best_idx is not None and idx < best_idx):
            best_score = score
            best_idx = idx
        if len(evaluated) >= limit:
            break

    # If no evaluable rows, best_idx remains None
    if best_idx is None and evaluated:
        best_idx = evaluated[0]
    return evaluated, best_idx


# ------------------------------------------------------------------
# CAR-14
# ------------------------------------------------------------------
def car_14(rows: List[Dict[str, Any]]) -> List[int]:
    goals = goal_columns(rows)
    mins, maxs = compute_min_max(rows, goals)

    # Compute scores for all rows that have complete goal data
    scored_rows: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows, start=1):
        score = compute_row_score(row, goals, mins, maxs)
        if score is not None:
            scored_rows.append((idx, score))

    # Order by score descending; ties by row number ascending
    scored_rows.sort(key=lambda x: (-x[1], x[0]))

    # Budget: at most 30 rows, but not more than available
    limit = min(30, len(scored_rows))
    order = [idx for idx, _ in scored_rows[:limit]]
    return order


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)

    # CAR-13
    print("== CAR-13 ==")
    evaluated, best = car_13(rows)
    print(f"Evaluated rows (up to 30): {evaluated}")
    print(f"Best row found: {best}")
    ans13 = {"evaluated": evaluated, "best": best}
    print(f"ANSWER CAR-13 {json.dumps(ans13, ensure_ascii=False)}")

    # CAR-14
    print("\n== CAR-14 ==")
    order = car_14(rows)
    print(f"Recommended evaluation order (top {len(order)} rows): {order}")
    ans14 = {"order": order}
    print(f"ANSWER CAR-14 {json.dumps(ans14, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

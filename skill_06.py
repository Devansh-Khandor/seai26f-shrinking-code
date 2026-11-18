#!/usr/bin/env python3
"""
skill_06.py

Implements CAR-11 and CAR-12 for the auto93 dataset.

CAR-11:
    When the system recommends a vehicle, explain its recommendation
    using understandable comparisons with similar vehicles in our inventory.

CAR-12:
    When two vehicles have different strengths, explain why one receives
    a higher overall recommendation and show the tradeoffs involved.

The script is self‑contained and uses only the Python standard library.
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def read_data(path: str) -> List[Dict[str, Any]]:
    """
    Read CSV into list of dicts.
    '?' or '' -> None.
    numeric strings -> float.
    others keep as str.
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
    """
    Return set of column names that are goals.
    Goals end with '-' (minimize) or '+' (maximize).
    """
    return {k for k in rows[0] if k.endswith('-') or k.endswith('+')}


# ------------------------------------------------------------------
# Scoring utilities
# ------------------------------------------------------------------
def compute_min_max(rows: List[Dict[str, Any]], goals: Set[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for col in goals:
        vals = [r[col] for r in rows if r[col] is not None]
        mins[col] = min(vals) if vals else 0.0
        maxs[col] = max(vals) if vals else 0.0
    return mins, maxs


def performance_score(row: Dict[str, Any], goals: Set[str], mins: Dict[str, float], maxs: Dict[str, float]) -> float:
    """Compute overall performance score for a row."""
    score = 0.0
    for col in goals:
        v = row.get(col)
        if v is None:
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm = 0.0
        else:
            if col.endswith('-'):
                norm = (mx - v) / (mx - mn)
            else:
                norm = (v - mn) / (mx - mn)
        score += norm
    return score


def normalized_vector(row: Dict[str, Any], goals: List[str], mins: Dict[str, float], maxs: Dict[str, float]) -> List[float]:
    vec: List[float] = []
    for col in goals:
        v = row[col]
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm = 0.0
        else:
            if col.endswith('-'):
                norm = (mx - v) / (mx - mn)
            else:
                norm = (v - mn) / (mx - mn)
        vec.append(norm)
    return vec


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


# ------------------------------------------------------------------
# CAR-11
# ------------------------------------------------------------------
def car_11(rows: List[Dict[str, Any]], goals: Set[str]) -> Dict[str, Any]:
    # Compute mins and maxs
    mins, maxs = compute_min_max(rows, goals)
    goal_list = sorted(goals)

    # Compute score for each row that has all goal values
    scored_rows: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows, start=1):
        if any(row[col] is None for col in goals):
            continue
        sc = performance_score(row, goals, mins, maxs)
        if sc is not None:
            scored_rows.append((idx, sc))

    if not scored_rows:
        # No fully scored rows; return empty
        return {"recommended": None, "neighbors": []}

    # Find recommended (best score, lowest row number on tie)
    recommended_idx, best_score = max(scored_rows, key=lambda x: (x[1], -x[0]))
    recommended_row = rows[recommended_idx - 1]

    # Normalized vectors
    vecs: Dict[int, List[float]] = {}
    for idx, row in enumerate(rows, start=1):
        if any(row[col] is None for col in goals):
            continue
        vecs[idx] = normalized_vector(row, goal_list, mins, maxs)

    # Compute distances to recommended
    rec_vec = vecs[recommended_idx]
    distances: List[Tuple[int, float]] = []
    for idx, vec in vecs.items():
        if idx == recommended_idx:
            continue
        dist = euclidean_distance(rec_vec, vec)
        distances.append((idx, dist))

    # Get three nearest neighbors
    distances.sort(key=lambda x: (x[1], x[0]))
    neighbor_idxs = [idx for idx, _ in distances[:3]]

    # Build neighbor values dict
    neighbors = []
    for nidx in neighbor_idxs:
        nrow = rows[nidx - 1]
        vals = {col: nrow[col] for col in goal_list}
        neighbors.append({"row": nidx, "values": vals})

    return {"recommended": recommended_idx, "neighbors": neighbors}


# ------------------------------------------------------------------
# CAR-12
# ------------------------------------------------------------------
def car_12(rows: List[Dict[str, Any]], goals: Set[str]) -> Dict[str, Any]:
    mins, maxs = compute_min_max(rows, goals)
    goal_list = sorted(goals)

    # Score all rows that have complete goals
    scored_rows: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows, start=1):
        if any(row[col] is None for col in goals):
            continue
        sc = performance_score(row, goals, mins, maxs)
        if sc is not None:
            scored_rows.append((idx, sc))

    if len(scored_rows) < 2:
        return {"a": None, "b": None, "preferred": None, "scores": {}, "differences": [], "tradeoffs": []}

    # Pick best and worst
    scored_rows.sort(key=lambda x: x[1])  # ascending
    worst_idx, worst_score = scored_rows[0]
    best_idx, best_score = scored_rows[-1]

    a_idx, a_score = best_idx, best_score
    b_idx, b_score = worst_idx, worst_score
    # Ensure 'a' is the preferred (higher score)
    if a_score < b_score:
        a_idx, b_idx = b_idx, a_idx
        a_score, b_score = b_score, a_score

    a_row = rows[a_idx - 1]
    b_row = rows[b_idx - 1]

    # Differences in goal values
    differences = []
    for col in goal_list:
        a_val = a_row[col]
        b_val = b_row[col]
        differences.append({"feature": col, "a": a_val, "b": b_val})

    # Tradeoffs: goals where the non-preferred row (b) is better
    tradeoffs = []
    for col in goal_list:
        a_val = a_row[col]
        b_val = b_row[col]
        if col.endswith('-'):
            # lower is better
            if b_val < a_val:
                tradeoffs.append(col)
        else:
            if b_val > a_val:
                tradeoffs.append(col)

    return {
        "a": a_idx,
        "b": b_idx,
        "preferred": a_idx,
        "scores": {"a": round(a_score, 4), "b": round(b_score, 4)},
        "differences": differences,
        "tradeoffs": tradeoffs,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)
    goals = goal_columns(rows)

    # ---- CAR-11 ----
    print("== CAR-11 ==")
    ans11 = car_11(rows, goals)
    if ans11["recommended"] is None:
        print("No fully scored vehicle found.")
    else:
        rec = ans11["recommended"]
        rec_row = rows[rec - 1]
        rec_score = performance_score(rec_row, goals,
                                      *compute_min_max(rows, goals))
        print(f"Recommended vehicle: row {rec} with score {rec_score:.2f}")
        neighbor_rows = [n["row"] for n in ans11["neighbors"]]
        print(f"Top 3 similar vehicles: rows {neighbor_rows}")
    print(f"ANSWER CAR-11 {json.dumps(ans11, ensure_ascii=False)}")

    # ---- CAR-12 ----
    print("\n== CAR-12 ==")
    ans12 = car_12(rows, goals)
    if ans12["a"] is None or ans12["b"] is None:
        print("Not enough fully scored vehicles to compare.")
    else:
        a = ans12["a"]
        b = ans12["b"]
        pref = ans12["preferred"]
        print(f"Compare row {a} (preferred) vs row {b} (not preferred)")
        print(f"Preferred score: {ans12['scores']['a']:.4f}, Non-preferred score: {ans12['scores']['b']:.4f}")
    print(f"ANSWER CAR-12 {json.dumps(ans12, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

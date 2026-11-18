#!/usr/bin/env python3
"""
skill_06.py: Recommendations and trade‑off explanations for auto93 dataset.
Usage: python skill_06.py <csv_path>
"""

import json
import sys
from math import sqrt
from typing import Dict, List, Tuple

# Reuse helpers from the original skill_01
from skill_01 import read_csv, build_table, normalize  # type: ignore
import ezr


# ------------------------------------------------------------------
# Common utilities

def goal_columns(tbl: ezr.TBL) -> List[ezr.COL]:
    """Return list of goal columns (goal != 0)."""
    return [col for col in tbl.cols.values() if col.goal != 0]


def compute_min_max(tbl: ezr.TBL, goal_cols: List[ezr.COL]) -> Dict[int, Tuple[float, float]]:
    """Return a dict mapping column index to (min, max) over all rows."""
    result: Dict[int, Tuple[float, float]] = {}
    for col in goal_cols:
        values = [r[col.at] for r in tbl.rows if r[col.at] != "?"]
        if values:
            result[col.at] = (min(values), max(values))
        else:
            result[col.at] = (0.0, 0.0)
    return result


def row_score(tbl: ezr.TBL, row: List) -> float:
    """Compute the overall score of a row based on all goal columns."""
    goal_cols = goal_columns(tbl)
    mins_maxs = compute_min_max(tbl, goal_cols)
    score = 0.0
    for col in goal_cols:
        val = row[col.at]
        if val == "?":
            continue
        mn, mx = mins_maxs[col.at]
        if mx == mn:
            comp = 1.0
        else:
            if col.goal == 1:  # maximize
                comp = (val - mn) / (mx - mn)
            else:  # minimize
                comp = (mx - val) / (mx - mn)
        score += comp
    return score


def goal_vector(row: List, goal_cols: List[ezr.COL]) -> List[float]:
    """Return the vector of goal values for a row, skipping missing values."""
    vec = []
    for col in goal_cols:
        val = row[col.at]
        if val != "?":
            vec.append(float(val))
    return vec


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Euclidean distance between two vectors of equal length."""
    return sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# ------------------------------------------------------------------
# CAR-11: Recommendation with similar vehicles

def car_11(tbl: ezr.TBL) -> Dict:
    """Identify the best vehicle and show a few similar ones."""
    goal_cols = goal_columns(tbl)
    scores = [(row_score(tbl, r), idx + 1, r) for idx, r in enumerate(tbl.rows)]
    best_score, best_idx, best_row = max(scores, key=lambda x: x[0])

    # find neighbors: Euclidean distance on goal vector
    best_vec = goal_vector(best_row, goal_cols)
    distances: List[Tuple[float, int, List]] = []
    for idx, r in enumerate(tbl.rows):
        if idx == best_idx - 1:
            continue
        vec = goal_vector(r, goal_cols)
        if not vec:  # skip if no comparable goals
            continue
        d = euclidean_distance(best_vec, vec)
        distances.append((d, idx + 1, r))

    distances.sort(key=lambda x: x[0])
    neighbors = []
    for _, row_no, row in distances[:3]:
        values = {col.txt: row[col.at] for col in goal_cols}
        neighbors.append({"row": row_no, "values": values})

    return {"recommended": best_idx, "neighbors": neighbors}


# ------------------------------------------------------------------
# CAR-12: Trade‑off explanation between two vehicles

def car_12(tbl: ezr.TBL) -> Dict:
    """Compare the best and worst vehicles in terms of overall score."""
    goal_cols = goal_columns(tbl)
    scores = [(row_score(tbl, r), idx + 1, r) for idx, r in enumerate(tbl.rows)]
    best_score, best_idx, best_row = max(scores, key=lambda x: x[0])
    worst_score, worst_idx, worst_row = min(scores, key=lambda x: x[0])

    # Compute differences on each goal column
    differences = []
    tradeoffs = []
    for col in goal_cols:
        val_a = best_row[col.at]
        val_b = worst_row[col.at]
        if val_a == "?" or val_b == "?":
            continue
        differences.append({"feature": col.txt, "a": val_a, "b": val_b})
        # Determine if worst row is better for this goal
        if col.goal == 1 and val_b > val_a:
            tradeoffs.append(col.txt)
        if col.goal == -1 and val_b < val_a:
            tradeoffs.append(col.txt)

    return {
        "a": best_idx,
        "b": worst_idx,
        "preferred": best_idx,
        "scores": {"a": best_score, "b": worst_score},
        "differences": differences,
        "tradeoffs": tradeoffs,
    }


# ------------------------------------------------------------------
# Main driver

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # CAR-11
    print("== CAR-11 ==")
    res11 = car_11(tbl)
    best = res11["recommended"]
    print(f"Recommended vehicle: row {best}")
    print("Similar vehicles:")
    for nb in res11["neighbors"]:
        print(f"  Row {nb['row']} with goals {nb['values']}")
    print(f"ANSWER CAR-11 {json.dumps(res11)}")

    # CAR-12
    print("\n== CAR-12 ==")
    res12 = car_12(tbl)
    a = res12["a"]
    b = res12["b"]
    pref = res12["preferred"]
    print(f"Comparing row {a} (preferred) vs row {b}")
    print(f"  Scores: row {a} = {res12['scores']['a']:.4f}, row {b} = {res12['scores']['b']:.4f}")
    print("Differences:")
    for d in res12["differences"]:
        print(f"  {d['feature']}: row {a} = {d['a']}, row {b} = {d['b']}")
    print("Trade‑offs where the non‑preferred row is better:")
    print("  " + ", ".join(res12["tradeoffs"]))
    print(f"ANSWER CAR-12 {json.dumps(res12)}")


if __name__ == "__main__":
    main()

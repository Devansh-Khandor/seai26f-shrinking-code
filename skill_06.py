#!/usr/bin/env python3
"""
skill_06.py: Implements CAR-11 and CAR-12 for used‑car dealership data.

Usage:
    python skill_06.py <path_to_csv>
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple, Any, Optional

# ------------------------------------------------------------------
# CSV reading utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for row in reader:
            conv_row: List[Any] = []
            for v in row:
                if v == "?":
                    conv_row.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv_row.append(float(v))
                        else:
                            conv_row.append(int(v))
                    except ValueError:
                        conv_row.append(v)
            rows.append(conv_row)
    return header, rows

# ------------------------------------------------------------------
# Helpers for goals
# ------------------------------------------------------------------
def goal_indices(header: List[str]) -> List[int]:
    """Indices of goal columns (ending with '+' or '-')."""
    return [i for i, h in enumerate(header) if h.endswith(("+", "-"))]

def goal_is_max(header: List[str]) -> Dict[int, bool]:
    """Map goal index -> True if maximize."""
    return {i: header[i].endswith("+") for i in goal_indices(header)}

def compute_goal_bounds(rows: List[List[Any]], gidx: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    mins, maxs = {}, {}
    for idx in gidx:
        vals = [r[idx] for r in rows if r[idx] is not None]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)
        else:
            mins[idx] = maxs[idx] = None
    return mins, maxs

def normalize_goal_value(v: Any, mn: float, mx: float, maximize: bool) -> float:
    """Return normalized value in [0,1] for a single goal."""
    if mn is None or mx is None or mx == mn:
        return 0.5
    if maximize:
        return (v - mn) / (mx - mn)
    else:
        return (mx - v) / (mx - mn)

def score_row(row: List[Any], gidx: List[int], mins: Dict[int, float],
              maxs: Dict[int, float], ismax: Dict[int, bool]) -> float:
    """Sum of normalized goal values, skip missing goals."""
    score = 0.0
    for idx in gidx:
        val = row[idx]
        if val is None:
            continue
        score += normalize_goal_value(val, mins[idx], maxs[idx], ismax[idx])
    return score

# ------------------------------------------------------------------
# Helpers for distance / normalization of other columns
# ------------------------------------------------------------------
def non_goal_indices(header: List[str]) -> List[int]:
    """Indices of columns that are not goals and not X."""
    return [
        i
        for i, h in enumerate(header)
        if not h.endswith(("X", "+", "-")) and not h.endswith("X")
    ]

def compute_column_bounds(rows: List[List[Any]], nidx: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    mins, maxs = {}, {}
    for idx in nidx:
        vals = [r[idx] for r in rows if r[idx] is not None and isinstance(r[idx], (int, float))]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)
        else:
            mins[idx] = maxs[idx] = None
    return mins, maxs

def normalize_numeric_value(v: Any, mn: float, mx: float) -> float:
    if mn is None or mx is None or mx == mn:
        return 0.5
    return (v - mn) / (mx - mn)

def distance_rows(r1: List[Any], r2: List[Any],
                  nidx: List[int],
                  mins: Dict[int, float],
                  maxs: Dict[int, float]) -> float:
    """Euclidean distance over numeric columns + Hamming for symbols.
    Ignore missing values; if no shared dimension, return large number.
    """
    sq = 0.0
    used = 0
    for idx in nidx:
        v1 = r1[idx]
        v2 = r2[idx]
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            n1 = normalize_numeric_value(v1, mins[idx], maxs[idx])
            n2 = normalize_numeric_value(v2, mins[idx], maxs[idx])
            diff = n1 - n2
            sq += diff * diff
        else:
            diff = 0.0 if v1 == v2 else 1.0
            sq += diff
        used += 1
    if used == 0:
        return float("inf")
    return math.sqrt(sq)

# ------------------------------------------------------------------
# CAR-11: recommendation and similar vehicles
# ------------------------------------------------------------------
def recommend_and_neighbors(header: List[str], rows: List[List[Any]]) -> Tuple[int, List[Dict[str, Any]]]:
    gidx = goal_indices(header)
    ismax = goal_is_max(header)
    mins_g, maxs_g = compute_goal_bounds(rows, gidx)

    # compute scores for all rows
    scores = {}
    for i, r in enumerate(rows, start=1):
        scores[i] = score_row(r, gidx, mins_g, maxs_g, ismax)

    # best row
    best_row = max(scores.items(), key=lambda kv: kv[1])[0]

    # distance to all others
    nidx = non_goal_indices(header)
    mins_n, maxs_n = compute_column_bounds(rows, nidx)

    distances: List[Tuple[float, int]] = []
    for i, r in enumerate(rows, start=1):
        if i == best_row:
            continue
        d = distance_rows(r, rows[best_row-1], nidx, mins_n, maxs_n)
        distances.append((d, i))
    # pick 3 smallest distances
    distances.sort(key=lambda kv: (kv[0], kv[1]))
    neighbor_rows = [i for _, i in distances[:3]]

    neighbors: List[Dict[str, Any]] = []
    for nr in neighbor_rows:
        neighbors.append({
            "row": nr,
            "values": {header[j]: rows[nr-1][j] for j in range(len(header))}
        })

    return best_row, neighbors

# ------------------------------------------------------------------
# CAR-12: explain trade‑offs between two vehicles
# ------------------------------------------------------------------
def explain_tradeoffs(header: List[str], rows: List[List[Any]], a: int, b: int) -> Dict[str, Any]:
    gidx = goal_indices(header)
    ismax = goal_is_max(header)
    mins_g, maxs_g = compute_goal_bounds(rows, gidx)

    scores_a = score_row(rows[a-1], gidx, mins_g, maxs_g, ismax)
    scores_b = score_row(rows[b-1], gidx, mins_g, maxs_g, ismax)

    preferred = a if scores_a >= scores_b else b

    # differences over non-goal columns
    nidx = non_goal_indices(header)
    diffs: List[Dict[str, Any]] = []

    for idx in nidx:
        va = rows[a-1][idx]
        vb = rows[b-1][idx]
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)):
            diff_val = abs(va - vb)
            if diff_val > 0:
                diffs.append({"feature": header[idx], "a": va, "b": vb, "diff": diff_val})
        else:
            if va != vb:
                diffs.append({"feature": header[idx], "a": va, "b": vb, "diff": 1})

    # keep only top 5 by diff magnitude
    diffs.sort(key=lambda d: -d["diff"])
    diffs = diffs[:5] if diffs else []

    # tradeoffs: goal columns where non‑preferred has better value
    tradeoffs: List[str] = []
    for idx in gidx:
        va = rows[a-1][idx]
        vb = rows[b-1][idx]
        if va is None or vb is None:
            continue
        # determine if non-preferred has better normalized value
        max_pre = ismax[idx]
        norm_a = normalize_goal_value(va, mins_g[idx], maxs_g[idx], max_pre)
        norm_b = normalize_goal_value(vb, mins_g[idx], maxs_g[idx], max_pre)
        if preferred == a and norm_b > norm_a:
            tradeoffs.append(header[idx])
        if preferred == b and norm_a > norm_b:
            tradeoffs.append(header[idx])

    return {
        "a": a,
        "b": b,
        "preferred": preferred,
        "scores": {"a": round(scores_a, 4), "b": round(scores_b, 4)},
        "differences": [{"feature": d["feature"], "a": d["a"], "b": d["b"]} for d in diffs],
        "tradeoffs": tradeoffs
    }

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # CAR-11
    print("== CAR-11 ==")
    rec_row, neighbors = recommend_and_neighbors(header, rows)
    score_rec = score_row(rows[rec_row-1], goal_indices(header),
                          *compute_goal_bounds(rows, goal_indices(header)),
                          goal_is_max(header))
    print(f"Recommended vehicle: row {rec_row} (score {round(score_rec,4)})")
    print("Top 3 similar vehicles:")
    for n in neighbors:
        dist = distance_rows(rows[rec_row-1], n["values"], non_goal_indices(header),
                             *compute_column_bounds(rows, non_goal_indices(header)))
        print(f"  Row {n['row']}: distance {round(dist,4)}")
    print(f"ANSWER CAR-11 {json.dumps({'recommended': rec_row, 'neighbors': neighbors}, separators=(',', ':'))}")

    # CAR-12
    print("\n== CAR-12 ==")
    # choose best and worst rows
    gidx = goal_indices(header)
    ismax = goal_is_max(header)
    mins_g, maxs_g = compute_goal_bounds(rows, gidx)
    scores = {i+1: score_row(r, gidx, mins_g, maxs_g, ismax) for i, r in enumerate(rows)}
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    worst = min(scores.items(), key=lambda kv: kv[1])[0]
    explanation = explain_tradeoffs(header, rows, best, worst)
    print(f"Comparing Row {best} (score {round(explanation['scores']['a'],4)}) vs Row {worst} (score {round(explanation['scores']['b'],4)})")
    print(f"Preferred: Row {explanation['preferred']}")
    if explanation['differences']:
        print("Differences:")
        for d in explanation['differences'][:3]:
            print(f"  {d['feature']}: Row{best}={d['a']}, Row{worst}={d['b']}")
    else:
        print("No significant differences found.")
    if explanation['tradeoffs']:
        print("Tradeoffs where non‑preferred is better:", explanation['tradeoffs'])
    else:
        print("No tradeoffs: non‑preferred not better on any goal.")
    print(f"ANSWER CAR-12 {json.dumps(explanation, separators=(',', ':'))}")

if __name__ == "__main__":
    main()

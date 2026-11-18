#!/usr/bin/env python3
"""
skill_09.py: Implements CAR-17 and CAR-18 for used‑car dealership data.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import json
import math
import sys
from typing import List, Any, Dict, Tuple

# --------------------- CSV utilities --------------------- #
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    rows: List[List[Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            conv: List[Any] = []
            for v in line:
                if v == "?":
                    conv.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv.append(float(v))
                        else:
                            conv.append(int(v))
                    except ValueError:
                        conv.append(v)
            rows.append(conv)
    return header, rows

# --------------------- helper functions --------------------- #
def goal_indices(header: List[str]) -> List[int]:
    return [i for i, h in enumerate(header) if h.endswith(("+", "-"))]

def input_indices(header: List[str]) -> List[int]:
    return [i for i, h in enumerate(header) if not h.endswith(("X", "+", "-"))]

def numeric_columns(header: List[str], indices: List[int]) -> List[int]:
    return [i for i in indices if isinstance(header[i], str) and header[i] not in ("",)]

def compute_min_max(rows: List[List[Any]], cols: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    mins, maxs = {}, {}
    for i in cols:
        vals = [r[i] for r in rows if r[i] is not None and isinstance(r[i], (int, float))]
        if vals:
            mins[i] = min(vals)
            maxs[i] = max(vals)
        else:
            mins[i] = maxs[i] = None
    return mins, maxs

def normalize_value(v: Any, mn: float, mx: float) -> float:
    if v is None:
        return None
    if mx == mn:
        return 0.0
    return (v - mn) / (mx - mn)

# distance between two rows over input columns
def distance_rows(r1: List[Any], r2: List[Any],
                  num_cols: List[int], cat_cols: List[int],
                  mins: Dict[int, float], maxs: Dict[int, float]) -> float:
    sq = 0.0
    # numeric
    for i in num_cols:
        v1 = r1[i]
        v2 = r2[i]
        if v1 is None or v2 is None:
            diff = 1.0
        else:
            n1 = normalize_value(v1, mins[i], maxs[i])
            n2 = normalize_value(v2, mins[i], maxs[i])
            diff = abs(n1 - n2)
        sq += diff * diff
    # categorical
    for i in cat_cols:
        v1 = r1[i]
        v2 = r2[i]
        if v1 is None or v2 is None:
            diff = 1.0
        else:
            diff = 0.0 if v1 == v2 else 1.0
        sq += diff
    return math.sqrt(sq)

# find nearest neighbor index for each row
def nearest_neighbors(rows: List[List[Any]],
                      num_cols: List[int], cat_cols: List[int],
                      mins: Dict[int, float], maxs: Dict[int, float]) -> List[int]:
    n = len(rows)
    neigh = []
    for idx in range(n):
        best = None
        bestdist = float("inf")
        for j in range(n):
            if j == idx:
                continue
            d = distance_rows(rows[idx], rows[j], num_cols, cat_cols, mins, maxs)
            if d < bestdist:
                bestdist = d
                best = j
        neigh.append(best)
    return neigh

# --------------------- CAR-17 --------------------- #
def car17(header: List[str], rows: List[List[Any]]) -> Tuple[List[int], Dict[int, float]]:
    inp = input_indices(header)
    num_cols = [i for i in inp if isinstance(rows[0][i], (int, float))]
    cat_cols = [i for i in inp if not isinstance(rows[0][i], (int, float))]

    mins, maxs = compute_min_max(rows, num_cols)

    neigh = nearest_neighbors(rows, num_cols, cat_cols, mins, maxs)
    scores = {}
    for i, nidx in enumerate(neigh):
        if nidx is None:
            continue
        d = distance_rows(rows[i], rows[nidx], num_cols, cat_cols, mins, maxs)
        scores[i + 1] = d  # 1‑based

    # select up to 40 rows with highest score
    sorted_rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    flagged = [r for r, _ in sorted_rows[:40]]
    flagged_scores = {r: scores[r] for r in flagged}
    return flagged, flagged_scores

# --------------------- CAR-18 --------------------- #
def car18(header: List[str], rows: List[List[Any]]) -> List[Dict[str, Any]]:
    goal = goal_indices(header)
    inp = input_indices(header)
    num_cols = [i for i in inp if isinstance(rows[0][i], (int, float))]
    cat_cols = [i for i in inp if not isinstance(rows[0][i], (int, float))]

    mins, maxs = compute_min_max(rows, num_cols)

    neigh = nearest_neighbors(rows, num_cols, cat_cols, mins, maxs)

    # gather diffs per goal
    diffs_per_goal: Dict[int, List[float]] = {g: [] for g in goal}
    records = []
    for idx, nidx in enumerate(neigh):
        if nidx is None:
            continue
        for g in goal:
            actual = rows[idx][g]
            expected = rows[nidx][g]
            if actual is None or expected is None:
                continue
            diff = abs(actual - expected)
            diffs_per_goal[g].append(diff)
            records.append((idx + 1, g, actual, expected, diff))

    flagged_entries = []
    for g in goal:
        diffs = diffs_per_goal[g]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        std = math.sqrt(var)
        threshold = mean + 2 * std
        # if std is zero, threshold equals mean
        for (rownum, col, act, exp, diff) in records:
            if col != g:
                continue
            if diff > threshold:
                flagged_entries.append({
                    "row": rownum,
                    "column": header[col],
                    "actual": act,
                    "expected": exp
                })

    # keep at most 40
    flagged_entries = sorted(flagged_entries, key=lambda e: e["row"])
    return flagged_entries[:40]

# --------------------- main --------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # CAR-17
    print("== CAR-17 ==")
    flagged, scores = car17(header, rows)
    # output at least 3 lines
    for i, r in enumerate(flagged[:3], 1):
        print(f"Flagged row {r} unusualness score: {scores[r]:.4f}")
    print(f"ANSWER CAR-17 {json.dumps({'flagged': flagged, 'scores': {str(k): round(v, 4) for k, v in scores.items()}})}")

    # CAR-18
    print("\n== CAR-18 ==")
    flagged18 = car18(header, rows)
    for i, e in enumerate(flagged18[:3], 1):
        print(f"Row {e['row']} flagged on {e['column']} (actual={e['actual']}, expected={e['expected']})")
    print(f"ANSWER CAR-18 {json.dumps({'flagged': flagged18})}")

if __name__ == "__main__":
    main()

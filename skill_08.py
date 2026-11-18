#!/usr/bin/env python3
"""
skill_08.py

Implements CAR-15 and CAR-16 for the auto93 dataset.

CAR-15:
    For the vehicle with the lowest overall performance score,
    find alternative vehicles in the historical inventory that are
    most similar on input features and have a better score.
    Output the changes in input features that differentiate them.

CAR-16:
    For the first vehicle that does not meet all preferred
    performance targets (mean thresholds for each goal),
    find the alternative that meets more targets and differs in
    as few input columns as possible.

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


def input_columns(rows: List[Dict[str, Any]]) -> Set[str]:
    """Return set of column names that are inputs (neither goal nor ending with 'X')."""
    if not rows:
        return set()
    goals = goal_columns(rows)
    return {k for k in rows[0] if k not in goals and not k.endswith('X')}


# ------------------------------------------------------------------
# Normalisation utilities
# ------------------------------------------------------------------
def min_max_for_cols(rows: List[Dict[str, Any]], cols: Set[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for col in cols:
        vals = [r[col] for r in rows if isinstance(r[col], float)]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs


def normalized_value(val: float, mn: float, mx: float, minimize: bool) -> float:
    if mx == mn:
        return 0.0
    return (mx - val) / (mx - mn) if minimize else (val - mn) / (mx - mn)


def overall_score(row: Dict[str, Any],
                  goals: Set[str],
                  mins: Dict[str, float],
                  maxs: Dict[str, float]) -> float | None:
    """Compute overall performance score: sum of normalized values of all goals."""
    for g in goals:
        if not isinstance(row[g], float):
            return None
    score = 0.0
    for g in goals:
        mn, mx = mins[g], maxs[g]
        val = row[g]
        score += normalized_value(val, mn, mx, minimize=g.endswith('-'))
    return score


# ------------------------------------------------------------------
# CAR-15
# ------------------------------------------------------------------
def car_15(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    goals = goal_columns(rows)
    inputs = input_columns(rows)

    # compute mins/maxs for goals and inputs
    goal_mins, goal_maxs = min_max_for_cols(rows, goals)
    input_mins, input_maxs = min_max_for_cols(rows, inputs)

    # compute scores for all rows
    scored = []
    for idx, r in enumerate(rows, start=1):
        sc = overall_score(r, goals, goal_mins, goal_maxs)
        if sc is not None:
            scored.append((idx, sc))

    if not scored:
        return {"row": None, "alternatives": []}

    # find poorest row
    poor_idx, poor_score = min(scored, key=lambda x: (x[1], x[0]))
    poor_row = rows[poor_idx - 1]

    # compute normalized input vectors for all rows
    norm_vectors: Dict[int, List[float]] = {}
    for idx, r in enumerate(rows, start=1):
        # skip rows with missing input
        if any(r[c] is None for c in inputs):
            continue
        vec = []
        for c in inputs:
            val = r[c]
            vec.append(normalized_value(val, input_mins[c], input_maxs[c], minimize=False))  # all inputs treated as to be maximised
        norm_vectors[idx] = vec

    # distance to poor row
    if poor_idx not in norm_vectors:
        # if poor row has missing input, cannot find similar ones
        alternatives = []
    else:
        poor_vec = norm_vectors[poor_idx]
        distances: List[Tuple[int, float]] = []
        for idx, vec in norm_vectors.items():
            if idx == poor_idx:
                continue
            # Euclidean distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(poor_vec, vec)))
            distances.append((idx, dist))

        # sort by distance ascending
        distances.sort(key=lambda x: x[1])

        # select up to 3 closest rows with higher score
        alternatives = []
        for alt_idx, _ in distances:
            alt_score = overall_score(rows[alt_idx - 1], goals, goal_mins, goal_maxs)
            if alt_score is None:
                continue
            if alt_score > poor_score:
                alt_row = rows[alt_idx - 1]
                changes = {}
                for c in inputs:
                    pv = poor_row[c]
                    av = alt_row[c]
                    if pv != av:
                        changes[c] = [pv, av]
                alternatives.append({"row": alt_idx, "changes": changes})
                if len(alternatives) >= 3:
                    break

    return {
        "row": poor_idx,
        "alternatives": alternatives
    }


# ------------------------------------------------------------------
# CAR-16
# ------------------------------------------------------------------
def car_16(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    goals = goal_columns(rows)
    inputs = input_columns(rows)

    # thresholds: mean of each goal column
    thresholds = {}
    for g in goals:
        vals = [r[g] for r in rows if isinstance(r[g], float)]
        thresholds[g] = sum(vals) / len(vals) if vals else 0.0

    # function to check if a row meets all targets
    def meets_targets(r: Dict[str, Any]) -> int:
        count = 0
        for g in goals:
            if not isinstance(r[g], float):
                continue
            val = r[g]
            thresh = thresholds[g]
            if g.endswith('-'):
                if val <= thresh:
                    count += 1
            else:
                if val >= thresh:
                    count += 1
        return count

    # find first row that does not meet all targets
    original_idx = None
    original_meets = None
    for idx, r in enumerate(rows, start=1):
        meets = meets_targets(r)
        if meets < len(goals):
            original_idx = idx
            original_meets = meets
            break

    if original_idx is None:
        return {"row": None, "targets": thresholds, "alternative": None, "changed": []}

    orig_row = rows[original_idx - 1]

    # evaluate all alternative rows
    alternatives: List[Tuple[int, int, int, List[str]]] = []  # (idx, meets, diff_count, diff_cols)
    for idx, r in enumerate(rows, start=1):
        if idx == original_idx:
            continue
        meets = meets_targets(r)
        if meets <= original_meets:
            continue  # need more targets met

        # compute differences in input columns
        diff_cols = []
        for c in inputs:
            ov = orig_row[c]
            av = r[c]
            if ov != av:
                diff_cols.append(c)
        diff_count = len(diff_cols)
        alternatives.append((idx, meets, diff_count, diff_cols))

    if not alternatives:
        return {"row": original_idx, "targets": thresholds, "alternative": None, "changed": []}

    # choose alternative with most targets met, fewest differences, lowest row number
    alternatives.sort(key=lambda x: (-x[1], x[2], x[0]))
    best_idx, best_meets, best_diff, best_cols = alternatives[0]

    return {
        "row": original_idx,
        "targets": thresholds,
        "alternative": best_idx,
        "changed": best_cols
    }


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)

    # CAR-15
    print("== CAR-15 ==")
    ans15 = car_15(rows)
    poor_row = ans15["row"]
    poor_score = None
    if poor_row is not None:
        # compute score
        goals = goal_columns(rows)
        goal_mins, goal_maxs = min_max_for_cols(rows, goals)
        poor_score = overall_score(rows[poor_row - 1], goals, goal_mins, goal_maxs)
    print(f"Poor vehicle row: {poor_row} score: {poor_score:.4f}" if poor_score is not None else f"Poor vehicle row: {poor_row}")
    print(f"Alternatives found: {len(ans15['alternatives'])}")
    for alt in ans15['alternatives']:
        print(f"Alternative row {alt['row']} changes: {json.dumps(alt['changes'])}")
    print(f"ANSWER CAR-15 {json.dumps(ans15, ensure_ascii=False)}")

    # CAR-16
    print("\n== CAR-16 ==")
    ans16 = car_16(rows)
    orig_row = ans16["row"]
    print(f"Original row: {orig_row}")
    print(f"Targets (mean thresholds): {json.dumps(ans16['targets'])}")
    if ans16["alternative"] is None:
        print("No suitable alternative found.")
    else:
        print(f"Chosen alternative: row {ans16['alternative']} meets more targets, changed columns: {ans16['changed']}")
    print(f"ANSWER CAR-16 {json.dumps(ans16, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_09.py

Implements CAR-17 and CAR-18 for the auto93 dataset.

CAR-17:
    Flag vehicle records whose combinations of characteristics are unusually
    different from the rest of our inventory, based on the Euclidean distance
    of their input columns from the mean of all inputs.

CAR-18:
    Identify vehicles whose reported fuel efficiency (Mpg+) or weight
    (Lbs-) differs substantially from what similar vehicles would suggest.
    Similar vehicles are defined by the 5 nearest neighbours in the
    input feature space.  A record is flagged if the absolute difference
    between the actual and the expected value exceeds two standard
    deviations of the target column.

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
    """Read CSV into list of dicts, converting numbers to float and
    converting '?' or '' to None."""
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
    """Columns that end in '+' or '-'."""
    if not rows:
        return set()
    return {k for k in rows[0] if k.endswith('-') or k.endswith('+')}


def input_columns(rows: List[Dict[str, Any]]) -> Set[str]:
    """Columns that are neither goals nor end in 'X'."""
    if not rows:
        return set()
    goals = goal_columns(rows)
    return {k for k in rows[0] if k not in goals and not k.endswith('X')}


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


# ------------------------------------------------------------------
# CAR-17 helpers
# ------------------------------------------------------------------
def compute_input_stats(rows: List[Dict[str, Any]],
                        inputs: Set[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return mean and std for each input column."""
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for col in inputs:
        vals = [r[col] for r in rows if isinstance(r[col], float)]
        if vals:
            means[col] = mean(vals)
            stds[col] = std(vals) if std(vals) > 0 else 1.0
        else:
            means[col] = stds[col] = 0.0
    return means, stds


def compute_unusualness(rows: List[Dict[str, Any]],
                        inputs: Set[str],
                        means: Dict[str, float],
                        stds: Dict[str, float]) -> List[Tuple[int, float]]:
    """Return list of (row_number, unusualness_score)."""
    scores: List[Tuple[int, float]] = []
    for idx, r in enumerate(rows, start=1):
        # skip rows with missing input
        if any(r[col] is None for col in inputs):
            continue
        zsum = 0.0
        for col in inputs:
            z = (r[col] - means[col]) / stds[col]
            zsum += z * z
        score = math.sqrt(zsum)
        scores.append((idx, score))
    return scores


# ------------------------------------------------------------------
# CAR-18 helpers
# ------------------------------------------------------------------
def build_std_matrix(rows: List[Dict[str, Any]],
                     inputs: Set[str]) -> List[List[float]]:
    """Return list of rows (as lists) of standardized input values."""
    means, stds = compute_input_stats(rows, inputs)
    matrix: List[List[float]] = []
    for r in rows:
        if any(r[col] is None for col in inputs):
            matrix.append(None)
            continue
        vec = [(r[col] - means[col]) / stds[col] for col in inputs]
        matrix.append(vec)
    return matrix


def k_nearest_neighbors(matrix: List[List[float]],
                        target_idx: int,
                        k: int = 5) -> List[int]:
    """Return indices (0‑based) of k nearest neighbours of target_idx."""
    if matrix[target_idx] is None:
        return []
    distances: List[Tuple[int, float]] = []
    for idx, vec in enumerate(matrix):
        if idx == target_idx or vec is None:
            continue
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(matrix[target_idx], vec)))
        distances.append((idx, dist))
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances[:k]]


def expected_target(rows: List[Dict[str, Any]],
                    matrix: List[List[float]],
                    target_col: str,
                    neighbors: List[int]) -> float:
    """Mean of target_col for given neighbours."""
    vals = [rows[n][target_col] for n in neighbors if rows[n][target_col] is not None]
    if not vals:
        return 0.0
    return mean(vals)


# ------------------------------------------------------------------
# CAR-17 main
# ------------------------------------------------------------------
def car_17(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    inputs = input_columns(rows)
    means, stds = compute_input_stats(rows, inputs)
    scores = compute_unusualness(rows, inputs, means, stds)
    # Sort by score descending
    scores.sort(key=lambda x: (-x[1], x[0]))
    flagged = [idx for idx, _ in scores[:10]]  # top 10
    flagged_scores = {str(idx): round(score, 4) for idx, score in scores if idx in flagged}
    return {"flagged": flagged, "scores": flagged_scores}


# ------------------------------------------------------------------
# CAR-18 main
# ------------------------------------------------------------------
def car_18(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    inputs = input_columns(rows)
    matrix = build_std_matrix(rows, inputs)
    target_cols = ["Mpg+", "Lbs-"]
    # global std for thresholds
    global_std = {col: std([r[col] for r in rows if isinstance(r[col], float)]) for col in target_cols}
    flagged_list: List[Dict[str, Any]] = []

    for idx, r in enumerate(rows, start=1):
        # skip rows with missing input or target
        if any(r[col] is None for col in inputs) or r["Mpg+"] is None or r["Lbs-"] is None:
            continue

        neigh = k_nearest_neighbors(matrix, idx - 1, k=5)
        if not neigh:
            continue

        for col in target_cols:
            expected = expected_target(rows, matrix, col, neigh)
            diff = abs(r[col] - expected)
            if diff > 2 * global_std[col]:
                flagged_list.append({
                    "row": idx,
                    "column": col,
                    "actual": round(r[col], 2),
                    "expected": round(expected, 2)
                })
                # break after first flagged to avoid duplicate for same row+col
                # but we may want both columns flagged separately, so not break
                # continue

    # keep up to 40, sorted by abs diff descending
    flagged_list.sort(key=lambda d: -abs(d["actual"] - d["expected"]))
    flagged_list = flagged_list[:40]
    return {"flagged": flagged_list}


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)

    # CAR-17
    print("== CAR-17 ==")
    res17 = car_17(rows)
    for row_num in res17["flagged"]:
        score = res17["scores"][str(row_num)]
        print(f"Row {row_num} unusualness score: {score}")
    answer17 = {"flagged": res17["flagged"], "scores": res17["scores"]}
    print(f"ANSWER CAR-17 {json.dumps(answer17, ensure_ascii=False)}")

    # CAR-18
    print("\n== CAR-18 ==")
    res18 = car_18(rows)
    for entry in res18["flagged"]:
        print(f"Row {entry['row']} column {entry['column']} actual {entry['actual']} expected {entry['expected']}")
    answer18 = {"flagged": res18["flagged"]}
    print(f"ANSWER CAR-18 {json.dumps(answer18, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

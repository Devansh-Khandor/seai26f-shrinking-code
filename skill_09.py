#!/usr/bin/env python3
"""
skill_09.py

Usage: python skill_09.py <csv path>
"""

import json
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return numeric value, or None if missing."""
    return None if v == "?" else v


def minmax(values: List[float]) -> Tuple[float, float]:
    """Return (min, max).  If all same, return (m, m+1)."""
    if not values:
        return 0.0, 1.0
    mn, mx = min(values), max(values)
    if mn == mx:
        mx = mn + 1.0
    return mn, mx


def normalize(v: float, mn: float, mx: float) -> float:
    """Map value to 0..1."""
    return 0.0 if mx == mn else (v - mn) / (mx - mn)


# ----------------------------------------------------------------------
# CAR-17: flag unusual combinations of input columns
# ----------------------------------------------------------------------
def car_17(rows: List[List[Any]], header: List[str]) -> Dict[str, Any]:
    # Input columns: not X, not goals
    input_names = [
        name
        for name in header
        if not name.endswith("X") and name[-1] not in "+-"
    ]
    input_indices = [header.index(name) for name in input_names]

    # Compute min/max for each input column
    mins_maxs: Dict[str, Tuple[float, float]] = {}
    for name, idx in zip(input_names, input_indices):
        vals = [numeric(r[idx]) for r in rows if numeric(r[idx]) is not None]
        mins_maxs[name] = minmax(vals)

    # Compute centroid in normalized space
    centroids = []
    for name in input_names:
        mn, mx = mins_maxs[name]
        # compute mean of normalized values
        vals = [
            normalize(v, mn, mx) for v in [numeric(r[header.index(name)]) for r in rows]
            if v is not None
        ]
        if vals:
            centroids.append(sum(vals) / len(vals))
        else:
            centroids.append(0.0)

    # Compute unusualness score (distance to centroid)
    scores: Dict[int, float] = {}
    for row_idx, row in enumerate(rows, start=1):  # 1-based
        vec = []
        for name, idx in zip(input_names, input_indices):
            v = numeric(row[idx])
            mn, mx = mins_maxs[name]
            if v is None:
                norm = 0.0
            else:
                norm = normalize(v, mn, mx)
            vec.append(norm)
        # Euclidean distance to centroid
        dist = (sum((vec[i] - centroids[i]) ** 2 for i in range(len(vec))) ** 0.5)
        scores[row_idx] = dist

    # Determine threshold: mean + std
    all_scores = list(scores.values())
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    std_score = (
        (sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)) ** 0.5
        if all_scores
        else 0.0
    )
    threshold = mean_score + std_score

    # Flag rows above threshold
    flagged = [row for row, score in scores.items() if score > threshold]

    # If none flagged, flag the single most unusual
    if not flagged:
        max_row = max(scores, key=scores.get)
        flagged = [max_row]

    answer = {"flagged": flagged, "scores": {str(k): v for k, v in scores.items()}}
    return answer


# ----------------------------------------------------------------------
# CAR-18: flag fuel efficiency or weight discrepancies
# ----------------------------------------------------------------------
def car_18(rows: List[List[Any]], header: List[str]) -> Dict[str, Any]:
    # Identify input columns (non-X, non-goals)
    input_names = [
        name
        for name in header
        if not name.endswith("X") and name[-1] not in "+-"
    ]
    input_indices = [header.index(name) for name in input_names]

    # Target columns
    targets = ["Mpg+", "Lbs-"]

    # Precompute numeric values for all rows
    numeric_rows = [[numeric(v) for v in row] for row in rows]
    n_rows = len(rows)

    # Compute differences per row & target
    diffs: List[Tuple[int, str, float, float]] = []  # row, column, actual, expected
    k = 3

    for row_idx, row_vals in enumerate(numeric_rows, start=1):
        # Build distance to all others based on input columns
        distances = []
        for other_idx, other_vals in enumerate(numeric_rows, start=1):
            if other_idx == row_idx:
                continue
            # compute distance in input space
            d = 0.0
            for idx in input_indices:
                a = row_vals[idx]
                b = other_vals[idx]
                if a is None or b is None:
                    d += 1.0  # treat missing as maximal distance
                else:
                    d += abs(a - b)
            distances.append((d, other_idx))
        distances.sort(key=lambda x: x[0])

        # For each target
        for target in targets:
            tgt_idx = header.index(target)
            actual = numeric_rows[row_idx - 1][tgt_idx]
            if actual is None:
                continue

            # Find neighbors with non-missing target
            neighbor_vals = []
            for _, other_idx in distances:
                val = numeric_rows[other_idx - 1][tgt_idx]
                if val is not None:
                    neighbor_vals.append(val)
                if len(neighbor_vals) >= k:
                    break
            if not neighbor_vals:
                continue
            expected = sum(neighbor_vals) / len(neighbor_vals)
            diff = abs(actual - expected)
            diffs.append((row_idx, target, actual, expected))

    # Determine threshold over all diffs
    all_diff_values = [abs(d[2] - d[3]) for d in diffs]
    mean_diff = sum(all_diff_values) / len(all_diff_values) if all_diff_values else 0.0
    std_diff = (
        (sum((d - mean_diff) ** 2 for d in all_diff_values) / len(all_diff_values))
        ** 0.5
        if all_diff_values
        else 0.0
    )
    threshold = mean_diff + std_diff

    flagged = [
        {"row": d[0], "column": d[1], "actual": d[2], "expected": d[3]}
        for d in diffs
        if abs(d[2] - d[3]) > threshold
    ]

    # If none flagged, flag the largest diff
    if not flagged and diffs:
        best = max(diffs, key=lambda d: abs(d[2] - d[3]))
        flagged = [
            {"row": best[0], "column": best[1], "actual": best[2], "expected": best[3]}
        ]

    answer = {"flagged": flagged}
    return answer


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows = list(rows_iter)

    # CAR-17
    print("== CAR-17 ==")
    res17 = car_17(data_rows, header)
    # Simple human readable output
    print(f"Flagged rows: {res17['flagged']}")
    # Show first few scores
    scores_sample = {k: res17["scores"][k] for k in list(res17["scores"])[:5]}
    print(f"Score sample: {scores_sample}")
    print(f"ANSWER CAR-17 {json.dumps(res17, separators=(',', ':'))}")

    # CAR-18
    print("\n== CAR-18 ==")
    res18 = car_18(data_rows, header)
    # Human readable
    print(f"Flagged entries: {res18['flagged'][:5]}")
    print(f"ANSWER CAR-18 {json.dumps(res18, separators=(',', ':'))}")


if __name__ == "__main__":
    main()

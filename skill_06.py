#!/usr/bin/env python3
"""
Skill 06 – Recommendation explanations.

Usage:
    python skill_06.py <csv_path>
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple, Optional


# --------------------------------------------------------------------------- #
# Helpers – copied from earlier skills
# --------------------------------------------------------------------------- #
def parse_float(s: str) -> Optional[float]:
    """Parse a string into float, return None for '?' or empty."""
    s = s.strip()
    if s == "" or s == "?":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Return mean, median, min, max for a list of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    # median
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return {
        "mean": mean,
        "median": median,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def normalize(value: float, vmin: float, vmax: float, maximize: bool) -> float:
    """Return 0‑1 score for a value. If maximize is False, smaller is better."""
    if vmax == vmin:
        return 1.0
    if maximize:
        return (value - vmin) / (vmax - vmin)
    else:
        return (vmax - value) / (vmax - vmin)


# --------------------------------------------------------------------------- #
# Overall score helper – identical to skill_05
# --------------------------------------------------------------------------- #
def compute_overall_scores(
    data: List[Dict[str, Optional[float]]],
    goal_cols: List[str],
    goal_directions: Dict[str, bool],
) -> List[Tuple[int, float]]:
    """Return list of (row_number, score) for rows with all goals present."""
    # Pre‑compute min/max per goal
    min_vals: Dict[str, float] = {}
    max_vals: Dict[str, float] = {}
    for col in goal_cols:
        vals = [row[col] for row in data if row[col] is not None]
        if vals:
            min_vals[col] = min(vals)
            max_vals[col] = max(vals)
        else:
            min_vals[col] = max_vals[col] = 0.0

    scores: List[Tuple[int, float]] = []
    for idx, row in enumerate(data, start=1):
        if any(row[col] is None for col in goal_cols):
            continue
        vals = [
            normalize(row[col], min_vals[col], max_vals[col], goal_directions[col])
            for col in goal_cols
        ]
        score = sum(vals) / len(vals)
        scores.append((idx, round(score, 6)))
    return scores


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Read data
    data: List[Dict[str, Optional[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {"Lbs-": False, "Acc+": True, "Mpg+": True}

    # Compute overall scores for all rows with complete goals
    overall_scores = compute_overall_scores(data, goal_cols, goal_directions)
    if not overall_scores:
        print("== CAR-11 ==")
        print("No complete rows to evaluate.")
        print("ANSWER CAR-11 {}")
        print("\n== CAR-12 ==")
        print("No complete rows to evaluate.")
        print("ANSWER CAR-12 {}")
        return

    # Sort scores descending for recommendation
    overall_scores.sort(key=lambda x: x[1], reverse=True)

    # Recommended vehicle (top row)
    rec_row_num, rec_score = overall_scores[0]
    rec_row = data[rec_row_num - 1]

    # Normalize values for all scored rows (reuse min/max from overall_scores)
    min_vals = {}
    max_vals = {}
    for col in goal_cols:
        vals = [row[col] for row in data if row[col] is not None]
        min_vals[col] = min(vals)
        max_vals[col] = max(vals)

    # Pre‑compute normalized vectors
    norm_vectors: Dict[int, List[float]] = {}
    for idx, row in enumerate(data, start=1):
        if any(row[col] is None for col in goal_cols):
            continue
        norm_vectors[idx] = [
            normalize(row[col], min_vals[col], max_vals[col], goal_directions[col])
            for col in goal_cols
        ]

    # ------------------------------------------------------------------ #
    # CAR-11: Recommendation explanation
    # ------------------------------------------------------------------ #
    # Compute distances to all other rows
    distances = []
    for other_idx, vec in norm_vectors.items():
        if other_idx == rec_row_num:
            continue
        # Euclidean distance squared
        dist2 = sum((v1 - v2) ** 2 for v1, v2 in zip(vec, norm_vectors[rec_row_num]))
        distances.append((dist2, other_idx))

    distances.sort(key=lambda x: x[0])
    neighbor_rows = [idx for _, idx in distances[:3]]

    neighbors = []
    for nidx in neighbor_rows:
        nrow = data[nidx - 1]
        neighbors.append(
            {
                "row": nidx,
                "values": {col: nrow[col] for col in goal_cols},
            }
        )

    # CAR-11 output
    print("== CAR-11 ==")
    print(f"Recommended vehicle is row {rec_row_num} with overall score {rec_score:.6f}.")
    print("Similar vehicles:")
    for n in neighbors:
        vals_str = ", ".join(f"{k}={v}" for k, v in n["values"].items())
        print(f"Row {n['row']}: {vals_str}")

    ans_car_11 = json.dumps(
        {
            "recommended": rec_row_num,
            "neighbors": neighbors,
        },
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-11 {ans_car_11}")

    # ------------------------------------------------------------------ #
    # CAR-12: Comparison of two vehicles
    # ------------------------------------------------------------------ #
    # Pick top and bottom scored rows
    low_row_num, low_score = overall_scores[-1]
    low_row = data[low_row_num - 1]

    # Scores dict
    scores_dict = {"a": rec_score, "b": low_score}

    # Differences per goal column
    differences = []
    for col in goal_cols:
        diff_dict = {
            "feature": col,
            "a": rec_row[col],
            "b": low_row[col],
        }
        differences.append(diff_dict)

    # Tradeoffs: columns where low_row is better
    tradeoffs = []
    for col in goal_cols:
        a_val = rec_row[col]
        b_val = low_row[col]
        if goal_directions[col]:  # maximize
            if b_val > a_val:
                tradeoffs.append(col)
        else:  # minimize
            if b_val < a_val:
                tradeoffs.append(col)

    # CAR-12 output
    print("\n== CAR-12 ==")
    print(
        f"Vehicle A (recommended) is row {rec_row_num} (score {rec_score:.6f})."
    )
    print(
        f"Vehicle B is row {low_row_num} (score {low_score:.6f})."
    )
    print("Differences:")
    for d in differences:
        print(
            f"{d['feature']}: a={d['a']}, b={d['b']}"
        )
    print("Trade‑offs where B is better:")
    if tradeoffs:
        print(", ".join(tradeoffs))
    else:
        print("None")

    ans_car_12 = json.dumps(
        {
            "a": rec_row_num,
            "b": low_row_num,
            "preferred": rec_row_num,
            "scores": scores_dict,
            "differences": differences,
            "tradeoffs": tradeoffs,
        },
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-12 {ans_car_12}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

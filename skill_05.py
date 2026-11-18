#!/usr/bin/env python3
"""
Skill 05 – Advanced analysis for used‑car dealership.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import json
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Helpers – reused from skill_01 / skill_03 / skill_04
# --------------------------------------------------------------------------- #
def parse_float(s: str) -> Optional[float]:
    """Parse a string into float, return None for '?' or empty."""
    s = s.strip()
    if s == '' or s == '?':
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
    return {"mean": mean, "median": median, "min": sorted_vals[0], "max": sorted_vals[-1]}


def normalize(value: float, vmin: float, vmax: float, maximize: bool) -> float:
    """Return 0-1 score for a value. If maximize is False, smaller is better."""
    if vmax == vmin:
        return 1.0
    if maximize:
        return (value - vmin) / (vmax - vmin)
    else:
        return (vmax - value) / (vmax - vmin)


# --------------------------------------------------------------------------- #
# Overall score helper
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
        scores.append((idx, score))
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

    # Goal columns
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {"Lbs-": False, "Acc+": True, "Mpg+": True}

    # ------------------------------------------------------------------ #
    # CAR-09: Poor fuel efficiency analysis
    # ------------------------------------------------------------------ #
    mpg_col = "Mpg+"

    # Identify rows with Mpg+ values
    mpg_rows = [(idx + 1, row[mpg_col]) for idx, row in enumerate(data) if row[mpg_col] is not None]
    # Sort ascending, take first 5 as unusually poor
    mpg_rows_sorted = sorted(mpg_rows, key=lambda x: x[1])
    poor_rows = [idx for idx, _ in mpg_rows_sorted[:5]]

    # Correlation ranking for non‑goal, non‑X columns
    # Non‑goal columns = not ending with '+' or '-' or 'X'
    all_columns = list(data[0].keys())
    non_goal_cols = [c for c in all_columns
                     if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))]

    corr_scores: List[Tuple[str, float]] = []
    for col in non_goal_cols:
        xs = []
        ys = []
        for row in data:
            x = row.get(col)
            y = row.get(mpg_col)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            continue
        # Compute Pearson correlation
        corr = np.corrcoef(xs, ys)[0, 1]
        if np.isnan(corr):
            continue
        corr_scores.append((col, abs(corr)))
    # Rank by absolute correlation descending
    corr_scores.sort(key=lambda x: x[1], reverse=True)
    top_features = [col for col, _ in corr_scores[:3]]

    # Print CAR-09 section
    print("== CAR-09 ==")
    print(f"Unusually poor Mpg+ rows: {poor_rows}")
    print(f"Top associated features: {top_features}")

    # Machine‑checked answer
    ans_car_09 = json.dumps(
        {"poor": poor_rows, "ranking": top_features},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-09 {ans_car_09}")

    # ------------------------------------------------------------------ #
    # CAR-10: Common characteristics of bad vs good overall performance
    # ------------------------------------------------------------------ #
    overall_scores = compute_overall_scores(data, goal_cols, goal_directions)
    # Sort scores ascending for bad, descending for good
    overall_scores.sort(key=lambda x: x[1])
    bad_rows = [row_num for row_num, _ in overall_scores[:5]]
    good_rows = [row_num for row_num, _ in overall_scores[-5:]]

    # Compute typical values per feature in bad and good groups
    def typical(values: List[float]) -> float:
        return float(np.mean(values))

    diff_list: List[Tuple[str, float, float, float]] = []  # (col, bad_val, good_val, diff)
    for col in non_goal_cols:
        # bad group
        bad_vals = [data[idx - 1][col] for idx in bad_rows if data[idx - 1][col] is not None]
        good_vals = [data[idx - 1][col] for idx in good_rows if data[idx - 1][col] is not None]
        if not bad_vals or not good_vals:
            continue
        bad_mean = typical(bad_vals)
        good_mean = typical(good_vals)
        diff = abs(bad_mean - good_mean)
        diff_list.append((col, bad_mean, good_mean, diff))

    # Sort by difference descending, take top 3
    diff_list.sort(key=lambda x: x[3], reverse=True)
    differences = [
        {"feature": col,
         "bad": round(bad_val, 6),
         "good": round(good_val, 6)}
        for col, bad_val, good_val, _ in diff_list[:3]
    ]

    # Print CAR-10 section
    print("\n== CAR-10 ==")
    print(f"Bad rows (unfavorable performance): {bad_rows}")
    print(f"Good rows (favorable performance): {good_rows}")
    print("Differences:")
    for d in differences:
        print(f"{d['feature']}: bad={d['bad']}, good={d['good']}")

    # Machine‑checked answer
    ans_car_10 = json.dumps(
        {"bad": bad_rows, "good": good_rows, "differences": differences},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-10 {ans_car_10}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

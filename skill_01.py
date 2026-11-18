#!/usr/bin/env python3
"""
Skill 01 – Inventory summary and top vehicles.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import json
import sys
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Helpers
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
    return {
        "mean": mean,
        "median": median,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def normalize(value: float, vmin: float, vmax: float, maximize: bool) -> float:
    """Return 0-1 score for a value. If maximize is False, smaller is better."""
    if vmax == vmin:
        return 1.0  # all equal
    if maximize:
        return (value - vmin) / (vmax - vmin)
    else:
        return (vmax - value) / (vmax - vmin)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Column names
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {
        "Lbs-": False,  # minimize
        "Acc+": True,   # maximize
        "Mpg+": True,   # maximize
    }

    # Read data
    data: List[Dict[str, Optional[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    # ------------------------------------------------------------------ #
    # CAR-01: Summary statistics
    # ------------------------------------------------------------------ #
    stats_per_col = {}
    for col in goal_cols:
        col_values = [row[col] for row in data if row[col] is not None]
        if not col_values:
            # All missing – skip; but should not happen in tests
            continue
        base_stats = compute_stats(col_values)
        # Determine best/worst based on goal direction
        maximize = goal_directions[col]
        if maximize:
            best = max(col_values)
            worst = min(col_values)
        else:
            best = min(col_values)
            worst = max(col_values)
        base_stats.update({"best": best, "worst": worst})
        stats_per_col[col] = base_stats

    # Print CAR-01
    print("== CAR-01 ==")
    for col in goal_cols:
        s = stats_per_col[col]
        print(f"{col}: mean={s['mean']:.2f}, median={s['median']:.2f}, "
              f"min={s['min']:.2f}, max={s['max']:.2f}, best={s['best']:.2f}, "
              f"worst={s['worst']:.2f}")

    # Machine-checked JSON for CAR-01
    ans_car_01 = json.dumps(
        {col: {k: round(v, 6) for k, v in stats_per_col[col].items()} for col in goal_cols},
        separators=(",", ":")
    )
    print(f"ANSWER CAR-01 {ans_car_01}")

    # ------------------------------------------------------------------ #
    # CAR-02: Top vehicles based on combined criteria
    # ------------------------------------------------------------------ #
    # Use min and max from stats for normalization
    min_vals = {col: stats_per_col[col]["min"] for col in goal_cols}
    max_vals = {col: stats_per_col[col]["max"] for col in goal_cols}

    scored_rows: List[Tuple[int, float]] = []
    for idx, row in enumerate(data, start=1):  # 1-based
        # Skip if any required value is missing
        if any(row[col] is None for col in goal_cols):
            continue
        # Compute normalized scores
        scores = []
        for col in goal_cols:
            val = row[col]
            scores.append(normalize(val, min_vals[col], max_vals[col], goal_directions[col]))
        overall_score = sum(scores) / len(scores)  # equal weights
        scored_rows.append((idx, overall_score))

    # Get top 10 (or fewer if not enough rows)
    scored_rows.sort(key=lambda x: x[1], reverse=True)
    top_rows = [row[0] for row in scored_rows[:10]]

    # Print CAR-02
    print("\n== CAR-02 ==")
    print("Top 10 vehicles (row numbers):")
    print(", ".join(map(str, top_rows)))

    # Machine-checked JSON for CAR-02
    ans_car_02 = json.dumps({"top": top_rows}, separators=(",", ":"))
    print(f"ANSWER CAR-02 {ans_car_02}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

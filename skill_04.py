#!/usr/bin/env python3
"""
Skill 04 – Real‑time inventory assessment and dynamic summaries.

Usage:
    python skill_04.py <csv_path>
"""

import csv
import json
import sys
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Helpers – same as in skill_01 / skill_03
# --------------------------------------------------------------------------- #
def parse_float(s: str) -> None | float:
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
        return 1.0  # all equal
    if maximize:
        return (value - vmin) / (vmax - vmin)
    else:
        return (vmax - value) / (vmax - vmin)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # ------------------------------------------------------------------ #
    # Read data
    # ------------------------------------------------------------------ #
    data: List[Dict[str, float | None]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    total_rows = len(data)
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {"Lbs-": False, "Acc+": True, "Mpg+": True}

    # ------------------------------------------------------------------ #
    # Arrival / history split
    # ------------------------------------------------------------------ #
    arrival_count = round(0.2 * total_rows)
    arrival_indices = list(range(total_rows - arrival_count, total_rows))
    history_indices = list(range(total_rows - arrival_count))

    # ------------------------------------------------------------------ #
    # CAR-07: Real‑time desirability assessment
    # ------------------------------------------------------------------ #
    assessments: List[Dict[str, object]] = []
    best_rows: List[Tuple[float, int]] = []  # (score, row_number)

    for idx in arrival_indices:
        row_num = idx + 1
        row = data[idx]
        # Compute score if all goal columns present
        if any(row[col] is None for col in goal_cols):
            score = None
        else:
            scores = [
                normalize(row[col], min([r[col] for r in data if r[col] is not None]),
                         max([r[col] for r in data if r[col] is not None]),
                         goal_directions[col])
                for col in goal_cols
            ]
            score = sum(scores) / len(scores)

        if score is not None:
            best_rows.append((score, row_num))
            # Keep best rows sorted
            best_rows.sort(key=lambda x: x[0], reverse=True)
            # Only keep top 5 for memory
            best_rows = best_rows[:5]

        # Desirability: arrival belongs to top 5 if its row number is in best_rows
        desirable = any(rn == row_num for _, rn in best_rows)
        assessments.append({"row": row_num, "desirable": desirable})

    # ------------------------------------------------------------------ #
    # CAR-08: Dynamic checkpoints
    # ------------------------------------------------------------------ #
    checkpoints = [0, 40, 80]
    checkpoint_results = []

    for m in checkpoints:
        # Clamp m to available arrivals
        m_clamped = min(m, arrival_count)
        considered_indices = history_indices + arrival_indices[:m_clamped]
        considered_rows = [data[i] for i in considered_indices]

        # Compute stats per goal column
        summary: Dict[str, Dict[str, float | int]] = {}
        for col in goal_cols:
            col_vals = [r[col] for r in considered_rows if r[col] is not None]
            n = len(col_vals)
            if n == 0:
                mean = min_val = max_val = None
            else:
                stats = compute_stats(col_vals)
                mean = stats["mean"]
                min_val = stats["min"]
                max_val = stats["max"]
            summary[col] = {
                "n": n,
                "mean": round(mean, 6) if mean is not None else None,
                "min": round(min_val, 6) if min_val is not None else None,
                "max": round(max_val, 6) if max_val is not None else None,
            }

        # Determine shortlist of top 5 rows based on overall score
        # Use min/max computed above
        min_vals = {col: summary[col]["min"] for col in goal_cols}
        max_vals = {col: summary[col]["max"] for col in goal_cols}
        scored_rows = []
        for idx in considered_indices:
            row_num = idx + 1
            row = data[idx]
            if any(row[col] is None for col in goal_cols):
                continue
            scores = [
                normalize(row[col], min_vals[col], max_vals[col], goal_directions[col])
                for col in goal_cols
            ]
            overall = sum(scores) / len(scores)
            scored_rows.append((overall, row_num))
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        shortlist = [rn for _, rn in scored_rows[:5]]

        checkpoint_results.append({
            "after": m_clamped,
            "summary": summary,
            "shortlist": shortlist,
        })

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    # CAR-07
    print("== CAR-07 ==")
    print(f"Total arrivals: {arrival_count}")
    print("Assessments:")
    for a in assessments:
        print(f"{a['row']}: {a['desirable']}")
    ans_car_07 = json.dumps(
        {"assessments": assessments},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-07 {ans_car_07}")

    # CAR-08
    print("\n== CAR-08 ==")
    for ck in checkpoint_results:
        after = ck["after"]
        print(f"\nCheckpoint after {after} arrivals:")
        print("Summary:")
        for col in goal_cols:
            s = ck["summary"][col]
            print(f"{col}: n={s['n']}, mean={s['mean']}, min={s['min']}, max={s['max']}")
        print("Shortlist (top 5 row numbers):")
        print(ck["shortlist"])
    ans_car_08 = json.dumps(
        {"checkpoints": checkpoint_results},
        separators=(",", ":"),
    )
    print(f"\nANSWER CAR-08 {ans_car_08}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

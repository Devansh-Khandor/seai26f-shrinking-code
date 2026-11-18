#!/usr/bin/env python3
"""
skill_01.py

Usage: python skill_01.py <csv path>
"""

import json
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return a numeric value, or None if missing."""
    return None if v == "?" else v

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load data
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data: List[List[Any]] = list(rows_iter)

    # Identify goal columns
    goal_indices = [i for i, name in enumerate(header) if name[-1] in "+-"]
    goal_names = [header[i] for i in goal_indices]

    # Extract values for each goal
    goal_values: Dict[str, List[float]] = {name: [] for name in goal_names}
    for r in data:
        for i, name in zip(goal_indices, goal_names):
            val = r[i]
            nval = numeric(val)
            if nval is not None:
                goal_values[name].append(nval)

    # Compute CAR-01 summary
    car01_stats: Dict[str, Dict[str, Any]] = {}
    for name in goal_names:
        vals = goal_values[name]
        direction = 1 if name[-1] == "+" else -1  # + maximize, - minimize
        best_val = max(vals) if direction == 1 else min(vals)
        worst_val = min(vals) if direction == 1 else max(vals)
        car01_stats[name] = {
            "mean": mean(vals),
            "median": median(vals),
            "min": min(vals),
            "max": max(vals),
            "best": best_val,
            "worst": worst_val,
        }

    # Output CAR-01 section
    print("== CAR-01 ==")
    for name, stats in car01_stats.items():
        print(f"{name} mean={stats['mean']:.3f} median={stats['median']:.3f} "
              f"min={stats['min']} max={stats['max']} "
              f"best={stats['best']} worst={stats['worst']}")
    print(f"ANSWER CAR-01 {json.dumps(car01_stats, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # CAR-02: composite ranking
    # ------------------------------------------------------------------
    # We only care about the three goals mentioned
    required = {"Lbs-": None, "Acc+": None, "Mpg+": None}
    # Determine indices for required goals
    req_indices: Dict[str, int] = {name: header.index(name) for name in required}
    # Gather column stats needed for normalization
    col_stats: Dict[str, Tuple[float, float]] = {}  # name -> (min, max)
    for name in required:
        idx = req_indices[name]
        vals = [numeric(r[idx]) for r in data if numeric(r[idx]) is not None]
        if not vals:
            col_stats[name] = (0.0, 1.0)
            continue
        mn = min(vals)
        mx = max(vals)
        if mn == mx:
            mx = mn + 1.0  # avoid division by zero
        col_stats[name] = (mn, mx)

    # Compute score for each row (1‑based index)
    scores: List[Tuple[float, int]] = []  # (score, row_number)
    for row_number, r in enumerate(data, start=1):
        # Lbs-: lower better
        v_lbs = numeric(r[req_indices["Lbs-"]])
        norm_lbs = 0.0
        mn, mx = col_stats["Lbs-"]
        if v_lbs is not None:
            norm_lbs = (mx - v_lbs) / (mx - mn)

        # Acc+: higher better
        v_acc = numeric(r[req_indices["Acc+"]])
        norm_acc = 0.0
        mn, mx = col_stats["Acc+"]
        if v_acc is not None:
            norm_acc = (v_acc - mn) / (mx - mn)

        # Mpg+: higher better
        v_mpg = numeric(r[req_indices["Mpg+"]])
        norm_mpg = 0.0
        mn, mx = col_stats["Mpg+"]
        if v_mpg is not None:
            norm_mpg = (v_mpg - mn) / (mx - mn)

        score = norm_lbs + norm_acc + norm_mpg
        scores.append((score, row_number))

    # Get top 10
    top_rows = [row for _, row in sorted(scores, key=lambda x: (-x[0], x[1]))][:10]
    car02_result = {"top": top_rows}

    # Output CAR-02 section
    print("\n== CAR-02 ==")
    print(f"Top 10 vehicle rows: {top_rows}")
    print(f"ANSWER CAR-02 {json.dumps(car02_result, separators=(',', ':'))}")

if __name__ == "__main__":
    main()

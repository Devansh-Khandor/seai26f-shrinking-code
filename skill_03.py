#!/usr/bin/env python3
"""
skill_03.py

Usage: python skill_03.py <csv path>
"""

import json
import sys
from typing import List, Dict, Tuple

import ezr

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def numeric(v):
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
# CAR-05: cylinder count vs fuel efficiency
# ----------------------------------------------------------------------
def car_05(rows: List[List], header: List[str]) -> None:
    print("== CAR-05 ==")

    # Identify goal columns
    goal_names = [name for name in header if name[-1] in "+-"]

    # Map cylinder count to (rows, mpg_values)
    groups: Dict[int, Dict] = {}
    mpg_index = header.index("Mpg+")

    for r in rows:
        cl = numeric(r[header.index("Clndrs")])
        if cl is None:
            continue
        if cl not in groups:
            groups[cl] = {"rows": [], "mpg": []}
        groups[cl]["rows"].append(r)
        m = numeric(r[mpg_index])
        if m is not None:
            groups[cl]["mpg"].append(m)

    # Prepare JSON structure
    groups_json = {}
    for cl, info in groups.items():
        n = len(info["rows"])
        mp = info["mpg"]
        groups_json[str(cl)] = {
            "n": n,
            "median": round(median(mp), 3) if mp else 0.0,
            "mean": round(mean(mp), 3) if mp else 0.0
        }

    # Print human readable summary
    for cl in sorted(groups_json, key=lambda x: int(x)):
        g = groups_json[cl]
        print(f"Clndrs {cl}: n={g['n']}, mean={g['mean']}, median={g['median']}")

    # Comparisons
    comparisons = []
    cl_values = sorted(groups_json, key=lambda x: int(x))
    for i in range(len(cl_values)):
        for j in range(i + 1, len(cl_values)):
            a = int(cl_values[i])
            b = int(cl_values[j])
            mp_a = groups[a]["mpg"]
            mp_b = groups[b]["mpg"]
            # Use ezr.same to decide if distributions differ reliably
            verdict = "different" if not ezr.same(mp_a, mp_b, eps=0) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
            print(f"Compare {a} vs {b}: {verdict}")

    # Machine-checked answer
    answer = {"groups": groups_json, "comparisons": comparisons}
    print(f"ANSWER CAR-05 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# CAR-06: model-year ranges vs overall performance
# ----------------------------------------------------------------------
def car_06(rows: List[List], header: List[str]) -> None:
    print("\n== CAR-06 ==")

    # Identify goal columns
    goal_names = [name for name in header if name[-1] in "+-"]
    goal_indices = [header.index(name) for name in goal_names]

    # Extract model values
    model_idx = header.index("Model")
    models = [numeric(r[model_idx]) for r in rows if numeric(r[model_idx]) is not None]
    unique_models = sorted(set(models))

    if not unique_models:
        print("No valid model data.")
        print("ANSWER CAR-06 {}")
        return

    # Determine split point (median of unique model values)
    mid = len(unique_models) // 2
    low_range = (unique_models[0], unique_models[mid])
    high_range = (unique_models[mid + 1] if mid + 1 < len(unique_models) else unique_models[-1],
                  unique_models[-1])

    # Helper to collect all goal values for rows in a range
    def collect_goals(rows_subset: List[List]) -> List[float]:
        vals = []
        for r in rows_subset:
            for i in goal_indices:
                v = numeric(r[i])
                if v is not None:
                    vals.append(v)
        return vals

    # Group rows
    group1_rows = [r for r in rows if low_range[0] <= numeric(r[model_idx]) <= low_range[1]]
    group2_rows = [r for r in rows if high_range[0] <= numeric(r[model_idx]) <= high_range[1]]

    # Flatten goal values
    g1_vals = collect_goals(group1_rows)
    g2_vals = collect_goals(group2_rows)

    # Determine verdict
    verdict = "different" if not ezr.same(g1_vals, g2_vals, eps=0) else "same"

    # Prepare JSON
    groups_json = [
        {"from": low_range[0], "to": low_range[1], "n": len(group1_rows)},
        {"from": high_range[0], "to": high_range[1], "n": len(group2_rows)},
    ]
    answer = {"groups": groups_json, "verdict": verdict}

    # Human readable
    print(f"Group 1: {low_range[0]}-{low_range[1]}, n={len(group1_rows)}")
    print(f"Group 2: {high_range[0]}-{high_range[1]}, n={len(group2_rows)}")
    print(f"Verdict: {verdict}")

    # Machine-checked answer
    print(f"ANSWER CAR-06 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows = list(rows_iter)

    car_05(data_rows, header)
    car_06(data_rows, header)

if __name__ == "__main__":
    main()

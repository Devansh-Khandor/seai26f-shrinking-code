#!/usr/bin/env python3
"""
Skill 03 – Statistical comparison of groups.

Usage:
    python skill_03.py <csv_path>
"""

import csv
import json
import sys
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Helpers – same as in skill_01.py
# --------------------------------------------------------------------------- #
def parse_float(s: str):
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
    # Read data
    data: List[Dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    # Goal columns
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {"Lbs-": False, "Acc+": True, "Mpg+": True}

    # Pre‑compute min/max for each goal column
    min_vals = {}
    max_vals = {}
    for col in goal_cols:
        col_vals = [row[col] for row in data if row[col] is not None]
        min_vals[col] = min(col_vals)
        max_vals[col] = max(col_vals)

    # ------------------------------------------------------------------ #
    # CAR-05: Cylinder groups
    # ------------------------------------------------------------------ #
    cyl_groups: Dict[int, List[float]] = {}
    for row in data:
        cyl = row["Clndrs"]
        if cyl is None:
            continue
        mpg = row["Mpg+"]
        if mpg is None:
            continue
        cyl_groups.setdefault(cyl, []).append(mpg)

    # Compute statistics per group
    groups_stats = {}
    for cyl, mpg_vals in cyl_groups.items():
        stats = compute_stats(mpg_vals)
        groups_stats[cyl] = {
            "n": len(mpg_vals),
            "median": round(stats["median"], 6),
            "mean": round(stats["mean"], 6),
        }

    # Compare every pair of groups
    comparisons = []
    cyl_values = sorted(groups_stats.keys())
    for i in range(len(cyl_values)):
        for j in range(i + 1, len(cyl_values)):
            a = cyl_values[i]
            b = cyl_values[j]
            # Prepare data
            mpg_a = np.array(cyl_groups[a])
            mpg_b = np.array(cyl_groups[b])
            mean_a = mpg_a.mean()
            mean_b = mpg_b.mean()
            std_a = mpg_a.std(ddof=1) if len(mpg_a) > 1 else 0.0
            std_b = mpg_b.std(ddof=1) if len(mpg_b) > 1 else 0.0
            n_a = len(mpg_a)
            n_b = len(mpg_b)
            # Standard error of difference
            se = np.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b) if n_a > 1 and n_b > 1 else float("inf")
            diff = abs(mean_a - mean_b)
            verdict = "different" if diff > 1.96 * se else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Print CAR-05 section
    print("== CAR-05 ==")
    print(f"Number of cylinder groups: {len(cyl_groups)}")
    print("Group statistics (mean, median, count of rows):")
    for cyl in sorted(groups_stats.keys()):
        s = groups_stats[cyl]
        print(f"Clndrs={cyl}: n={s['n']}, mean={s['mean']}, median={s['median']}")
    print("Comparisons:")
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Machine‑checked answer
    car_05_ans = json.dumps(
        {"groups": groups_stats, "comparisons": comparisons},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-05 {car_05_ans}")

    # ------------------------------------------------------------------ #
    # CAR-06: Model‑year range comparison
    # ------------------------------------------------------------------ #
    # Build overall performance score per row (as in CAR‑02)
    def overall_score(row: Dict[str, float]) -> float:
        vals = []
        for col in goal_cols:
            if row[col] is None:
                return None
            vals.append(normalize(row[col], min_vals[col], max_vals[col], goal_directions[col]))
        return sum(vals) / len(vals)

    scores = [overall_score(r) for r in data]
    # Filter rows with complete goals
    valid_indices = [i for i, sc in enumerate(scores) if sc is not None]
    valid_scores = np.array([scores[i] for i in valid_indices])

    # Define two model ranges: split by sorted unique Model values
    model_vals = sorted({row["Model"] for row in data if row["Model"] is not None})
    mid = len(model_vals) // 2
    group1_models = set(model_vals[:mid])
    group2_models = set(model_vals[mid:])

    # Assign rows to groups
    group_scores = {1: [], 2: []}
    group_counts = {1: 0, 2: 0}
    for idx, row in enumerate(data):
        model = row["Model"]
        if model is None:
            continue
        group = 1 if model in group1_models else 2
        group_counts[group] += 1
        sc = scores[idx]
        if sc is not None:
            group_scores[group].append(sc)

    # Compute stats per group
    group_info = []
    for g in [1, 2]:
        arr = np.array(group_scores[g])
        mean = float(arr.mean()) if len(arr) > 0 else 0.0
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        group_info.append(
            {
                "from": min(model_vals[:mid]) if g == 1 else min(model_vals[mid:]),
                "to": max(model_vals[:mid]) if g == 1 else max(model_vals[mid:]),
                "n": group_counts[g],
                "mean": round(mean, 6),
                "std": round(std, 6),
            }
        )

    # Compare overall performance
    mean1 = group_info[0]["mean"]
    mean2 = group_info[1]["mean"]
    std1 = group_info[0]["std"]
    std2 = group_info[1]["std"]
    n1 = group_info[0]["n"]
    n2 = group_info[1]["n"]
    se = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2) if n1 > 1 and n2 > 1 else float("inf")
    diff = abs(mean1 - mean2)
    verdict = "different" if diff > 1.96 * se else "same"

    # Print CAR-06 section
    print("\n== CAR-06 ==")
    print(f"Model ranges: {group_info[0]['from']}–{group_info[0]['to']} (n={group_info[0]['n']}) "
          f"vs {group_info[1]['from']}–{group_info[1]['to']} (n={group_info[1]['n']})")
    print(f"Overall performance difference: mean1={group_info[0]['mean']}, mean2={group_info[1]['mean']}")
    print(f"Verdict: {verdict}")

    # Machine‑checked answer
    car_06_ans = json.dumps(
        {"groups": group_info, "verdict": verdict},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-06 {car_06_ans}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

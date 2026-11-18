#!/usr/bin/env python3
"""
Skill 03: Group comparisons for inventory decisions.

Usage:
    python skill_03.py <csv_path>
"""

import sys
import csv
import math
import json
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, compute_stats, normalize
except Exception:
    # Fallback minimal implementations if skill_01 is not available
    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except ValueError:
            return math.nan

    def compute_stats(values: List[float]) -> Dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": math.nan, "median": math.nan}
        mean_val = sum(values) / n
        sorted_vals = sorted(values)
        mid = n // 2
        if n % 2 == 1:
            med_val = sorted_vals[mid]
        else:
            med_val = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return {"mean": mean_val, "median": med_val}

    def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
        if max_val == min_val:
            return 1.0
        if maximize:
            return (val - min_val) / (max_val - min_val)
        return (max_val - val) / (max_val - min_val)

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def stddev(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean_val = sum(values) / n
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

def normal_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def t_test(mean_a: float, mean_b: float,
           std_a: float, std_b: float,
           n_a: int, n_b: int) -> float:
    """Return two-sided p-value for difference in means."""
    if n_a <= 1 or n_b <= 1:
        return 1.0
    denom = math.sqrt((std_a ** 2) / n_a + (std_b ** 2) / n_b)
    if denom == 0:
        return 0.0
    t = abs(mean_a - mean_b) / denom
    # Approximate using normal distribution
    p = 2 * (1 - normal_cdf(t))
    return p

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify goal columns (ending with +/-)
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}

    # ----------------------------------------------------------- #
    # CAR-05: Compare fuel efficiency across Clndrs groups
    # ----------------------------------------------------------- #
    mpg_col = 'Mpg+'
    clndrs_col = 'Clndrs'

    # Build mapping from Clndrs value to list of Mpg+ values
    groups: Dict[str, List[float]] = {}
    for idx, row in enumerate(rows, start=1):
        clndrs_val = row.get(clndrs_col, '?')
        mpg_val = row.get(mpg_col, '?')
        if is_missing(clndrs_val) or is_missing(mpg_val):
            continue
        clndrs_key = str(to_float(clndrs_val))
        groups.setdefault(clndrs_key, []).append(to_float(mpg_val))

    # Prepare group statistics
    group_stats = {}
    for cl, vals in groups.items():
        stats = compute_stats(vals)
        n = len(vals)
        group_stats[cl] = {"n": n, "median": stats["median"], "mean": stats["mean"]}

    # Pairwise comparisons
    cl_values = sorted(groups.keys(), key=lambda x: float(x))
    comparisons = []
    for i in range(len(cl_values)):
        for j in range(i + 1, len(cl_values)):
            a = cl_values[i]
            b = cl_values[j]
            vals_a = groups[a]
            vals_b = groups[b]
            n_a = len(vals_a)
            n_b = len(vals_b)
            mean_a = compute_stats(vals_a)["mean"]
            mean_b = compute_stats(vals_b)["mean"]
            std_a = stddev(vals_a)
            std_b = stddev(vals_b)
            p = t_test(mean_a, mean_b, std_a, std_b, n_a, n_b)
            verdict = "different" if p < 0.05 else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Print human readable section
    print("== CAR-05 ==")
    print("Fuel efficiency by cylinder count:")
    for cl in cl_values:
        gs = group_stats[cl]
        print(f"  Clndrs={cl}: n={gs['n']}, mean={gs['mean']:.2f}, median={gs['median']:.2f}")
    print("\nPairwise comparisons:")
    for comp in comparisons:
        print(f"  {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")
    print()

    # Machine-checked answer
    car05_json = {"groups": group_stats, "comparisons": comparisons}
    print(f"ANSWER CAR-05 {json.dumps(car05_json, separators=(',', ':'))}")

    # ----------------------------------------------------------- #
    # CAR-06: Compare overall performance across two Model ranges
    # ----------------------------------------------------------- #
    model_col = 'Model'
    # Collect unique sorted model values (as numbers)
    model_values = sorted({to_float(r[model_col]) for r in rows if not is_missing(r[model_col])})
    if len(model_values) < 2:
        # Not enough data to create two groups
        print("== CAR-06 ==")
        print("Not enough distinct model years to form two groups.")
        print()
        print(f"ANSWER CAR-06 {{\"groups\":[],\"verdict\":\"same\"}}")
        return

    # Split into two halves
    mid = len(model_values) // 2
    group1_models = model_values[:mid]
    group2_models = model_values[mid:]

    group_ranges = [
        {"from": int(math.floor(min(group1_models))), "to": int(math.ceil(max(group1_models)))},
        {"from": int(math.floor(min(group2_models))), "to": int(math.ceil(max(group2_models)))},
    ]

    # Precompute column min/max for normalization (over all rows with all goal columns present)
    col_min, col_max = {}, {}
    for col, dirn in goal_columns.items():
        vals = [to_float(row[col]) for row in rows if not is_missing(row[col])]
        if vals:
            col_min[col] = min(vals)
            col_max[col] = max(vals)
        else:
            col_min[col] = col_max[col] = 0.0

    # Compute composite score for each row that has all goal columns
    scores_by_row: List[Tuple[int, float]] = []  # (row_index, score)
    for idx, row in enumerate(rows, start=1):
        has_all = True
        vals = []
        for col, dirn in goal_columns.items():
            v = row[col]
            if is_missing(v):
                has_all = False
                break
            vals.append(to_float(v))
        if not has_all:
            continue
        # Normalization
        score = 0.0
        for col, dirn in goal_columns.items():
            v = to_float(row[col])
            best = normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
            score += best
        scores_by_row.append((idx, score))

    # Assign rows to groups based on Model value
    group_scores = {0: [], 1: []}
    for idx, score in scores_by_row:
        row = rows[idx - 1]
        model_val = to_float(row[model_col])
        if model_val in group1_models:
            group_scores[0].append(score)
        elif model_val in group2_models:
            group_scores[1].append(score)

    # Compute statistics per group
    group_stats_performance = []
    for g_idx in [0, 1]:
        scores = group_scores[g_idx]
        n = len(scores)
        mean = sum(scores) / n if n > 0 else 0.0
        std = stddev(scores)
        group_stats_performance.append({"n": n, "mean": mean, "std": std})

    # Compare the two groups
    n1, n2 = group_stats_performance[0]["n"], group_stats_performance[1]["n"]
    mean1, mean2 = group_stats_performance[0]["mean"], group_stats_performance[1]["mean"]
    std1, std2 = group_stats_performance[0]["std"], group_stats_performance[1]["std"]

    p_val = t_test(mean1, mean2, std1, std2, n1, n2)
    verdict = "different" if p_val < 0.05 else "same"

    # Print human readable section
    print("== CAR-06 ==")
    print("Overall performance comparison between two model year ranges:")
    for i, rng in enumerate(group_ranges):
        gs = group_stats_performance[i]
        print(f"  Group {i+1}: from={rng['from']} to={rng['to']}, n={gs['n']}")
    print(f"Verdict: overall performance is {verdict}")
    print()

    # Machine-checked answer
    car06_json = {
        "groups": group_ranges,
        "verdict": verdict
    }
    print(f"ANSWER CAR-06 {json.dumps(car06_json, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

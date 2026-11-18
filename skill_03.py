#!/usr/bin/env python3
"""
Skill 03: Statistical analysis of the auto93 dataset.

Implements:
 - CAR-05: Compare fuel efficiency (Mpg+) across different cylinder counts (Clndrs).
 - CAR-06: Compare overall performance between two Model ranges.
"""

import csv
import json
import sys
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a CSV cell into float or None."""
    if v == '?' or v.strip() == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col_name: str) -> bool:
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')

def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')

# ----------------------------------------------------------------------
# Read CSV
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def median(values: List[float]) -> float:
    n = len(values)
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    else:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

def t_test_reliable(mean1: float, var1: float, n1: int,
                    mean2: float, var2: float, n2: int) -> bool:
    """Return True if difference is statistically significant (p < 0.05)
    using normal approximation with 1.96 threshold."""
    if n1 == 0 or n2 == 0:
        return False
    # Handle case where variance is zero (all equal)
    se = math.sqrt((var1 / n1) + (var2 / n2))
    if se == 0:
        return False
    diff = abs(mean1 - mean2)
    return diff > 1.96 * se

# ----------------------------------------------------------------------
# CAR-05: Cylinder count vs fuel efficiency
# ----------------------------------------------------------------------
def car_05(rows: List[Dict[str, str]], fieldnames: List[str]) -> Dict[str, Any]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        clndrs_raw = row.get('Clndrs', '')
        mpg_raw = row.get('Mpg+', '')
        clndrs = parse_value(clndrs_raw)
        mpg = parse_value(mpg_raw)
        if clndrs is None or mpg is None:
            continue
        groups[str(int(clndrs))].append(mpg)

    # Compute group stats
    group_stats = {}
    for cl, values in groups.items():
        n = len(values)
        mean_val = sum(values) / n if n > 0 else 0.0
        med_val = median(values) if n > 0 else 0.0
        group_stats[cl] = {"n": n, "mean": mean_val, "median": med_val}

    # Comparisons
    comparisons = []
    cl_values = sorted(group_stats.keys())
    for i in range(len(cl_values)):
        for j in range(i + 1, len(cl_values)):
            a = cl_values[i]
            b = cl_values[j]
            stats_a = group_stats[a]
            stats_b = group_stats[b]
            # For variance, compute sample variance
            vals_a = groups[a]
            vals_b = groups[b]
            n_a = stats_a["n"]
            n_b = stats_b["n"]
            mean_a = stats_a["mean"]
            mean_b = stats_b["mean"]
            var_a = sum((x - mean_a) ** 2 for x in vals_a) / (n_a - 1) if n_a > 1 else 0.0
            var_b = sum((x - mean_b) ** 2 for x in vals_b) / (n_b - 1) if n_b > 1 else 0.0
            verdict = "different" if t_test_reliable(mean_a, var_a, n_a, mean_b, var_b, n_b) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    return {"groups": group_stats, "comparisons": comparisons}

# ----------------------------------------------------------------------
# CAR-06: Model range vs overall performance
# ----------------------------------------------------------------------
def car_06(rows: List[Dict[str, str]], fieldnames: List[str]) -> Dict[str, Any]:
    # Parse model years
    model_vals = []
    for row in rows:
        m_raw = row.get('Model', '')
        m = parse_value(m_raw)
        if m is not None:
            model_vals.append(int(m))
    if not model_vals:
        return {"groups": [], "verdict": "same"}

    # Determine median to split groups
    sorted_models = sorted(model_vals)
    mid = len(sorted_models) // 2
    if len(sorted_models) % 2 == 0:
        # even number, use lower median
        median_val = sorted_models[mid - 1]
    else:
        median_val = sorted_models[mid]

    group_a = []
    group_b = []
    for row in rows:
        m_raw = row.get('Model', '')
        m = parse_value(m_raw)
        if m is None:
            continue
        m_int = int(m)
        if m_int <= median_val:
            group_a.append(row)
        else:
            group_b.append(row)

    groups = []
    for grp in [group_a, group_b]:
        if not grp:
            continue
        models_in_grp = [int(parse_value(r['Model'])) for r in grp]
        groups.append({"from": min(models_in_grp), "to": max(models_in_grp), "n": len(grp)})

    # Compute global min/max for goal columns
    goal_cols = [c for c in fieldnames if is_goal(c)]
    global_min = {}
    global_max = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            global_min[col] = min(vals)
            global_max[col] = max(vals)

    # Compute normalized score per row
    def row_score(row):
        score = 0.0
        for col in goal_cols:
            v = parse_value(row[col])
            if v is None:
                continue
            mn = global_min[col]
            mx = global_max[col]
            if mx == mn:
                norm = 1.0
            else:
                if is_maximize(col):
                    norm = (v - mn) / (mx - mn)
                else:
                    norm = (mx - v) / (mx - mn)
            score += norm
        return score

    # Compute group means and variances
    def stats_for_group(grp):
        scores = [row_score(r) for r in grp]
        n = len(scores)
        if n == 0:
            return (0.0, 0.0, 0)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
        return (mean, var, n)

    mean_a, var_a, n_a = stats_for_group(group_a)
    mean_b, var_b, n_b = stats_for_group(group_b)
    verdict = "different" if t_test_reliable(mean_a, var_a, n_a, mean_b, var_b, n_b) else "same"

    return {"groups": groups, "verdict": verdict}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    # CAR-05
    res05 = car_05(rows, fieldnames)
    print("== CAR-05 ==")
    for cl, stats in sorted(res05["groups"].items(), key=lambda x: int(x[0])):
        print(f"Clndrs {cl}: n={stats['n']} mean={stats['mean']:.6f} median={stats['median']:.6f}")
    for comp in res05["comparisons"]:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")
    print("ANSWER CAR-05", json.dumps(res05, ensure_ascii=False))

    # CAR-06
    res06 = car_06(rows, fieldnames)
    print("\n== CAR-06 ==")
    for g in res06["groups"]:
        print(f"Model range {g['from']}–{g['to']}: n={g['n']}")
    print(f"Overall performance verdict: {res06['verdict']}")
    print("ANSWER CAR-06", json.dumps(res06, ensure_ascii=False))

if __name__ == "__main__":
    main()

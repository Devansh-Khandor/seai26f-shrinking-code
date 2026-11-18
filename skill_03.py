#!/usr/bin/env python3
"""
skill_03.py <csv_path>
Implements CAR-05 (cylinder comparison) and CAR-06 (model range comparison).
"""

import csv
import json
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def median(values: List[float]) -> float:
    """Return median of a non-empty list of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    else:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def sample_variance(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    return sum((x - mu) ** 2 for x in values) / (n - 1)

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_csv(csv_path: str) -> Tuple[List[Dict[str, Optional[float]]], List[str]]:
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            parsed: Dict[str, Optional[float]] = {}
            for k, v in row.items():
                if v == '?' or v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append(parsed)
    return rows, headers

# --------------------------------------------------------------------------- #
# Composite score calculation
# --------------------------------------------------------------------------- #

GOAL_COLUMNS = ["Lbs-", "Acc+", "Mpg+"]

def compute_min_max(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Tuple[float, float]]:
    min_max = {}
    for col in GOAL_COLUMNS:
        values = [row[col] for row in rows if row.get(col) is not None]
        if values:
            min_max[col] = (min(values), max(values))
    return min_max

def normalize_value(v: float, mn: float, mx: float, col: str) -> float:
    if mn == mx:
        return 0.0
    if col.endswith('+'):
        return (v - mn) / (mx - mn)
    else:
        return (mx - v) / (mx - mn)

def composite_scores(rows: List[Dict[str, Optional[float]]], min_max: Dict[str, Tuple[float, float]]) -> List[float]:
    scores = []
    for row in rows:
        # skip rows with any missing goal
        if any(row.get(col) is None for col in GOAL_COLUMNS):
            scores.append(None)
            continue
        norm_scores = []
        for col in GOAL_COLUMNS:
            v = row[col]
            mn, mx = min_max[col]
            norm_scores.append(normalize_value(v, mn, mx, col))
        scores.append(sum(norm_scores) / len(norm_scores))
    return scores

# --------------------------------------------------------------------------- #
# CAR-05: cylinder comparison
# --------------------------------------------------------------------------- #

def car_05(rows: List[Dict[str, Optional[float]]]) -> Dict:
    # Group by Clndrs
    groups: Dict[float, List[float]] = defaultdict(list)
    for row in rows:
        cl = row.get("Clndrs")
        mpg = row.get("Mpg+")
        if cl is None or mpg is None:
            continue
        groups[cl].append(mpg)

    # Build stats per group
    group_stats: Dict[str, Dict[str, float]] = {}
    for cl, mpgs in groups.items():
        n = len(mpgs)
        group_stats[str(cl)] = {
            "n": n,
            "median": median(mpgs),
            "mean": mean(mpgs)
        }

    # Comparisons
    comps = []
    sorted_keys = sorted(groups.keys())
    for i in range(len(sorted_keys)):
        for j in range(i + 1, len(sorted_keys)):
            a = sorted_keys[i]
            b = sorted_keys[j]
            mpgs_a = groups[a]
            mpgs_b = groups[b]
            n1, n2 = len(mpgs_a), len(mpgs_b)
            mu1, mu2 = mean(mpgs_a), mean(mpgs_b)
            var1 = sample_variance(mpgs_a)
            var2 = sample_variance(mpgs_b)
            if n1 < 2 or n2 < 2:
                verdict = "same"
            else:
                se = (var1 / n1 + var2 / n2) ** 0.5
                diff = abs(mu1 - mu2)
                verdict = "different" if diff > 1.96 * se else "same"
            comps.append({
                "a": str(a),
                "b": str(b),
                "verdict": verdict
            })

    return {"groups": group_stats, "comparisons": comps}

# --------------------------------------------------------------------------- #
# CAR-06: model range comparison
# --------------------------------------------------------------------------- #

def car_06(rows: List[Dict[str, Optional[float]]]) -> Dict:
    # Determine model ranges: split unique models at median
    models = sorted({row["Model"] for row in rows if row.get("Model") is not None})
    if not models:
        return {"groups": [], "verdict": "same"}
    mid_idx = len(models) // 2
    if mid_idx == 0:
        mid_val = models[0]
    else:
        mid_val = models[mid_idx - 1]
    # Define ranges
    range1 = (min(models), mid_val)
    range2 = (models[mid_idx], max(models)) if mid_idx < len(models) else (mid_val, mid_val)

    # Compute min-max for goals once
    min_max = compute_min_max(rows)
    # Composite scores
    comp_scores = composite_scores(rows, min_max)

    # Assign rows to groups based on Model value
    group1_scores: List[float] = []
    group2_scores: List[float] = []
    for row, score in zip(rows, comp_scores):
        if score is None:
            continue
        m = row.get("Model")
        if m is None:
            continue
        if range1[0] <= m <= range1[1]:
            group1_scores.append(score)
        elif range2[0] <= m <= range2[1]:
            group2_scores.append(score)

    n1, n2 = len(group1_scores), len(group2_scores)
    mu1 = mean(group1_scores) if group1_scores else 0.0
    mu2 = mean(group2_scores) if group2_scores else 0.0
    var1 = sample_variance(group1_scores)
    var2 = sample_variance(group2_scores)
    if n1 < 2 or n2 < 2:
        verdict = "same"
    else:
        se = (var1 / n1 + var2 / n2) ** 0.5
        diff = abs(mu1 - mu2)
        verdict = "different" if diff > 1.96 * se else "same"

    groups_output = [
        {"from": range1[0], "to": range1[1], "n": n1},
        {"from": range2[0], "to": range2[1], "n": n2}
    ]

    return {"groups": groups_output, "verdict": verdict}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, headers = load_csv(csv_path)

    # CAR-05
    print("== CAR-05 ==")
    car05_res = car_05(rows)
    # Print some human-readable results
    print("Cylinder groups statistics:")
    for cl, stats in car05_res["groups"].items():
        print(f"  {cl}: n={stats['n']}, mean Mpg+={stats['mean']:.2f}, median Mpg+={stats['median']:.2f}")
    print("Pairwise comparisons:")
    for comp in car05_res["comparisons"]:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    print(f"ANSWER CAR-05 {json.dumps(car05_res)}")

    # CAR-06
    print("== CAR-06 ==")
    car06_res = car_06(rows)
    print("Model range groups:")
    for g in car06_res["groups"]:
        print(f"  {g['from']} to {g['to']} : n={g['n']}")
    print(f"Verdict: {car06_res['verdict']}")
    print(f"ANSWER CAR-06 {json.dumps(car06_res)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_03.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])

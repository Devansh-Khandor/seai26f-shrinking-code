#!/usr/bin/env python3
"""
skill_03.py - CAR-05 and CAR-06 analysis for used‑car inventory

Usage:
    python skill_03.py <csv_path>
"""

import sys
import csv
import json
import math

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts (rows). Convert numeric columns to float or None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Numeric columns
            for col in ['Lbs-', 'Acc+', 'Mpg+']:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Feature columns (always numeric in this dataset)
            for col in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']:
                try:
                    row[col] = float(row[col])
                except ValueError:
                    row[col] = None
            rows.append(row)
    return rows

def median(lst):
    """Return median of a sorted list of numbers. lst must not be empty."""
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2.0

def mean(lst):
    """Return mean of list. lst must not be empty."""
    return sum(lst) / len(lst) if lst else None

def stddev(lst):
    """Return sample standard deviation (unbiased) of list. Returns None for len<2."""
    n = len(lst)
    if n < 2:
        return None
    mu = mean(lst)
    return math.sqrt(sum((x - mu) ** 2 for x in lst) / (n - 1))

# ----------------------------------------------------------------------
# CAR-05: Cylinder vs Fuel Efficiency
# ----------------------------------------------------------------------
def car05(rows, budget):
    # Group by Clndrs
    groups = {}
    for idx, row in enumerate(rows, start=1):
        if idx > budget:
            break
        cyl = int(row['Clndrs'])
        mpg = row['Mpg+']
        if mpg is None:
            continue
        if cyl not in groups:
            groups[cyl] = []
        groups[cyl].append(mpg)

    # Build group stats
    group_stats = {}
    for cyl, mpgs in sorted(groups.items()):
        group_stats[str(cyl)] = {
            "n": len(mpgs),
            "median": median(mpgs),
            "mean": mean(mpgs)
        }

    # Comparisons
    comparisons = []
    cyl_list = sorted(groups.keys())
    for i in range(len(cyl_list)):
        for j in range(i + 1, len(cyl_list)):
            a, b = cyl_list[i], cyl_list[j]
            mpg_a = groups[a]
            mpg_b = groups[b]
            n_a, n_b = len(mpg_a), len(mpg_b)
            mean_a, mean_b = mean(mpg_a), mean(mpg_b)
            std_a, std_b = stddev(mpg_a), stddev(mpg_b)
            se_a = std_a / math.sqrt(n_a) if std_a is not None else float('inf')
            se_b = std_b / math.sqrt(n_b) if std_b is not None else float('inf')
            se_comb = math.sqrt(se_a**2 + se_b**2)
            diff = abs(mean_a - mean_b)
            verdict = "different" if diff > 2 * se_comb else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    return group_stats, comparisons

# ----------------------------------------------------------------------
# CAR-06: Model year ranges vs Overall performance
# ----------------------------------------------------------------------
def normalize(val, v_min, v_max, reverse=False):
    if v_max == v_min:
        return 0.5
    score = (val - v_min) / (v_max - v_min)
    return 1.0 - score if reverse else score

def car06(rows, budget):
    # Compute min/max for each goal over all rows (not limited by budget)
    lbs_vals = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    acc_vals = [r['Acc+'] for r in rows if r['Acc+'] is not None]
    mpg_vals = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]

    lbs_min, lbs_max = min(lbs_vals), max(lbs_vals)
    acc_min, acc_max = min(acc_vals), max(acc_vals)
    mpg_min, mpg_max = min(mpg_vals), max(mpg_vals)

    # Compute overall scores for rows that have all three goals
    scores = []
    models = []
    for idx, row in enumerate(rows, start=1):
        if row['Lbs-'] is None or row['Acc+'] is None or row['Mpg+'] is None:
            continue
        score = (
            normalize(row['Lbs-'], lbs_min, lbs_max, reverse=True) +
            normalize(row['Acc+'], acc_min, acc_max, reverse=False) +
            normalize(row['Mpg+'], mpg_min, mpg_max, reverse=False)
        )
        scores.append(score)
        models.append(row['Model'])

    # Determine split point on Model
    unique_models = sorted(set(models))
    mid_index = len(unique_models) // 2
    split_value = unique_models[mid_index]  # all values <= split in group1

    # Build groups
    group_bounds = []
    group_counts = []
    for g in [1, 2]:
        if g == 1:
            low, high = unique_models[0], split_value
            condition = lambda m, lo=low, hi=high: lo <= m <= hi
        else:
            low, high = split_value + 1, unique_models[-1]
            condition = lambda m, lo=low, hi=high: lo <= m <= hi
        group_bounds.append({"from": low, "to": high})

    # Count rows per group that have all three goals
    counts = [0, 0]
    for m, s in zip(models, scores):
        if m <= split_value:
            counts[0] += 1
        else:
            counts[1] += 1

    # Compute mean scores per group
    group_scores = [[], []]
    for m, s in zip(models, scores):
        if m <= split_value:
            group_scores[0].append(s)
        else:
            group_scores[1].append(s)

    means = [mean(gr) for gr in group_scores]
    stds = [stddev(gr) for gr in group_scores]
    ns = [len(gr) for gr in group_scores]

    # Determine verdict
    se_a = stds[0] / math.sqrt(ns[0]) if stds[0] is not None else float('inf')
    se_b = stds[1] / math.sqrt(ns[1]) if stds[1] is not None else float('inf')
    se_comb = math.sqrt(se_a**2 + se_b**2)
    diff = abs(means[0] - means[1])
    verdict = "different" if diff > 2 * se_comb else "same"

    groups = [
        {"from": group_bounds[0]["from"], "to": group_bounds[0]["to"], "n": counts[0]},
        {"from": group_bounds[1]["from"], "to": group_bounds[1]["to"], "n": counts[1]}
    ]

    return groups, verdict

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    n_rows = len(rows)

    # Budget: at most min(30, number of rows // 5) rows may be evaluated
    budget = min(30, n_rows // 5)

    # CAR-05
    print("== CAR-05 ==")
    group_stats, comparisons = car05(rows, budget)
    # Print a few human‑friendly lines
    print(f"Number of groups: {len(group_stats)}")
    sample_group = next(iter(group_stats.items()))
    print(f"Group {sample_group[0]}: n={sample_group[1]['n']}, median={sample_group[1]['median']:.4g}, mean={sample_group[1]['mean']:.4g}")
    print(f"Number of pair comparisons: {len(comparisons)}")
    # Machine‑checked answer
    ans_car05 = {"groups": group_stats, "comparisons": comparisons}
    print("ANSWER CAR-05", json.dumps(ans_car05, separators=(',', ':')))

    # CAR-06
    print("\n== CAR-06 ==")
    groups, verdict = car06(rows, budget)
    # Print human‑friendly lines
    print(f"Groups: {groups}")
    print(f"Verdict: {verdict}")
    # Machine‑checked answer
    ans_car06 = {"groups": groups, "verdict": verdict}
    print("ANSWER CAR-06", json.dumps(ans_car06, separators=(',', ':')))

if __name__ == "__main__":
    main()

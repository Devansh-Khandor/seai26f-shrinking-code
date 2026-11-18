#!/usr/bin/env python3
import csv
import json
import sys
import math

# ------------------------------------------------------------------
# Utility functions (copied from previous skills)
# ------------------------------------------------------------------
def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(values):
    """Return dict with mean, median, min, max, variance, std."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    # variance
    var = sum((x - mean) ** 2 for x in sorted_vals) / n if n > 0 else 0.0
    std = math.sqrt(var)
    return {
        'mean': mean,
        'median': median,
        'min': sorted_vals[0],
        'max': sorted_vals[-1],
        'var': var,
        'std': std
    }

# ------------------------------------------------------------------
# CAR-05: Cylinder group analysis
# ------------------------------------------------------------------
def car_05(parsed_rows):
    print("== CAR-05 ==")

    # Group by Clndrs
    groups = {}
    for idx, row in enumerate(parsed_rows, start=1):
        cyl = row.get('Clndrs')
        mpg = row.get('Mpg+')
        if cyl is None or mpg is None:
            continue
        groups.setdefault(cyl, []).append(mpg)

    # Summary stats per group
    group_stats = {}
    for cyl, mpgs in sorted(groups.items()):
        stats = compute_stats(mpgs)
        group_stats[cyl] = {
            'n': len(mpgs),
            'median': round(stats['median'], 3),
            'mean': round(stats['mean'], 3),
            'var': stats['var'],
            'std': stats['std']
        }
        print(f"Clndrs {cyl}: n={len(mpgs)}, median={stats['median']:.3f}, mean={stats['mean']:.3f}")

    # Pairwise comparisons
    comparisons = []
    cyl_values = sorted(groups.keys())
    for i in range(len(cyl_values)):
        for j in range(i + 1, len(cyl_values)):
            a, b = cyl_values[i], cyl_values[j]
            a_stats, b_stats = group_stats[a], group_stats[b]
            diff = abs(a_stats['mean'] - b_stats['mean'])
            se = math.sqrt(a_stats['var'] / a_stats['n'] + b_stats['var'] / b_stats['n'])
            threshold = 1.96 * se
            verdict = "different" if diff > threshold else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
            print(f"Compare Clndrs {a} vs {b}: diff={diff:.3f}, threshold={threshold:.3f} -> {verdict}")

    answer = {
        "groups": {str(cyl): {
            "n": stats["n"],
            "median": stats["median"],
            "mean": stats["mean"]
        } for cyl, stats in group_stats.items()},
        "comparisons": comparisons
    }
    print(f"ANSWER CAR-05 {json.dumps(answer, separators=(',', ':'))}")

# ------------------------------------------------------------------
# CAR-06: Model range comparison
# ------------------------------------------------------------------
def car_06(parsed_rows):
    print("\n== CAR-06 ==")

    # Get all Model values (integers)
    model_vals = [row['Model'] for row in parsed_rows if row['Model'] is not None]
    if not model_vals:
        print("No valid Model values.")
        return

    sorted_models = sorted(set(model_vals))
    split_idx = len(sorted_models) // 2
    split_value = sorted_models[split_idx]  # first value of upper group

    # Define groups by model ranges
    group_ranges = [
        (min(sorted_models), split_value),                     # group 1
        (split_value + 1, max(sorted_models))                 # group 2
    ]

    # Prepare group stats
    group_stats = []
    for g_from, g_to in group_ranges:
        scores = []
        n_rows = 0
        for row in parsed_rows:
            model = row.get('Model')
            if model is None:
                continue
            if g_from <= model <= g_to:
                n_rows += 1
                # compute performance score if all goal values present
                lbs = row.get('Lbs-')
                acc = row.get('Acc+')
                mpg = row.get('Mpg+')
                if None in (lbs, acc, mpg):
                    continue
                score = -lbs + acc + mpg
                scores.append(score)
        if scores:
            stats = compute_stats(scores)
        else:
            stats = {'mean': 0.0, 'var': 0.0, 'std': 0.0}
        group_stats.append({
            "from": g_from,
            "to": g_to,
            "n": n_rows,
            "mean": stats['mean'],
            "var": stats['var'],
            "std": stats['std']
        })
        print(f"Group from {g_from} to {g_to}: n={n_rows}")

    # Comparison
    a, b = group_stats[0], group_stats[1]
    diff = abs(a['mean'] - b['mean'])
    se = math.sqrt(a['var'] / a['n'] + b['var'] / b['n'])
    threshold = 1.96 * se
    verdict = "different" if diff > threshold else "same"
    print(f"Overall performance difference: diff={diff:.3f}, threshold={threshold:.3f} -> {verdict}")

    answer = {
        "groups": [
            {"from": a["from"], "to": a["to"], "n": a["n"]},
            {"from": b["from"], "to": b["to"], "n": b["n"]}
        ],
        "verdict": verdict
    }
    print(f"ANSWER CAR-06 {json.dumps(answer, separators=(',', ':'))}")

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    # Parse numeric values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for key, val in r.items():
            parsed[key] = parse_value(val)
        parsed_rows.append(parsed)

    # Run CAR-05 and CAR-06
    car_05(parsed_rows)
    car_06(parsed_rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

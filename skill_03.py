#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>

Prints:
    CAR-05: Comparison of MPG across cylinder counts with confidence intervals.
    CAR-06: Comparison of MPG between two model‑year ranges with confidence interval.
"""

import csv
import sys
import math
from itertools import combinations

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts with numeric conversion."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                row['Clndrs'] = int(row['Clndrs'])
                row['Model']  = int(row['Model'])
                row['Volume'] = float(row['Volume'])
                row['HpX']    = float(row['HpX'])
                row['origin'] = int(row['origin'])
                row['Lbs-']   = float(row['Lbs-'])
                row['Acc+']   = float(row['Acc+'])
                row['Mpg+']   = float(row['Mpg+'])
                row['_idx']   = idx
                rows.append(row)
            except (ValueError, KeyError):
                # Skip rows with invalid data
                continue
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def mean_variance(values):
    """Return mean, variance (population), and count."""
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, var, n

def ci_diff(mean1, var1, n1, mean2, var2, n2, alpha=0.05):
    """Return (diff, lower, upper, z, p) for two independent samples."""
    diff = mean1 - mean2
    se = math.sqrt(var1 / n1 + var2 / n2)
    z = diff / se if se != 0 else 0.0
    # Two‑tailed p‑value using normal approximation
    p = math.erfc(abs(z) / math.sqrt(2))
    # 95% CI
    lower = diff - 1.96 * se
    upper = diff + 1.96 * se
    return diff, lower, upper, z, p

# ----------------------------------------------------------------------
# CAR-05: Cylinder comparison
# ----------------------------------------------------------------------
def car_05(rows):
    """Compare MPG across cylinder counts."""
    # Group by cylinder
    groups = {}
    for r in rows:
        cyl = r['Clndrs']
        groups.setdefault(cyl, []).append(r['Mpg+'])
    # Compute stats per group
    group_stats = {}
    for cyl, mpgs in groups.items():
        m, v, n = mean_variance(mpgs)
        group_stats[cyl] = (m, v, n)
    # Prepare pairwise comparisons
    pairs = list(combinations(sorted(group_stats.keys()), 2))
    print("== CAR-05 ==")
    if not pairs:
        print("Not enough cylinder groups to compare.")
        return
    # Show first three pair comparisons
    for i, (c1, c2) in enumerate(pairs[:3], 1):
        m1, v1, n1 = group_stats[c1]
        m2, v2, n2 = group_stats[c2]
        diff, lower, upper, z, p = ci_diff(m1, v1, n1, m2, v2, n2)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"{i}. Cylinders {c1} vs {c2}: ΔMPG = {diff:.2f} "
              f"({lower:.2f}, {upper:.2f}), p={p:.3f} → {sig}")

# ----------------------------------------------------------------------
# CAR-06: Model year range comparison
# ----------------------------------------------------------------------
def car_06(rows):
    """Compare MPG between two model‑year ranges."""
    # Extract years
    years = [r['Model'] for r in rows]
    if not years:
        print("== CAR-06 ==")
        print("No model year data available.")
        return
    median_year = sorted(years)[len(years)//2]
    group1_mpgs = [r['Mpg+'] for r in rows if r['Model'] <= median_year]
    group2_mpgs = [r['Mpg+'] for r in rows if r['Model'] > median_year]
    m1, v1, n1 = mean_variance(group1_mpgs)
    m2, v2, n2 = mean_variance(group2_mpgs)
    diff, lower, upper, z, p = ci_diff(m1, v1, n1, m2, v2, n2)
    print("\n== CAR-06 ==")
    sig = "significant" if p < 0.05 else "not significant"
    print(f"Older models (≤{median_year}) mean MPG: {m1:.2f} ({n1} cars)")
    print(f"Newer models (> {median_year}) mean MPG: {m2:.2f} ({n2} cars)")
    print(f"ΔMPG = {diff:.2f} ({lower:.2f}, {upper:.2f}), p={p:.3f} → {sig}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)
    car_05(rows)
    car_06(rows)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle inventory statistical analysis")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)

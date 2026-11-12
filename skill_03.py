#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-05 : Compare the fuel efficiency of vehicles with different cylinder counts
           and indicate whether the observed differences are statistically reliable.
  CAR-06 : Compare the overall performance (fuel efficiency) of two groups of vehicles
           from different model‑year ranges and decide if the evidence is strong enough.

The output follows the requested format exactly: for each request, a line that is
exactly `== <ID> ==` followed by at least three lines of results, including the actual numbers.
"""

import csv
import sys
import math
import numpy as np

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Reads the CSV file and returns a list of dicts with numeric fields.
    Missing numeric values are stored as None.
    Expected columns (in order):
        Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            try:
                rec = {
                    'Clndrs': float(row[0]) if row[0].strip() else None,
                    'Volume': float(row[1]) if row[1].strip() else None,
                    'HpX': float(row[2]) if row[2].strip() else None,
                    'Model': float(row[3]) if row[3].strip() else None,
                    'origin': float(row[4]) if row[4].strip() else None,
                    'Lbs-': float(row[5]) if row[5].strip() else None,
                    'Acc+': float(row[6]) if row[6].strip() else None,
                    'Mpg+': float(row[7]) if row[7].strip() else None,
                }
            except (IndexError, ValueError):
                continue
            # Skip rows with no data at all
            if all(v is None for v in rec.values()):
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def compute_stats(values):
    """Return mean, std, count for a list of numeric values."""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    count = len(arr)
    return mean, std, count

# ---------------------------------------------------------------------------

def map_cylinders(record):
    """
    Roughly map engine displacement (Volume) to cylinder count:
    <200 => 4 cylinders
    200-300 => 6 cylinders
    >300 => 8 cylinders
    """
    vol = record['Volume']
    if vol is None:
        return None
    if vol < 200:
        return 4
    elif vol <= 300:
        return 6
    else:
        return 8

def group_by_cylinders(records):
    """Group mpg values by estimated cylinder count."""
    groups = {4: [], 6: [], 8: []}
    for r in records:
        cyl = map_cylinders(r)
        mpg = r['Mpg+']
        if cyl is not None and mpg is not None:
            groups[cyl].append(mpg)
    # Filter out empty groups
    return {c: vals for c, vals in groups.items() if vals}

# ---------------------------------------------------------------------------

def t_test_mean_diff(group1, group2):
    """
    Perform two-sample t-test assuming unequal variances.
    Returns t-statistic and approximate two-sided p-value.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return None, None
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    se = math.sqrt((s1**2)/n1 + (s2**2)/n2)
    t = (m1 - m2) / se if se != 0 else 0.0
    # Approximate p-value using normal distribution
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p

# ---------------------------------------------------------------------------

def compare_cylinders(records):
    """Generate results for CAR-05."""
    groups = group_by_cylinders(records)
    lines = []
    for cyl in sorted(groups.keys()):
        mpg_vals = groups[cyl]
        mean, std, count = compute_stats(mpg_vals)
        lines.append(f"  {cyl}-cyl: mean mpg={mean:.2f}, std={std:.2f}, count={count}")
    # Pairwise comparison between 8 and 4 cylinders (most extreme)
    if 4 in groups and 8 in groups:
        t, p = t_test_mean_diff(groups[8], groups[4])
        if t is not None:
            mean8, _, _ = compute_stats(groups[8])
            mean4, _, _ = compute_stats(groups[4])
            diff = mean8 - mean4
            lines.append(f"  Difference (8cyl vs 4cyl): Δmpg={diff:.2f}, t={t:.2f}, p={p:.3f}")
    return lines

# ---------------------------------------------------------------------------

def compare_year_ranges(records, early_range, later_range):
    """Generate results for CAR-06."""
    early_mpg = [r['Mpg+'] for r in records if early_range[0] <= r['Model'] <= early_range[1] and r['Mpg+' ] is not None]
    later_mpg = [r['Mpg+'] for r in records if later_range[0] <= r['Model'] <= later_range[1] and r['Mpg+' ] is not None]
    lines = []
    if early_mpg:
        mean, std, count = compute_stats(early_mpg)
        lines.append(f"  Early years ({int(early_range[0])}-{int(early_range[1])}): mean mpg={mean:.2f}, std={std:.2f}, count={count}")
    if later_mpg:
        mean, std, count = compute_stats(later_mpg)
        lines.append(f"  Later years ({int(later_range[0])}-{int(later_range[1])}): mean mpg={mean:.2f}, std={std:.2f}, count={count}")
    if early_mpg and later_mpg:
        t, p = t_test_mean_diff(later_mpg, early_mpg)
        if t is not None:
            mean_l, _, _ = compute_stats(later_mpg)
            mean_e, _, _ = compute_stats(early_mpg)
            diff = mean_l - mean_e
            lines.append(f"  Difference (later vs early): Δmpg={diff:.2f}, t={t:.2f}, p={p:.3f}")
    return lines

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # CAR-05
    print("== CAR-05 ==")
    for line in compare_cylinders(records):
        print(line)
    if not any(line.startswith('  ') for line in compare_cylinders(records)):
        print("  No data available for cylinder comparison.")

    # CAR-06
    # Define year ranges: 70-76 and 77-82 (the full range in the dataset)
    early_range = (70, 76)
    later_range = (77, 82)
    print("\n== CAR-06 ==")
    for line in compare_year_ranges(records, early_range, later_range):
        print(line)
    if not any(line.startswith('  ') for line in compare_year_ranges(records, early_range, later_range)):
        print("  No data available for year‑range comparison.")

if __name__ == "__main__":
    main()

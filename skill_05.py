#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-09 : When a vehicle has unusually poor fuel efficiency, identify which
           of its recorded characteristics are most strongly associated with that outcome.
  CAR-10 : Find the common characteristics of vehicles with unfavorable overall
           performance and help me understand how they differ from our better‑performing vehicles.

The output follows the requested format exactly: for each request, a line that
is exactly `== <ID> ==` followed by at least three lines of results,
including the actual numbers.
"""

import csv
import sys
import numpy as np
import math

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Read the CSV file and return a list of records.
    All numeric fields are stored as float. Missing values are set to None.
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
                    'HpX':    float(row[2]) if row[2].strip() else None,
                    'Model':  float(row[3]) if row[3].strip() else None,
                    'origin': float(row[4]) if row[4].strip() else None,
                    'Lbs-':   float(row[5]) if row[5].strip() else None,
                    'Acc+':   float(row[6]) if row[6].strip() else None,
                    'Mpg+':   float(row[7]) if row[7].strip() else None,
                }
            except (IndexError, ValueError):
                continue
            # Skip rows with no data at all
            if all(v is None for v in rec.values()):
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def correlation_matrix(records, target_key):
    """
    Compute Pearson correlation coefficient of each numeric feature
    with the target (e.g. 'Mpg+').
    Only rows where both the feature and target are present are used.
    Returns a dict mapping feature name -> correlation coefficient.
    """
    features = [k for k in records[0].keys() if k != target_key]
    corrs = {}
    # Build arrays
    target_vals = []
    feature_vals = {f: [] for f in features}
    for rec in records:
        target = rec.get(target_key)
        if target is None:
            continue
        for f in features:
            val = rec.get(f)
            if val is None:
                break
        else:
            target_vals.append(target)
            for f in features:
                feature_vals[f].append(rec[f])

    target_arr = np.array(target_vals, dtype=float)
    for f in features:
        f_arr = np.array(feature_vals[f], dtype=float)
        if len(f_arr) < 2:
            continue
        corr = np.corrcoef(target_arr, f_arr)[0, 1]
        corrs[f] = corr
    return corrs

# ---------------------------------------------------------------------------

def car_09(records):
    """Implement CAR-09: identify features most strongly associated
    with the worst fuel‑efficiency vehicle."""
    # Find vehicle with lowest mpg
    lowest = None
    lowest_idx = None
    for i, rec in enumerate(records, start=1):
        mpg = rec.get('Mpg+')
        if mpg is None:
            continue
        if lowest is None or mpg < lowest:
            lowest = mpg
            lowest_idx = i

    print("== CAR-09 ==")
    if lowest is None:
        print("No MPG data available.")
        return

    print(f"Vehicle row {lowest_idx} has the lowest mpg = {lowest:.1f}.")

    # Compute correlations with mpg
    corrs = correlation_matrix(records, 'Mpg+')
    # Sort by absolute correlation descending
    sorted_corrs = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Pick top 3 features with strongest negative correlation
    top_features = sorted_corrs[:3]
    print("Features most strongly negatively associated with mpg:")
    for f, corr in top_features:
        print(f"  {f}: r={corr:.3f}")

# ---------------------------------------------------------------------------

def car_10(records):
    """Implement CAR-10: compare common characteristics of
    low‑performing vs high‑performing vehicles."""
    # Determine median mpg to split
    mpg_vals = [rec['Mpg+'] for rec in records if rec['Mpg+'] is not None]
    if not mpg_vals:
        print("== CAR-10 ==")
        print("No MPG data available.")
        return
    median = np.median(mpg_vals)

    low_group = [rec for rec in records if rec['Mpg+'] is not None and rec['Mpg+'] <= median]
    high_group = [rec for rec in records if rec['Mpg+'] is not None and rec['Mpg+'] > median]

    print("\n== CAR-10 ==")
    print(f"Median mpg = {median:.2f}.")
    print(f"Low‑performing group: {len(low_group)} vehicles, high‑performing group: {len(high_group)} vehicles.")

    # Features to compare
    features = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+']

    print("\nFeature comparisons (mean ± std):")
    header = f"{'Feature':>8} | {'Low Mean (std)':>15} | {'High Mean (std)':>15} | {'Diff (high‑low)':>15}"
    print(header)
    print("-" * len(header))

    for f in features:
        low_vals = [rec[f] for rec in low_group if rec[f] is not None]
        high_vals = [rec[f] for rec in high_group if rec[f] is not None]
        if not low_vals or not high_vals:
            continue
        low_mean, low_std = np.mean(low_vals), np.std(low_vals, ddof=1)
        high_mean, high_std = np.mean(high_vals), np.std(high_vals, ddof=1)
        diff = high_mean - low_mean
        print(f"{f:>8} | {low_mean:>7.1f}±{low_std:>5.1f} | {high_mean:>7.1f}±{high_std:>5.1f} | {diff:>10.2f}")

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

    car_09(records)
    car_10(records)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

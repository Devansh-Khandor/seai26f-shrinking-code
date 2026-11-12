#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv_path>

Prints:
    CAR-09: Correlation of characteristics with very low fuel efficiency.
    CAR-10: Comparative summary of common characteristics between the worst and best performers.
"""

import csv
import sys
import math
import numpy as np
from statistics import mean

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV and return list of dicts with numeric conversion.
    Rows with missing numeric values are discarded.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                # numeric columns
                row['Clndrs'] = int(row['Clndrs'])
                row['Volume'] = float(row['Volume'])
                row['HpX']    = float(row['HpX'])
                row['Model']  = int(row['Model'])
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
# Correlation helpers
# ----------------------------------------------------------------------
def compute_correlations(rows, target_key='Mpg+'):
    """
    Compute Pearson correlation between each numeric feature (excluding target) and the target.
    Returns a dict: feature -> correlation coefficient.
    """
    features = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+']
    correlations = {}
    target_vals = np.array([r[target_key] for r in rows])
    for feat in features:
        feat_vals = np.array([r[feat] for r in rows])
        if len(target_vals) < 2:
            corr = float('nan')
        else:
            corr = np.corrcoef(target_vals, feat_vals)[0, 1]
        correlations[feat] = corr
    return correlations

# ----------------------------------------------------------------------
# CAR-09: Identify characteristics associated with poor fuel efficiency
# ----------------------------------------------------------------------
def car_09(rows):
    """
    Find the vehicle with the lowest MPG and report the three strongest
    negative correlations between its features and MPG across the dataset.
    """
    print("== CAR-09 ==")

    # Find vehicle with lowest MPG
    low_mpg_row = min(rows, key=lambda r: r['Mpg+'])
    low_mpg = low_mpg_row['Mpg+']
    print(f"Vehicle #{low_mpg_row['_idx']} (ID={low_mpg_row['Clndrs']}) "
          f"has the lowest MPG of {low_mpg:.1f}.")

    # Compute correlations
    corr = compute_correlations(rows)
    # Sort by absolute magnitude descending; pick those with negative correlation
    neg_corr = {k: v for k, v in corr.items() if v < 0}
    sorted_neg = sorted(neg_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Print top three negative correlations
    if not sorted_neg:
        print("No negative correlations found.")
    else:
        top3 = sorted_neg[:3]
        for i, (feat, coeff) in enumerate(top3, 1):
            print(f"{i}. Feature '{feat}' has correlation {coeff:.3f} with MPG.")
    print()

# ----------------------------------------------------------------------
# CAR-10: Compare common characteristics of worst vs best performers
# ----------------------------------------------------------------------
def car_10(rows):
    """
    Compute a combined desirability score and compare the bottom 20% (worst)
    and top 20% (best) vehicles across key characteristics.
    """
    print("== CAR-10 ==")

    # Compute z-scores for weight, mpg, acceleration
    weights = np.array([r['Lbs-'] for r in rows])
    mpgs    = np.array([r['Mpg+'] for r in rows])
    accs    = np.array([r['Acc+'] for r in rows])

    # Helper: z-score
    def z(arr):
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=0)
        return (arr - mean_val) / std_val if std_val != 0 else np.zeros_like(arr)

    wz = z(weights)
    mz = z(mpgs)
    az = z(accs)

    # Combined desirability score: lower weight, higher mpg, lower acc => higher score
    scores = -wz + mz - az

    # Sort rows by score
    sorted_idx = np.argsort(-scores)  # descending
    n = len(rows)
    bottom_n = max(1, n // 5)  # bottom 20%
    top_n = max(1, n // 5)     # top 20%

    bottom_indices = sorted_idx[-bottom_n:]  # worst performers
    top_indices = sorted_idx[:top_n]         # best performers

    # Prepare feature lists
    features = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+', 'Mpg+']

    # Compute mean for each group
    def group_mean(indices):
        return {f: mean([rows[i][f] for i in indices]) for f in features}

    bottom_means = group_mean(bottom_indices)
    top_means    = group_mean(top_indices)

    # Print summary of differences
    print(f"Dataset size: {n} vehicles")
    print(f"Worst performers: {bottom_n} vehicles (lowest 20%)")
    print(f"Best performers: {top_n} vehicles (top 20%)\n")

    # Show at least three contrasting characteristics
    print("Key contrasting characteristics:")
    for f in ['Clndrs', 'Volume', 'Lbs-', 'Mpg+']:
        bw = bottom_means[f]
        tb = top_means[f]
        diff = tb - bw
        print(f"- {f}: worst={bw:.2f}, best={tb:.2f}, Δ={diff:+.2f}")

    # Additional insight: overall mean differences
    print("\nAverage differences across all features:")
    for f in features:
        diff = top_means[f] - bottom_means[f]
        print(f"  {f}: Δ={diff:+.2f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    car_09(rows)
    car_10(rows)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle performance diagnostics")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)

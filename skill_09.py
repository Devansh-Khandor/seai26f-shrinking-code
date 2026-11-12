#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <csv_path>

The script prints:
    * CAR-17: Flag vehicle records whose combinations of characteristics are unusually different from the rest of our inventory.
    * CAR-18: Identify vehicles whose reported fuel efficiency or weight differs substantially from what similar vehicles would lead us to expect.
"""

import sys
import math
from pathlib import Path

# Import helpers from the previous skills
from skill_01 import read_csv, format_number
from skill_02 import (
    linear_regression_coefficients,
    regression_statistics,
    predict_using_coefficients,
)
from skill_03 import mean, stddev

# --------------------------------------------------------------------------- #
# Helper functions for outlier detection
# --------------------------------------------------------------------------- #

def zscore(value, mean_val, sd_val):
    """Return z‑score, handling division by zero."""
    return (value - mean_val) / sd_val if sd_val != 0 else 0.0

def compute_mahalanobis_like(vehicle, means, stds, numeric_keys):
    """Compute sum of squared z‑scores for a vehicle."""
    sum_sq = 0.0
    for k in numeric_keys:
        val = vehicle[k]
        mz = zscore(val, means[k], stds[k])
        sum_sq += mz ** 2
    return sum_sq

# --------------------------------------------------------------------------- #
# CAR-17: Flag unusual combination of characteristics
# --------------------------------------------------------------------------- #

def print_car_17_outliers(data):
    numeric_keys = [
        'Clndrs',
        'Volume',
        'HpX',
        'Model',
        'origin',
        'Lbs-',
        'Acc+',
        'Mpg+',
    ]

    # Compute mean and stddev for each numeric key
    means = {k: mean([v[k] for v in data]) for k in numeric_keys}
    stds  = {k: stddev([v[k] for v in data]) for k in numeric_keys}

    # Compute a simple Mahalanobis‑like distance (sum of squared z‑scores)
    outliers = []
    for v in data:
        dist_sq = compute_mahalanobis_like(v, means, stds, numeric_keys)
        outliers.append((dist_sq, v))

    # Sort by distance (descending)
    outliers.sort(key=lambda x: x[0], reverse=True)

    # Threshold: any vehicle with distance > 9 (≈ 3 SDs combined)
    threshold = 9.0
    flagged = [(d, v) for d, v in outliers if d > threshold]

    print("\n== CAR-17 ==")
    if not flagged:
        print("  No vehicle records flagged as unusual.")
        return

    # Show top 5 (or fewer if not enough)
    for i, (dist_sq, v) in enumerate(flagged[:5], start=1):
        print(
            f"  {i}. Vehicle (Clndrs={v['Clndrs']}, Model={v['Model']}), "
            f"Composite distance = {format_number(dist_sq)}"
        )
    # Ensure at least three lines
    if len(flagged) < 3:
        print(f"  Total flagged vehicles: {len(flagged)}")

# --------------------------------------------------------------------------- #
# CAR-18: Flag discrepancy in fuel efficiency or weight
# --------------------------------------------------------------------------- #

def print_car_18_discrepancies(data):
    # Regression for MPG: features Lbs-, Acc+, Volume, HpX
    mpg_features = ['Lbs-', 'Acc+', 'Volume', 'HpX']
    # Regression for Weight: features Mpg+, Acc+, Volume, HpX
    weight_features = ['Mpg+', 'Acc+', 'Volume', 'HpX']

    try:
        mpg_beta = linear_regression_coefficients(data, 'Mpg+', mpg_features)
        weight_beta = linear_regression_coefficients(data, 'Lbs-', weight_features)
    except ValueError as exc:
        print("\n== CAR-18 ==")
        print(f"  Regression training failed: {exc}")
        return

    # Standard errors
    mpg_sigma = regression_statistics(data, 'Mpg+', mpg_features, mpg_beta)
    weight_sigma = regression_statistics(data, 'Lbs-', weight_features, weight_beta)

    # Thresholds: 2 standard deviations
    mpg_thresh = 2 * mpg_sigma
    weight_thresh = 2 * weight_sigma

    flagged = []
    for v in data:
        actual_mpg = v['Mpg+']
        pred_mpg = predict_using_coefficients(mpg_beta, v, mpg_features)
        mpg_res = actual_mpg - pred_mpg

        actual_wt = v['Lbs-']
        pred_wt = predict_using_coefficients(weight_beta, v, weight_features)
        wt_res = actual_wt - pred_wt

        if abs(mpg_res) > mpg_thresh or abs(wt_res) > weight_thresh:
            flagged.append((v, mpg_res, wt_res))

    print("\n== CAR-18 ==")
    if not flagged:
        print("  No vehicles with significant fuel‑efficiency or weight discrepancies.")
        return

    # Show top 5 discrepancies by magnitude (sum of absolute residuals)
    flagged.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
    for i, (v, mpg_res, wt_res) in enumerate(flagged[:5], start=1):
        print(f"  {i}. Vehicle (Clndrs={v['Clndrs']}, Model={v['Model']}):")
        print(
            f"     Actual MPG={format_number(v['Mpg+'])}, "
            f"Predicted MPG={format_number(v['Mpg+'] - mpg_res)}"
            f" (Δ={format_number(mpg_res)})"
        )
        print(
            f"     Actual Weight={format_number(v['Lbs-'])} kg, "
            f"Predicted Weight={format_number(v['Lbs-'] - wt_res)} kg"
            f" (Δ={format_number(wt_res)})"
        )
    # Ensure at least three lines of output
    if len(flagged) < 3:
        print(f"  Total vehicles with discrepancies: {len(flagged)}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    print_car_17_outliers(data)
    print_car_18_discrepancies(data)

if __name__ == "__main__":
    main()

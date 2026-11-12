#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-17 – Flag vehicle records whose combinations of characteristics are unusually
             different from the rest of our inventory so that staff can review them.
    CAR-18 – Identify vehicles whose reported fuel efficiency or weight differs
             substantially from what similar vehicles would lead us to expect,
             so we can check for possible recording mistakes.

Usage:
    python skill_09.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any, Tuple
import math

# Import helpers from previous skills
from skill_01 import read_csv, format_float
from skill_02 import compute_regression_parameters, predict

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _numeric_fields(rows: List[Dict[str, Any]]) -> List[str]:
    """Return list of numeric keys present in the dataset."""
    if not rows:
        return []
    sample = next((r for r in rows if r), {})
    return [k for k, v in sample.items()
            if isinstance(v, (int, float)) and v is not None]

def _compute_mean_std(rows: List[Dict[str, Any]], keys: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return dictionaries of mean and std for each key."""
    means = {}
    stds = {}
    for key in keys:
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            means[key] = float('nan')
            stds[key] = float('nan')
            continue
        m = sum(vals) / len(vals)
        variance = sum((x - m) ** 2 for x in vals) / len(vals)
        s = math.sqrt(variance)
        means[key] = m
        stds[key] = s if s != 0 else 1.0  # avoid division by zero
    return means, stds

def _zscore_distance(row: Dict[str, Any], means: Dict[str, float], stds: Dict[str, float]) -> float:
    """Compute Euclidean distance of a row in standardized feature space."""
    sq = 0.0
    for key, m in means.items():
        s = stds[key]
        if row[key] is None:
            continue
        z = (row[key] - m) / s
        sq += z * z
    return math.sqrt(sq)

def _print_car17(rows: List[Dict[str, Any]]) -> None:
    print("== CAR-17 ==")
    numeric_keys = _numeric_fields(rows)
    means, stds = _compute_mean_std(rows, numeric_keys)
    distances = []
    for r in rows:
        dist = _zscore_distance(r, means, stds)
        distances.append((r, dist))
    # Sort by descending distance
    distances.sort(key=lambda x: x[1], reverse=True)
    # Show top 5 or at least 3
    top_n = max(3, min(5, len(distances)))
    if not distances:
        print("No records to analyze.")
        return
    print(f"Flagged top {top_n} outlier(s) based on standardized Euclidean distance:")
    for i in range(top_n):
        r, d = distances[i]
        print(f"{i+1}. Vehicle Clndrs={int(r['Clndrs'])}, Volume={int(r['Volume'])}, "
              f"HpX={int(r['HpX'])}, Model={int(r['Model'])}, origin={int(r['origin'])}, "
              f"Lbs-={int(r['Lbs-'])}, Acc+={format_float(r['Acc+'])}, "
              f"Mpg+={int(r['Mpg+'])} | Distance={format_float(d)}")

def _print_car18(rows: List[Dict[str, Any]]) -> None:
    print("\n== CAR-18 ==")

    # -------------------------------------------------------------
    # 1) Weight model
    # -------------------------------------------------------------
    weight_features = ['Acc+', 'Volume', 'HpX', 'Model', 'origin']
    # Ensure all rows have these features where needed
    weight_training = [r for r in rows if r['Lbs-'] is not None and
                       all(r.get(f) is not None for f in weight_features)]
    if weight_training:
        beta_weight, rmse_weight = compute_regression_parameters(
            weight_training, weight_features, 'Lbs-')
        # Flag weight anomalies
        weight_anomalies = []
        for r in rows:
            if r['Lbs-'] is None or any(r.get(f) is None for f in weight_features):
                continue
            pred = predict(r, beta_weight, weight_features)
            if pred is None:
                continue
            residual = abs(r['Lbs-'] - pred)
            if residual > 1.5 * rmse_weight:
                weight_anomalies.append((r, residual))
        weight_anomalies.sort(key=lambda x: x[1], reverse=True)
        top_n_weight = max(3, min(5, len(weight_anomalies)))
        if weight_anomalies:
            print(f"Weight anomalies (top {top_n_weight}):")
            for i in range(top_n_weight):
                r, resid = weight_anomalies[i]
                print(f"{i+1}. Vehicle Clndrs={int(r['Clndrs'])}, Lbs-={int(r['Lbs-'])} "
                      f"pred={format_float(predict(r, beta_weight, weight_features))} "
                      f"| Residual={format_float(resid)} lbs")
        else:
            print("No significant weight anomalies detected.")
    else:
        print("Insufficient data to train weight regression model.")

    # -------------------------------------------------------------
    # 2) MPG model
    # -------------------------------------------------------------
    mpg_features = ['Lbs-', 'Acc+', 'Volume', 'HpX', 'Model', 'origin']
    mpg_training = [r for r in rows if r['Mpg+'] is not None and
                    all(r.get(f) is not None for f in mpg_features)]
    if mpg_training:
        beta_mpg, rmse_mpg = compute_regression_parameters(
            mpg_training, mpg_features, 'Mpg+')
        mpg_anomalies = []
        for r in rows:
            if r['Mpg+'] is None or any(r.get(f) is None for f in mpg_features):
                continue
            pred = predict(r, beta_mpg, mpg_features)
            if pred is None:
                continue
            residual = abs(r['Mpg+'] - pred)
            if residual > 1.5 * rmse_mpg:
                mpg_anomalies.append((r, residual))
        mpg_anomalies.sort(key=lambda x: x[1], reverse=True)
        top_n_mpg = max(3, min(5, len(mpg_anomalies)))
        if mpg_anomalies:
            print(f"\nMPG anomalies (top {top_n_mpg}):")
            for i in range(top_n_mpg):
                r, resid = mpg_anomalies[i]
                print(f"{i+1}. Vehicle Clndrs={int(r['Clndrs'])}, Mpg+={int(r['Mpg+'])} "
                      f"pred={format_float(predict(r, beta_mpg, mpg_features))} "
                      f"| Residual={format_float(resid)} mpg")
        else:
            print("\nNo significant mpg anomalies detected.")
    else:
        print("\nInsufficient data to train mpg regression model.")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    _print_car17(rows)
    _print_car18(rows)

if __name__ == "__main__":
    main()

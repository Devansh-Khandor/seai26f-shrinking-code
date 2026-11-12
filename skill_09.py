#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <csv_path>

Prints:
    CAR-17: Flag vehicle records whose combinations of characteristics are unusually different
            from the rest of the inventory.
    CAR-18: Identify vehicles whose reported fuel efficiency or weight differs
            substantially from what similar vehicles would predict.
"""

import csv
import sys
import math
import argparse
from statistics import mean, stdev
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read the CSV file and return a list of dicts.
    Rows with any missing or invalid numeric data are skipped.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                r = {}
                r['Clndrs'] = int(row['Clndrs'])
                r['Volume'] = float(row['Volume'])
                r['HpX']    = float(row['HpX'])
                r['Model']  = int(row['Model'])
                r['origin'] = int(row['origin'])
                r['Lbs-']   = float(row['Lbs-'])
                r['Acc+']   = float(row['Acc+'])
                r['Mpg+']   = float(row['Mpg+'])
                r['_idx']   = idx
                rows.append(r)
            except (ValueError, KeyError):
                # Skip rows with any missing/invalid numeric data
                continue
    return rows

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def z_score(x, mean_val, std_val):
    return (x - mean_val) / std_val if std_val != 0 else 0.0

def multivariate_z(row, means, stds):
    """Return Euclidean distance in z‑space for weight, mpg, acceleration."""
    wz = z_score(row['Lbs-'], means['Lbs-'], stds['Lbs-'])
    mz = z_score(row['Mpg+'], means['Mpg+'], stds['Mpg+'])
    az = z_score(row['Acc+'], means['Acc+'], stds['Acc+'])
    return math.sqrt(wz**2 + mz**2 + az**2)

def build_regression(rows, target_key, feature_keys):
    """
    Build a linear regression model to predict target_key using feature_keys.
    Returns the model, residuals (list), and standard deviation of residuals.
    """
    X = []
    y = []
    for r in rows:
        try:
            X.append([r[f] for f in feature_keys])
            y.append(r[target_key])
        except KeyError:
            continue
    X = np.array(X)
    y = np.array(y)
    model = LinearRegression(fit_intercept=True).fit(X, y)
    preds = model.predict(X)
    residuals = y - preds
    resid_std = np.std(residuals, ddof=0)  # population std
    return model, residuals, resid_std

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-17: Unusual combinations (multivariate z‑score)
    # ------------------------------------------------------------------
    print("== CAR-17 ==")
    # Compute means and stds
    weight_vals = [r['Lbs-'] for r in rows]
    mpg_vals    = [r['Mpg+'] for r in rows]
    acc_vals    = [r['Acc+'] for r in rows]
    means = {'Lbs-': mean(weight_vals),
             'Mpg+': mean(mpg_vals),
             'Acc+': mean(acc_vals)}
    stds  = {'Lbs-': stdev(weight_vals) if len(rows) > 1 else 0.0,
             'Mpg+': stdev(mpg_vals)   if len(rows) > 1 else 0.0,
             'Acc+': stdev(acc_vals)   if len(rows) > 1 else 0.0}

    # Determine threshold (e.g., 3.0 in z‑space)
    threshold = 3.0
    outliers = []
    for r in rows:
        dist = multivariate_z(r, means, stds)
        if dist > threshold:
            outliers.append((r, dist))

    if not outliers:
        print("No records found with unusually different characteristic combinations.")
        print(f"Threshold used: {threshold:.2f} z‑space distance.")
        print(f"Total vehicles inspected: {len(rows)}")
    else:
        print(f"Threshold used: {threshold:.2f} z‑space distance.")
        print(f"Number of outlier records: {len(outliers)}")
        print("Examples:")
        for i, (r, dist) in enumerate(outliers[:3], 1):
            print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> "
                  f"z‑distance: {dist:.3f}")

    # ------------------------------------------------------------------
    # CAR-18: Unexpected weight or MPG given similar vehicles
    # ------------------------------------------------------------------
    print("\n== CAR-18 ==")
    # Build two regressions: weight ~ [Volume, HpX, Model, origin, Acc+]
    # and mpg ~ [Weight, Acc+, Volume, HpX, origin]
    # Use all rows for training (no separate train/test split)
    weight_features = ['Volume', 'HpX', 'Model', 'origin', 'Acc+']
    mpg_features    = ['Lbs-', 'Acc+', 'Volume', 'HpX', 'origin']

    # Weight model
    model_w, resid_w, std_w = build_regression(rows, 'Lbs-', weight_features)
    # MPG model
    model_m, resid_m, std_m = build_regression(rows, 'Mpg+', mpg_features)

    # Compute residuals for each vehicle
    # Use the models to predict for every row (including itself)
    X_w = np.array([[r[f] for f in weight_features] for r in rows])
    X_m = np.array([[r[f] for f in mpg_features] for r in rows])
    pred_w = model_w.predict(X_w)
    pred_m = model_m.predict(X_m)

    # Threshold: 2 * residual std (approx 95% for normal)
    thresh_w = 2 * std_w
    thresh_m = 2 * std_m

    weight_outliers = []
    mpg_outliers = []

    for r, p_w, p_m, res_w, res_m in zip(rows, pred_w, pred_m, resid_w, resid_m):
        if abs(res_w) > thresh_w:
            weight_outliers.append((r, res_w, p_w))
        if abs(res_m) > thresh_m:
            mpg_outliers.append((r, res_m, p_m))

    if not weight_outliers and not mpg_outliers:
        print("No vehicles exhibit unexpectedly high or low weight/MPG relative to similar models.")
        print(f"Weight residual threshold: ±{thresh_w:.2f}")
        print(f"MPG residual threshold:   ±{thresh_m:.2f}")
    else:
        print(f"Weight residual threshold: ±{thresh_w:.2f}")
        print(f"MPG residual threshold:   ±{thresh_m:.2f}")
        print("\nVehicles with unusually high/low weight:")
        if weight_outliers:
            for i, (r, res, pred) in enumerate(weight_outliers[:3], 1):
                diff = r['Lbs-'] - pred
                print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> "
                      f"Observed: {r['Lbs-']:.1f} lbs, Predicted: {pred:.1f} lbs, "
                      f"Residual: {res:.1f} lbs")
        else:
            print("(None)")

        print("\nVehicles with unusually high/low MPG:")
        if mpg_outliers:
            for i, (r, res, pred) in enumerate(mpg_outliers[:3], 1):
                diff = r['Mpg+'] - pred
                print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> "
                      f"Observed: {r['Mpg+']:.1f} MPG, Predicted: {pred:.1f} MPG, "
                      f"Residual: {res:.1f} MPG")
        else:
            print("(None)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle outlier and discrepancy analysis")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)

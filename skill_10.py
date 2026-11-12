#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <csv_path>

Prints:
    CAR-19: Compare the vehicle selection system against random choice under the same evaluation budget.
    CAR-20: Compare predictive accuracy of our regression models to a simple historical‑average baseline
            on unseen records, reporting mean absolute error and improvement percentage.
"""

import csv
import sys
import math
import random
from statistics import mean
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read the CSV file and return a list of dicts with numeric conversion.
    Rows with any missing or non‑numeric value are skipped.
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
                # Skip rows with invalid data
                continue
    return rows

# ----------------------------------------------------------------------
# Utility: z‑scores and combined score
# ----------------------------------------------------------------------
def compute_z_scores(values):
    """Return mean, std, and z‑scores array."""
    arr = np.array(values, dtype=float)
    m = np.mean(arr)
    s = np.std(arr, ddof=0)
    z = (arr - m) / s if s != 0 else np.zeros_like(arr)
    return m, s, z

def combined_score(row, w_mean, w_std, m_mean, m_std, a_mean, a_std):
    """Compute combined desirability score for a single row."""
    wz = (row['Lbs-'] - w_mean) / w_std if w_std != 0 else 0.0
    mz = (row['Mpg+'] - m_mean) / m_std if m_std != 0 else 0.0
    az = (row['Acc+'] - a_mean) / a_std if a_std != 0 else 0.0
    return -wz + mz - az

# ----------------------------------------------------------------------
# CAR-19: Selection vs. Random
# ----------------------------------------------------------------------
def car_19(rows, budget=10, trials=20):
    """
    Compare the system's top‑budget selection to random selections.
    Prints mean combined score for both approaches.
    """
    print("== CAR-19 ==")
    # Compute global stats for z‑scoring
    weights = [r['Lbs-'] for r in rows]
    mpgs    = [r['Mpg+'] for r in rows]
    accs    = [r['Acc+'] for r in rows]
    w_mean, w_std, _ = compute_z_scores(weights)
    m_mean, m_std, _ = compute_z_scores(mpgs)
    a_mean, a_std, _ = compute_z_scores(accs)

    # Compute score for each vehicle
    for r in rows:
        r['score'] = combined_score(r, w_mean, w_std, m_mean, m_std, a_mean, a_std)

    # Our system: top‑budget vehicles
    sorted_rows = sorted(rows, key=lambda r: r['score'], reverse=True)
    our_best = sorted_rows[:budget]
    our_mean_score = mean(r['score'] for r in our_best)

    # Random selections
    random_means = []
    rng = random.Random(0)  # deterministic seed
    n = len(rows)
    for _ in range(trials):
        sample = rng.sample(rows, budget)
        sample_mean = mean(r['score'] for r in sample)
        random_means.append(sample_mean)
    rand_mean_score = mean(random_means)
    rand_std_score  = np.std(random_means, ddof=0)

    # Output
    print(f"Evaluation budget: {budget} vehicles")
    print(f"System mean score: {our_mean_score:.3f}")
    print(f"Random mean score: {rand_mean_score:.3f} ± {rand_std_score:.3f} (over {trials} trials)")

# ----------------------------------------------------------------------
# CAR-20: Predictive accuracy vs. historical average
# ----------------------------------------------------------------------
def car_20(rows):
    """
    For each vehicle, predict weight and mpg using leave‑one‑out linear regression,
    compare mean absolute error to a simple historical‑average baseline.
    """
    print("\n== CAR-20 ==")

    # Feature sets
    weight_features = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Acc+']
    mpg_features    = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Acc+', 'Lbs-']

    # Prepare arrays
    X_weight = np.array([[r[f] for f in weight_features] for r in rows], dtype=float)
    y_weight = np.array([r['Lbs-'] for r in rows], dtype=float)

    X_mpg    = np.array([[r[f] for f in mpg_features] for r in rows], dtype=float)
    y_mpg    = np.array([r['Mpg+'] for r in rows], dtype=float)

    n = len(rows)
    if n < 2:
        print("Insufficient data for leave‑one‑out evaluation.")
        return

    # Global means for baseline
    weight_global_mean = np.mean(y_weight)
    mpg_global_mean    = np.mean(y_mpg)

    # Store errors
    weight_mae_model = []
    weight_mae_base  = []

    mpg_mae_model    = []
    mpg_mae_base     = []

    # Leave‑one‑out loop
    for i in range(n):
        # Weight prediction
        X_train_w = np.delete(X_weight, i, axis=0)
        y_train_w = np.delete(y_weight, i)
        reg_w = LinearRegression(fit_intercept=True).fit(X_train_w, y_train_w)
        pred_w = reg_w.predict(X_weight[i:i+1])[0]
        err_w = abs(pred_w - y_weight[i])
        weight_mae_model.append(err_w)
        weight_mae_base.append(abs(weight_global_mean - y_weight[i]))

        # MPG prediction
        X_train_m = np.delete(X_mpg, i, axis=0)
        y_train_m = np.delete(y_mpg, i)
        reg_m = LinearRegression(fit_intercept=True).fit(X_train_m, y_train_m)
        pred_m = reg_m.predict(X_mpg[i:i+1])[0]
        err_m = abs(pred_m - y_mpg[i])
        mpg_mae_model.append(err_m)
        mpg_mae_base.append(abs(mpg_global_mean - y_mpg[i]))

    # Compute overall MAEs
    w_mae_model = np.mean(weight_mae_model)
    w_mae_base  = np.mean(weight_mae_base)

    m_mae_model = np.mean(mpg_mae_model)
    m_mae_base  = np.mean(mpg_mae_base)

    # Percent improvement
    w_improve = (w_mae_base - w_mae_model) / w_mae_base * 100 if w_mae_base else float('nan')
    m_improve = (m_mae_base - m_mae_model) / m_mae_base * 100 if m_mae_base else float('nan')

    # Output
    print(f"Weight prediction MAE (model): {w_mae_model:.2f} lbs")
    print(f"Weight prediction MAE (baseline): {w_mae_base:.2f} lbs")
    print(f"Weight MAE improvement: {w_improve:.1f}%")

    print(f"\nMPG prediction MAE (model): {m_mae_model:.2f} MPG")
    print(f"MPG prediction MAE (baseline): {m_mae_base:.2f} MPG")
    print(f"MPG MAE improvement: {m_improve:.1f}%")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    car_19(rows, budget=10, trials=20)
    car_20(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

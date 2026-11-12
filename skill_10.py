#!/usr/bin/env python3
"""
Skill 10: Apache configuration selection and prediction quality analysis

Usage:
    python skill_10.py <csv path>

Reports:

  - APACHE-19  Compare our configuration‑selection system against random selection
               under a fixed benchmarking budget, showing which approach discovers
               better-performing configurations.

  - APACHE-20  Compare our performance predictions against a simple historical‑average
               estimate and show whether our system provides a meaningful improvement
               on previously unseen configurations.

All calculations use only the Python standard library.
"""

import csv
import sys
import math
import random
from collections import defaultdict

# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read the CSV file and return a list of rows as dicts of ints, plus fieldnames."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: int(v) for k, v in r.items()})
    return rows, reader.fieldnames


def get_performance_column(fieldnames):
    """Return the name of the performance column (last column or ending with '-')."""
    for name in fieldnames:
        if name.lower() == 'performance' or name.endswith('-'):
            return name
    return fieldnames[-1]

# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def mean_std(values):
    """Return (mean, std_dev) for a list of numbers (sample std)."""
    n = len(values)
    if n == 0:
        return (None, None)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return (mean, math.sqrt(var))


# --------------------------------------------------------------------------- #
# Regression utilities (same as skill_02)
# --------------------------------------------------------------------------- #

def compute_regression(rows, perf_col, config_cols):
    """
    Compute simple linear regression coefficients based on correlation.
    Returns:
        coeffs: dict mapping config column -> coefficient
        mean_perf: mean of performance values
    """
    perf_vals = [row[perf_col] for row in rows]
    mean_perf, std_perf = mean_std(perf_vals)

    coeffs = {}
    for col in config_cols:
        cfg_vals = [row[col] for row in rows]
        mean_cfg, std_cfg = mean_std(cfg_vals)
        cov = sum((row[col] - mean_cfg) * (row[perf_col] - mean_perf) for row in rows) / (
            len(rows) - 1) if len(rows) > 1 else 0.0
        if std_cfg == 0 or std_perf == 0:
            corr = 0.0
        else:
            corr = cov / (std_cfg * std_perf)
        coeff = corr * std_perf / std_cfg if std_cfg != 0 else 0.0
        coeffs[col] = coeff
    return coeffs, mean_perf


def predict_performance(row, coeffs, mean_perf, config_cols):
    """Predict performance for a single row."""
    pred = mean_perf
    for col in config_cols:
        pred += coeffs[col] * row[col]
    return pred

# --------------------------------------------------------------------------- #
# Report APACHE-19
# --------------------------------------------------------------------------- #

def report_apache_19(rows, perf_col, config_cols):
    """Compare our recommendation system vs random selection under a fixed budget."""
    print("== APACHE-19 ==")

    if len(rows) == 0:
        print("No data to evaluate.")
        return

    # Determine benchmarking budget (number of configurations to test)
    budget = min(10, len(rows))
    if len(rows) <= budget:
        print("Dataset too small for a meaningful comparison; using full set.")
        budget = len(rows)

    # Compute regression and predict performance for all rows
    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)
    for row in rows:
        row['_pred'] = predict_performance(row, coeffs, mean_perf, config_cols)

    # Order by predicted performance (ascending: lower is better)
    ordered = sorted(rows, key=lambda r: r['_pred'])

    # Best performance discovered by recommendation
    rec_best = min(row[perf_col] for row in ordered[:budget])

    # Random sampling
    simulations = 200
    best_random_list = []
    for _ in range(simulations):
        sample = random.sample(rows, len(rows))
        best_random = min(row[perf_col] for row in sample[:budget])
        best_random_list.append(best_random)

    rand_avg, rand_std = mean_std(best_random_list)

    print(f"Benchmarking budget (configurations tested): {budget}")
    print(f"Best performance found by recommendation: {rec_best}")
    print(f"Average best performance from random selection: {rand_avg:.2f}")
    print(f"Standard deviation of random best performances: {rand_std:.2f}")
    improvement = rand_avg - rec_best
    print(f"Performance improvement over random: {improvement:.2f} (lower is better)")

# --------------------------------------------------------------------------- #
# Report APACHE-20
# --------------------------------------------------------------------------- #

def report_apache_20(rows, perf_col, config_cols):
    """Compare regression predictions vs historical-average predictions."""
    print("== APACHE-20 ==")

    if len(rows) == 0:
        print("No data to evaluate.")
        return

    # Regression predictions
    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)
    preds = [predict_performance(row, coeffs, mean_perf, config_cols) for row in rows]
    actuals = [row[perf_col] for row in rows]

    # Mean absolute error (MAE) of regression
    mae_pred = sum(abs(a - p) for a, p in zip(actuals, preds)) / len(actuals)

    # Historical average prediction
    historical_mean = sum(actuals) / len(actuals)
    mae_mean = sum(abs(a - historical_mean) for a in actuals) / len(actuals)

    print(f"Mean absolute error of regression predictions: {mae_pred:.2f}")
    print(f"Mean absolute error of historical average prediction: {mae_mean:.2f}")
    improvement_pct = 100 * (mae_mean - mae_pred) / mae_mean if mae_mean != 0 else 0
    print(f"Regression improves MAE by {improvement_pct:.1f}% over simple average")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    report_apache_19(rows, perf_col, config_cols)
    print()
    report_apache_20(rows, perf_col, config_cols)


if __name__ == "__main__":
    main()

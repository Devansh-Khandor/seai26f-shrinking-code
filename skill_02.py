#!/usr/bin/env python3
"""
Skill 02: Apache performance estimation

Usage:
    python skill_02.py <csv path>

This script implements two new reports for the client:

  - APACHE-03  Estimate performance for a configuration based on past data.
  - APACHE-04  Compare two configurations that differ in exactly one setting
               and estimate their performance difference with an uncertainty.

All calculations are done with plain Python (no external libraries).
"""

import csv
import sys
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV and return list of rows as dict of int and fieldnames."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: int(v) for k, v in r.items()})
    return rows, reader.fieldnames

def split_columns(fieldnames):
    """Return the performance column and all configuration columns."""
    perf_col = fieldnames[-1]  # last column in dataset
    config_cols = fieldnames[:-1]
    return perf_col, config_cols

def mean_std(values):
    """Return mean and sample standard deviation of a list of numbers."""
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var)

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
        # covariance
        cov = sum((row[col] - mean_cfg) * (row[perf_col] - mean_perf)
                  for row in rows) / (len(rows) - 1) if len(rows) > 1 else 0.0
        # correlation
        if std_cfg == 0 or std_perf == 0:
            corr = 0.0
        else:
            corr = cov / (std_cfg * std_perf)
        # coefficient in original scale
        # coeff_i = corr * std_perf / std_cfg
        if std_cfg == 0:
            coeff = 0.0
        else:
            coeff = corr * std_perf / std_cfg
        coeffs[col] = coeff
    return coeffs, mean_perf

def predict_performance(row, coeffs, mean_perf, config_cols):
    """Predict performance for a single row."""
    pred = mean_perf
    for col in config_cols:
        pred += coeffs[col] * row[col]
    return pred

def residuals(rows, perf_col, coeffs, mean_perf, config_cols):
    """Return list of residuals (actual - predicted) for each row."""
    return [row[perf_col] - predict_performance(row, coeffs, mean_perf, config_cols)
            for row in rows]

def std_of_residuals(resid_list):
    return math.sqrt(sum(r ** 2 for r in resid_list) / (len(resid_list) - 1)) if len(resid_list) > 1 else 0.0

def find_pairs_one_diff(rows, config_cols):
    """
    Find all pairs of rows that differ by exactly one configuration setting.
    Return list of tuples (idx1, idx2).
    """
    pairs = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            diff_cnt = sum(1 for col in config_cols if rows[i][col] != rows[j][col])
            if diff_cnt == 1:
                pairs.append((i, j))
    return pairs

# --------------------------------------------------------------------------- #
# Reporting functions
# --------------------------------------------------------------------------- #

def report_apache_03(rows, perf_col, config_cols, coeffs, mean_perf):
    """Print estimated performance for the first few configurations."""
    print("== APACHE-03 ==")
    header = ", ".join(config_cols) + f", actual_perf, estimated_perf"
    print(header)
    for i, row in enumerate(rows[:3], start=1):
        est = predict_performance(row, coeffs, mean_perf, config_cols)
        vals = ", ".join(str(row[col]) for col in config_cols)
        print(f"{vals}, {row[perf_col]}, {est:.2f}")

def report_apache_04(rows, perf_col, config_cols, coeffs, residual_std):
    """Compare pairs of configurations that differ by one setting."""
    print("== APACHE-04 ==")
    pairs = find_pairs_one_diff(rows, config_cols)
    if not pairs:
        print("No configuration pairs differ by exactly one setting.")
        return
    print(f"Comparing {len(pairs)} pairs; showing first 3.")
    for idx, (i, j) in enumerate(pairs[:3], start=1):
        row_i = rows[i]
        row_j = rows[j]
        actual_diff = row_j[perf_col] - row_i[perf_col]
        est_i = predict_performance(row_i, coeffs, mean_perf, config_cols)
        est_j = predict_performance(row_j, coeffs, mean_perf, config_cols)
        est_diff = est_j - est_i
        # Uncertainty: propagate residual_std via standardized difference
        std_diff = 0.0
        for col in config_cols:
            cfg_i = row_i[col]
            cfg_j = row_j[col]
            diff = cfg_j - cfg_i
            # Use standard deviation of config column
            cfg_vals = [r[col] for r in rows]
            _, std_cfg = mean_std(cfg_vals)
            if std_cfg != 0:
                std_diff += (diff / std_cfg) ** 2
        std_diff = math.sqrt(std_diff) * residual_std
        print(f"Pair {idx}: actual diff={actual_diff}, estimated diff={est_diff:.2f}, "
              f"uncertainty=±{std_diff:.2f}")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col, config_cols = split_columns(fieldnames)

    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)
    resid = residuals(rows, perf_col, coeffs, mean_perf, config_cols)
    residual_std = std_of_residuals(resid)

    report_apache_03(rows, perf_col, config_cols, coeffs, mean_perf)
    report_apache_04(rows, perf_col, config_cols, coeffs, residual_std)

if __name__ == "__main__":
    main()

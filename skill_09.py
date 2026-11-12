#!/usr/bin/env python3
"""
Skill 09: Apache configuration anomaly detection

Usage:
    python skill_09.py <csv path>

This script implements two new reports requested by the client:

  - APACHE-17  Identify configurations whose recorded performance values
               are unusually different from what their settings and comparable
               historical configurations would lead us to expect.

  - APACHE-18  Flag configurations whose combinations of settings differ
               substantially from the most similar configurations in our
               historical benchmark records, even when their measured
               performance is not obviously poor.

All calculations are performed with the Python standard library only.
"""

import csv
import sys
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV and return rows as dict of ints plus fieldnames."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: int(v) for k, v in r.items()})
    return rows, reader.fieldnames


def get_performance_column(fieldnames):
    """Return the name of the performance column (last column or ending in '-')."""
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
# Regression utilities (identical to skill_02)
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
        # covariance
        cov = sum((row[col] - mean_cfg) * (row[perf_col] - mean_perf)
                  for row in rows) / (len(rows) - 1) if len(rows) > 1 else 0.0
        # correlation
        if std_cfg == 0 or std_perf == 0:
            corr = 0.0
        else:
            corr = cov / (std_cfg * std_perf)
        # coefficient in original scale
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
# Reporting functions
# --------------------------------------------------------------------------- #

def report_apache_17(rows, perf_col, config_cols, coeffs, mean_perf):
    """APACHE-17: Unexpected performance compared to prediction."""
    print("== APACHE-17 ==")

    # Compute residuals
    residuals = [row[perf_col] - predict_performance(row, coeffs, mean_perf, config_cols)
                 for row in rows]
    _, resid_std = mean_std(residuals)
    threshold = 2 * resid_std if resid_std is not None else 0

    # Gather unusually different configurations
    flagged = []
    for idx, (row, resid) in enumerate(zip(rows, residuals), start=1):
        if abs(resid) > threshold:
            flagged.append((idx, row[perf_col], predict_performance(row, coeffs, mean_perf, config_cols), resid))

    if not flagged:
        print("No configurations with unexpectedly different performance detected.")
        return

    # Show at least three results
    for i, (idx, actual, predicted, resid) in enumerate(flagged[:max(3, len(flagged))], start=1):
        print(f"{idx}. actual={actual}, predicted={predicted:.2f}, residual={resid:.2f}")

def hamming_distance(a, b, config_cols):
    """Count differing configuration settings between two rows."""
    return sum(1 for col in config_cols if a[col] != b[col])

def report_apache_18(rows, perf_col, config_cols, residuals, resid_std):
    """APACHE-18: Config combinations unusually far from similar records."""
    print("== APACHE-18 ==")

    # Determine min Hamming distance for each row
    min_dists = []
    for i, row in enumerate(rows):
        min_dist = math.inf
        for j, other in enumerate(rows):
            if i == j:
                continue
            dist = hamming_distance(row, other, config_cols)
            if dist < min_dist:
                min_dist = dist
        min_dists.append(min_dist)

    # Threshold: consider rows with min distance > 2 as substantially different
    distance_threshold = 2

    # Flag rows that are far in config space but not poorly performing
    flagged = []
    for idx, (row, min_dist, resid) in enumerate(zip(rows, min_dists, residuals), start=1):
        # We want configurations that are not obviously poor: residual within +/- 1 std
        if min_dist > distance_threshold and abs(resid) <= resid_std:
            flagged.append((idx, row[perf_col], min_dist, resid))

    if not flagged:
        print("No configurations with substantially different setting combinations found.")
        return

    # Show at least three results
    for i, (idx, actual, dist, resid) in enumerate(flagged[:max(3, len(flagged))], start=1):
        print(f"{idx}. actual={actual}, dist={dist}, residual={resid:.2f}")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)

    # Precompute residuals and their std once
    residuals = [row[perf_col] - predict_performance(row, coeffs, mean_perf, config_cols)
                 for row in rows]
    _, resid_std = mean_std(residuals)

    report_apache_17(rows, perf_col, config_cols, coeffs, mean_perf)
    report_apache_18(rows, perf_col, config_cols, residuals, resid_std)


if __name__ == "__main__":
    main()

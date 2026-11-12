#!/usr/bin/env python3
"""
Skill 07: Apache configuration search and recommendation

Usage:
    python skill_07.py <csv path>

This script implements two new reports requested by the client:

  - APACHE-13  Find a configuration with a low performance‑metric value
               while evaluating as few candidate configurations as possible.

  - APACHE-14  Recommend which configurations to evaluate first when
               benchmarking budget is limited.

All calculations are performed with the Python standard library only.
"""

import csv
import sys
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV and return list of rows as dict of ints plus fieldnames."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: int(v) for k, v in r.items()})
    return rows, reader.fieldnames


def get_performance_column(fieldnames):
    """Return the name of the performance column."""
    for name in fieldnames:
        if name.lower() == 'performance' or name.endswith('-'):
            return name
    return fieldnames[-1]


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
# Regression utilities
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

def report_apache_13(rows, perf_col, config_cols):
    """Find a low‑performance configuration while evaluating as few candidates as possible."""
    print("== APACHE-13 ==")

    # We'll simulate evaluating configurations until we find the best.
    # Count evaluations until the best performance is discovered.
    best_perf = None
    best_row = None
    eval_count = 0

    for row in rows:
        eval_count += 1
        perf = row[perf_col]
        if best_perf is None or perf < best_perf:
            best_perf = perf
            best_row = row

    # Prepare output
    config_str = ",".join(f"{col}={best_row[col]}" for col in config_cols)
    print(f"Best configuration found after evaluating {eval_count} candidate(s).")
    print(f"Performance: {best_perf}")
    print(f"Configuration: {config_str}")


def report_apache_14(rows, perf_col, config_cols, coeffs, mean_perf):
    """Recommend configurations to evaluate first under a limited benchmarking budget."""
    print("== APACHE-14 ==")

    # Predict performance for all rows
    for row in rows:
        row['_predicted'] = predict_performance(row, coeffs, mean_perf, config_cols)

    # Sort rows by predicted performance (ascending: lower is better)
    sorted_rows = sorted(rows, key=lambda r: r['_predicted'])

    # Recommend first 5 configurations (or fewer if dataset is smaller)
    top_n = min(5, len(sorted_rows))
    print(f"Recommend evaluating the first {top_n} configurations (lowest predicted performance).")
    for i in range(top_n):
        row = sorted_rows[i]
        config_str = ",".join(f"{col}={row[col]}" for col in config_cols)
        print(f"{i+1}. Predicted={row['_predicted']:.2f} | Actual={row[perf_col]} | Config={config_str}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)

    report_apache_13(rows, perf_col, config_cols)
    report_apache_14(rows, perf_col, config_cols, coeffs, mean_perf)


if __name__ == "__main__":
    main()

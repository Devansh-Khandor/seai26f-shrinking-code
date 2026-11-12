#!/usr/bin/env python3
"""
Skill 06: Apache configuration recommendation and comparison analysis

Usage:
    python skill_06.py <csv path>

This script implements two new reports requested by the client:

  - APACHE-11  Explain recommended configuration(s) by comparing them
               to the closest previously tested configurations.
  - APACHE-12  Explain differences between two configurations that
               have distinct performance assessments, indicating
               which settings differ and how those differences relate
               to recorded or predicted outcomes.

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
# Regression utilities (from skill_02)
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

def report_apache_11(rows, perf_col, config_cols):
    """Explain recommended configuration(s)."""
    print("== APACHE-11 ==")
    # Find best (lowest) performance
    best_perf = min(row[perf_col] for row in rows)
    best_rows = [row for row in rows if row[perf_col] == best_perf]

    if not best_rows:
        print("No configurations found.")
        return

    # For each recommended config, find 3 closest in performance
    for idx, rec in enumerate(best_rows, start=1):
        # Compute distances to all others
        diffs = []
        for other in rows:
            if other is rec:
                continue
            diffs.append((abs(other[perf_col] - rec[perf_col]), other))
        diffs.sort(key=lambda x: x[0])
        top3 = diffs[:3]

        config_vals = ",".join(str(rec[col]) for col in config_cols)
        print(f"Recommended #{idx}: config={config_vals} | perf={rec[perf_col]}")
        print(f"  Closest alternatives (by performance):")
        for dist, alt in top3:
            alt_vals = ",".join(str(alt[col]) for col in config_cols)
            print(f"    config={alt_vals} | perf={alt[perf_col]} | Δperf={dist:.2f}")
        print()


def report_apache_12(rows, perf_col, config_cols, coeffs, mean_perf):
    """Explain differences between two configurations that differ in one setting."""
    print("== APACHE-12 ==")
    pairs = find_pairs_one_diff(rows, config_cols)
    if not pairs:
        print("No configuration pairs differ by exactly one setting.")
        return

    # We'll output the first three pairs
    for idx, (i, j) in enumerate(pairs[:3], start=1):
        row_i = rows[i]
        row_j = rows[j]
        # Identify differing setting
        diff_col = None
        for col in config_cols:
            if row_i[col] != row_j[col]:
                diff_col = col
                break
        # Compute actual performance diff
        actual_diff = row_j[perf_col] - row_i[perf_col]
        # Predict performance for both
        pred_i = predict_performance(row_i, coeffs, mean_perf, config_cols)
        pred_j = predict_performance(row_j, coeffs, mean_perf, config_cols)
        pred_diff = pred_j - pred_i

        print(f"Pair {idx}: configs {i+1} vs {j+1}")
        print(f"  Differing setting: {diff_col} ({row_i[diff_col]} -> {row_j[diff_col]})")
        print(f"  Actual performance diff: {actual_diff}")
        print(f"  Predicted performance diff: {pred_diff:.2f}")
        print()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    # Compute regression coefficients once
    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)

    report_apache_11(rows, perf_col, config_cols)
    report_apache_12(rows, perf_col, config_cols, coeffs, mean_perf)


if __name__ == "__main__":
    main()

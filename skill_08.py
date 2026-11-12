#!/usr/bin/env python3
"""
Skill 08: Apache configuration improvement and optimization

Usage:
    python skill_08.py <csv path>

This script implements two new reports requested by the client:

  - APACHE-15  Recommend a small change to a poorly performing configuration
               that could reduce its predicted performance, based on historically
               observed valid configurations.

  - APACHE-16  Find the smallest number of configuration-setting changes that
               could bring a poorly performing configuration closer to the
               best observed performance, avoiding unsupported or invalid
               combinations.

All calculations are performed with the Python standard library only.
"""

import csv
import sys
import math
from collections import defaultdict, Counter

# --------------------------------------------------------------------------- #
# CSV helper
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


# --------------------------------------------------------------------------- #
# Statistics utilities
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
# Report APACHE-15
# --------------------------------------------------------------------------- #

def report_apache_15(rows, perf_col, config_cols, coeffs, mean_perf):
    """Recommend a small change to a poorly performing configuration."""
    print("== APACHE-15 ==")

    # Identify a poorly performing configuration
    all_perf = [row[perf_col] for row in rows]
    mean_perf_all, std_perf_all = mean_std(all_perf)
    threshold = mean_perf_all + std_perf_all

    # Choose first config above threshold; fallback to worst
    poor_row = None
    for row in rows:
        if row[perf_col] >= threshold:
            poor_row = row
            break
    if poor_row is None:
        poor_row = max(rows, key=lambda r: r[perf_col])

    poor_perf = poor_row[perf_col]
    config_vals = ", ".join(f"{col}={poor_row[col]}" for col in config_cols)

    # Search for an alternative with at most 2 setting changes and lower performance
    best_improvement = 0
    best_alt = None
    best_changes = []

    for row in rows:
        if row is poor_row:
            continue
        diff_cols = [col for col in config_cols if row[col] != poor_row[col]]
        if 1 <= len(diff_cols) <= 2:
            improvement = poor_perf - row[perf_col]
            if improvement > best_improvement:
                best_improvement = improvement
                best_alt = row
                best_changes = diff_cols

    if best_alt is None:
        print(f"No alternative configuration with ≤2 setting changes improves performance.")
        return

    alt_perf = best_alt[perf_col]
    pred_alt = predict_performance(best_alt, coeffs, mean_perf, config_cols)

    print(f"Poorly performing configuration (perf={poor_perf}): {config_vals}")
    changes_desc = "; ".join(f"{col} {poor_row[col]}→{best_alt[col]}"
                             for col in best_changes)
    print(f"Recommended change(s): {changes_desc}")
    print(f"Resulting actual performance: {alt_perf}")
    print(f"Predicted performance: {pred_alt:.2f}")


# --------------------------------------------------------------------------- #
# Report APACHE-16
# --------------------------------------------------------------------------- #

def report_apache_16(rows, perf_col, config_cols):
    """Find minimal changes to reach best performance."""
    print("== APACHE-16 ==")

    # Determine best performance target
    best_perf = min(row[perf_col] for row in rows)
    print(f"Target performance (best observed): {best_perf}")

    # Identify a poorly performing configuration (worst)
    worst_row = max(rows, key=lambda r: r[perf_col])
    worst_perf = worst_row[perf_col]
    print(f"Starting from worst configuration (perf={worst_perf}): "
          + ", ".join(f"{c}={worst_row[c]}" for c in config_cols))

    # Find configurations that meet or beat the target
    candidate_rows = [row for row in rows if row[perf_col] <= best_perf]

    if not candidate_rows:
        print("No configuration in dataset achieves the target performance.")
        return

    # Compute minimal number of setting changes needed
    min_changes = None
    best_candidate = None

    for cand in candidate_rows:
        changes = sum(1 for c in config_cols if cand[c] != worst_row[c])
        if min_changes is None or changes < min_changes:
            min_changes = changes
            best_candidate = cand
        elif changes == min_changes:
            # Prefer the one closer to target (lower perf)
            if cand[perf_col] < best_candidate[perf_col]:
                best_candidate = cand

    if min_changes is None:
        print("Could not find a suitable configuration.")
        return

    changes_list = [c for c in config_cols if best_candidate[c] != worst_row[c]]
    print(f"Minimal number of changes required: {min_changes}")
    print("Suggested configuration changes:")
    for c in changes_list:
        print(f"  {c}: {worst_row[c]} → {best_candidate[c]}")
    print(f"Achieved performance: {best_candidate[perf_col]}")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    coeffs, mean_perf = compute_regression(rows, perf_col, config_cols)

    report_apache_15(rows, perf_col, config_cols, coeffs, mean_perf)
    report_apache_16(rows, perf_col, config_cols)


if __name__ == "__main__":
    main()

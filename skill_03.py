#!/usr/bin/env python3
"""
skill_03.py
-----------

This script implements two Apache data‑analysis requests:

  APACHE-05: Compare configurations with KeepAlive enabled and disabled,
             and decide if the observed difference in performance is
             meaningful for decision making.

  APACHE-06: Compare configurations with access logging enabled and
             disabled, and decide if one group consistently performs
             better or if the evidence is inconclusive.

Usage:
    python skill_03.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import numpy as np

# --------------------------------------------------------------------------- #
# Utility: read CSV
# --------------------------------------------------------------------------- #
def read_csv(csv_path):
    """
    Read the CSV file.

    Returns:
        feature_names: list of feature column names
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("Empty CSV header")

        perf_col = header[-1]
        feature_names = header[:-1]

        X_rows = []
        y_rows = []

        for row in reader:
            if len(row) != len(header):
                continue  # skip malformed rows
            try:
                X_rows.append([int(float(v)) for v in row[:-1]])
                y_rows.append(float(row[-1]))
            except ValueError:
                continue

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

    return feature_names, X, y

# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #
def mean_and_std(values):
    """Return mean and standard deviation of a 1‑D numpy array."""
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = values.mean()
    std = values.std(ddof=1) if n > 1 else 0.0
    return mean, std

def ci_for_difference(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    """
    Compute confidence interval for difference of means
    (mean1 - mean2) using normal approximation.
    """
    se = np.sqrt(std1**2 / n1 + std2**2 / n2) if n1 > 0 and n2 > 0 else np.nan
    diff = mean1 - mean2
    z = 1.96  # 95% CI
    lower = diff - z * se
    upper = diff + z * se
    return diff, se, lower, upper

# --------------------------------------------------------------------------- #
# APACHE‑05: KeepAlive comparison
# --------------------------------------------------------------------------- #
def analyze_keepalive(feature_names, X, y):
    """
    Compare performance for KeepAlive=1 vs KeepAlive=0.
    Return a dict of statistics.
    """
    keepalive_idx = feature_names.index("keepAlive")
    enabled = y[X[:, keepalive_idx] == 1]
    disabled = y[X[:, keepalive_idx] == 0]

    mean_en, std_en = mean_and_std(enabled)
    mean_dis, std_dis = mean_and_std(disabled)
    n_en, n_dis = len(enabled), len(disabled)

    diff, se, lower, upper = ci_for_difference(
        mean_en, std_en, n_en, mean_dis, std_dis, n_dis
    )

    meaningful = (lower > 0) or (upper < 0)

    return {
        "mean_enabled": mean_en,
        "std_enabled": std_en,
        "n_enabled": n_en,
        "mean_disabled": mean_dis,
        "std_disabled": std_dis,
        "n_disabled": n_dis,
        "diff": diff,
        "ci_low": lower,
        "ci_high": upper,
        "meaningful": meaningful,
    }

# --------------------------------------------------------------------------- #
# APACHE‑06: Access log comparison
# --------------------------------------------------------------------------- #
def analyze_accesslog(feature_names, X, y):
    """
    Compare performance for accessLog=1 vs accessLog=0.
    Return a dict of statistics.
    """
    access_idx = feature_names.index("accessLog")
    enabled = y[X[:, access_idx] == 1]
    disabled = y[X[:, access_idx] == 0]

    mean_en, std_en = mean_and_std(enabled)
    mean_dis, std_dis = mean_and_std(disabled)
    n_en, n_dis = len(enabled), len(disabled)

    diff, se, lower, upper = ci_for_difference(
        mean_en, std_en, n_en, mean_dis, std_dis, n_dis
    )

    meaningful = (lower > 0) or (upper < 0)

    return {
        "mean_enabled": mean_en,
        "std_enabled": std_en,
        "n_enabled": n_en,
        "mean_disabled": mean_dis,
        "std_disabled": std_dis,
        "n_disabled": n_dis,
        "diff": diff,
        "ci_low": lower,
        "ci_high": upper,
        "meaningful": meaningful,
    }

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        feature_names, X, y = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # APACHE-05
    keep_stats = analyze_keepalive(feature_names, X, y)
    print("== APACHE-05 ==")
    print(f"KeepAlive enabled: mean={keep_stats['mean_enabled']:.2f} "
          f"std={keep_stats['std_enabled']:.2f} n={keep_stats['n_enabled']}")
    print(f"KeepAlive disabled: mean={keep_stats['mean_disabled']:.2f} "
          f"std={keep_stats['std_disabled']:.2f} n={keep_stats['n_disabled']}")
    print(f"Mean difference (enabled - disabled): {keep_stats['diff']:.2f}")
    print(f"95% CI for difference: [{keep_stats['ci_low']:.2f}, "
          f"{keep_stats['ci_high']:.2f}]")
    conclusion = ("The observed difference is "
                  "meaningful for decision making." if keep_stats["meaningful"]
                  else "The observed difference is inconclusive.")
    print(conclusion)

    # APACHE-06
    access_stats = analyze_accesslog(feature_names, X, y)
    print("\n== APACHE-06 ==")
    print(f"Access logging enabled: mean={access_stats['mean_enabled']:.2f} "
          f"std={access_stats['std_enabled']:.2f} n={access_stats['n_enabled']}")
    print(f"Access logging disabled: mean={access_stats['mean_disabled']:.2f} "
          f"std={access_stats['std_disabled']:.2f} n={access_stats['n_disabled']}")
    print(f"Mean difference (enabled - disabled): {access_stats['diff']:.2f}")
    print(f"95% CI for difference: [{access_stats['ci_low']:.2f}, "
          f"{access_stats['ci_high']:.2f}]")
    conclusion2 = ("One group consistently performs better." if access_stats["meaningful"]
                    else "Evidence is inconclusive.")
    print(conclusion2)

if __name__ == "__main__":
    main()

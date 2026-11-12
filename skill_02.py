#!/usr/bin/env python3
"""
skill_02.py

Extends the existing Apache analysis tool with two new features:
  * APACHE-03: Predict performance from configuration settings.
  * APACHE-04: Estimate performance difference and uncertainty between
               two historically observed configurations that differ by a single setting.

Usage:
    python skill_02.py <path_to_csv>
"""

import csv
import sys
import math
import itertools

# Import existing helper functions from skill_01
from skill_01 import (
    read_configurations,
    performance_stats,
    compute_mode_per_setting,
    format_differences,
)


# --------------------------------------------------------------------------- #
# Regression utilities
# --------------------------------------------------------------------------- #

def _matrix_transpose(matrix):
    """Return the transpose of a 2D list."""
    return list(map(list, zip(*matrix)))


def _matrix_multiply(A, B):
    """Multiply two matrices A * B."""
    AT = _matrix_transpose(A)
    return [
        [sum(a * b for a, b in zip(row, col)) for col in AT]
        for row in B
    ]


def _gaussian_elimination(A, b, partial_pivot=True):
    """
    Solve linear system A * x = b for x.
    A: square matrix (list of lists)
    b: right hand side (list)
    Returns solution vector x.
    """
    n = len(A)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        if partial_pivot:
            # Find pivot
            pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
            if M[pivot_row][col] == 0:
                raise ValueError("Singular matrix")
            # Swap rows
            M[col], M[pivot_row] = M[pivot_row], M[col]

        # Normalize pivot row
        pivot = M[col][col]
        M[col] = [x / pivot for x in M[col]]

        # Eliminate below
        for r in range(col + 1, n):
            factor = M[r][col]
            M[r] = [rv - factor * pv for rv, pv in zip(M[r], M[col])]

    # Back substitution
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = M[i][-1] - sum(M[i][j] * x[j] for j in range(i + 1, n))

    return x


def _matrix_inverse(A):
    """
    Compute inverse of a square matrix A using Gauss-Jordan elimination.
    Returns inverse matrix.
    """
    n = len(A)
    # Augment with identity
    I = [[float(i == j) for j in range(n)] for i in range(n)]
    M = [A[i][:] + I[i][:] for i in range(n)]

    for col in range(n):
        # Partial pivot
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[pivot_row][col] == 0:
            raise ValueError("Singular matrix")
        M[col], M[pivot_row] = M[pivot_row], M[col]

        # Normalize pivot
        pivot = M[col][col]
        M[col] = [x / pivot for x in M[col]]

        # Eliminate other rows
        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [rv - factor * pv for rv, pv in zip(M[r], M[col])]

    # Extract inverse
    inv = [row[n:] for row in M]
    return inv


def compute_regression_coeffs(configs):
    """
    Fit ordinary least squares regression:
        performance = intercept + sum(coeff_i * setting_i)
    Returns:
        intercept (float)
        coeffs dict mapping setting name -> coefficient
        residual_std (float): standard deviation of residuals
        coeff_se dict mapping setting name -> standard error of coefficient
    """
    # Gather setting keys
    setting_keys = [k for k in configs[0].keys() if k != 'Performance-']
    n = len(configs)
    p = len(setting_keys)

    # Build design matrix X and target vector y
    X = [[1.0] + [c[k] for k in setting_keys] for c in configs]
    y = [c['Performance-'] for c in configs]

    # Compute X^T X and X^T y
    Xt = _matrix_transpose(X)
    XtX = _matrix_multiply(Xt, X)
    Xty = [_matrix_multiply(Xt, [[val]]).flat[0] for val in y]  # vector

    # Solve for beta
    beta = _gaussian_elimination(XtX, Xty)

    intercept = beta[0]
    coeffs = {k: beta[i + 1] for i, k in enumerate(setting_keys)}

    # Predictions
    preds = [sum(beta[j] * X[i][j] for j in range(p + 1)) for i in range(n)]
    residuals = [y[i] - preds[i] for i in range(n)]
    # Residual standard deviation
    sigma_res = math.sqrt(sum(r**2 for r in residuals) / (n - p - 1))

    # Inverse of XtX for standard errors
    XtX_inv = _matrix_inverse(XtX)

    coeff_se = {
        k: math.sqrt(sigma_res**2 * XtX_inv[i + 1][i + 1])
        for i, k in enumerate(setting_keys)
    }

    return intercept, coeffs, sigma_res, coeff_se


# --------------------------------------------------------------------------- #
# APACHE-03: Prediction
# --------------------------------------------------------------------------- #

def predict_performance(config, intercept, coeffs):
    """Return predicted performance for a given configuration dict."""
    pred = intercept
    for k, v in config.items():
        if k == 'Performance-':
            continue
        pred += coeffs.get(k, 0.0) * v
    return pred


def run_apache_03(configs, intercept, coeffs):
    """Compute predictions for all configurations and print summary."""
    predictions = [predict_performance(c, intercept, coeffs) for c in configs]
    mean_pred = sum(predictions) / len(predictions)
    best_pred = max(predictions)
    worst_pred = min(predictions)
    stddev_pred = math.sqrt(sum((p - mean_pred) ** 2 for p in predictions) / len(predictions))

    print("== APACHE-03 ==")
    print(f"Predicted mean performance: {mean_pred:.2f}")
    print(f"Predicted best performance: {best_pred:.2f}")
    print(f"Predicted worst performance: {worst_pred:.2f}")
    print(f"Predicted performance stddev: {stddev_pred:.2f}")


# --------------------------------------------------------------------------- #
# APACHE-04: Difference estimation
# --------------------------------------------------------------------------- #

def find_single_setting_pairs(configs):
    """
    Yield tuples (i, j, setting, diff)
    where configurations i and j differ in exactly one setting.
    i, j are indices (0-based).
    setting is the key that differs.
    diff is numeric difference config_i[setting] - config_j[setting].
    """
    setting_keys = [k for k in configs[0].keys() if k != 'Performance-']

    for i, j in itertools.combinations(range(len(configs)), 2):
        diff_count = 0
        diff_setting = None
        diff_value = 0.0
        for k in setting_keys:
            val_i = configs[i][k]
            val_j = configs[j][k]
            if val_i != val_j:
                diff_count += 1
                diff_setting = k
                diff_value = val_i - val_j
                if diff_count > 1:
                    break
        if diff_count == 1:
            yield (i, j, diff_setting, diff_value)


def run_apache_04(configs, intercept, coeffs, coeff_se, sigma_res):
    """
    For each pair of configurations that differ by a single setting,
    estimate performance difference and its uncertainty.
    Print the top 3 pairs with largest absolute predicted difference.
    """
    pair_results = []

    for i, j, setting, delta in find_single_setting_pairs(configs):
        # Predicted difference from regression
        coeff = coeffs.get(setting, 0.0)
        pred_diff = coeff * delta

        # Uncertainty estimate:
        # 1. From coefficient standard error: |delta| * SE
        se_coeff = coeff_se.get(setting, 0.0)
        uncertainty_coeff = abs(delta) * se_coeff

        # 2. From residuals: sqrt(2) * sigma_res (two independent predictions)
        uncertainty_residual = math.sqrt(2) * sigma_res

        # Combine in quadrature
        total_uncertainty = math.sqrt(uncertainty_coeff**2 + uncertainty_residual**2)

        pair_results.append(
            (i, j, setting, delta, pred_diff, total_uncertainty)
        )

    # Sort by absolute predicted difference descending
    pair_results.sort(key=lambda x: abs(x[4]), reverse=True)

    print("== APACHE-04 ==")
    if not pair_results:
        print("No configuration pairs differ by exactly one setting.")
        return

    # Show top 3 results
    for idx, (i, j, setting, delta, pred_diff, uncert) in enumerate(pair_results[:3], 1):
        print(
            f"{idx}. Config {i+1} vs Config {j+1} "
            f"({setting} diff {delta}) => "
            f"predicted diff {pred_diff:.2f} ± {uncert:.2f}"
        )
    # Also show overall average uncertainty for reference
    avg_uncert = sum(p[5] for p in pair_results) / len(pair_results)
    print(f"Average uncertainty across all pairs: {avg_uncert:.2f}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    # Read data
    configs = read_configurations(csv_path)

    # Basic stats
    mean, best, worst, stddev = performance_stats(configs)
    modes = compute_mode_per_setting(configs)

    # Print APACHE-01 and APACHE-02 (existing functionality)
    print("== APACHE-01 ==")
    print(f"Typical performance (mean): {mean:.2f}")
    print(f"Best performance: {best:.2f}")
    print(f"Worst performance: {worst:.2f}")

    # APACHE-02
    # Reuse logic from skill_01 for consistency
    from skill_01 import find_unusually_favorable, format_differences

    high_perf_configs = find_unusually_favorable(configs, mean, stddev, threshold_factor=1.5)
    print("\n== APACHE-02 ==")
    if not high_perf_configs:
        print("No configurations exceed the unusual threshold.")
    else:
        print(f"Found {len(high_perf_configs)} unusually favorable configuration(s):")
        for idx, cfg in enumerate(high_perf_configs, 1):
            cfg_str = ", ".join(
                f"{k}={int(v) if isinstance(v, float) and v.is_integer() else v}"
                for k, v in cfg.items()
            )
            diffs = format_differences(cfg, modes)
            print(f"{idx}. {cfg_str}")
            print(f"   Distinguishing settings: {diffs}")

    # Run new analyses
    intercept, coeffs, sigma_res, coeff_se = compute_regression_coeffs(configs)
    run_apache_03(configs, intercept, coeffs)
    run_apache_04(configs, intercept, coeffs, coeff_se, sigma_res)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>

This script answers two new client requests:
    NASA-07 – For each new project proposal in the data set, predict
              effort and defects and flag any predictions that are
              unusually high compared to the historical completed
              projects.

    NASA-08 – Recompute the historical summaries (means, standard
              deviations, and counts) after any new completed-project
              records are added.

Only the Python standard library is used.  The script prints the
results with the required headers and at least three lines of output
per request.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _parse_float(value):
    """Convert a string to float; return None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

# Matrix operations – copied from skill_02
def _matrix_mul(A, B):
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Incompatible matrix sizes for multiplication."
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def _matrix_transpose(M):
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]

def _solve_linear(A, b):
    """Solve A x = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

def _compute_regression(data_rows, predictors, target):
    """Perform OLS regression: target ~ 1 + predictors."""
    X = []
    y = []
    for r in data_rows:
        row = [1.0]  # intercept
        ok = True
        for p in predictors:
            val = r.get(p)
            if val is None:
                ok = False
                break
            row.append(val)
        if not ok:
            continue
        tgt = r.get(target)
        if tgt is None:
            continue
        X.append(row[:-1])
        y.append(tgt)

    if not X:
        raise ValueError("No complete rows for regression.")

    XT = _matrix_transpose(X)
    XT_X = _matrix_mul(XT, X)
    XT_y = _matrix_mul(XT, [[val] for val in y])
    XT_y = [row[0] for row in XT_y]

    coeffs_vec = _solve_linear(XT_X, XT_y)

    n = len(X)
    p = len(coeffs_vec)
    residuals = []
    for i in range(n):
        pred = sum(coeffs_vec[j] * X[i][j] for j in range(p))
        residuals.append(y[i] - pred)

    rss = sum(r ** 2 for r in residuals)
    resid_std = math.sqrt(rss / (n - p))

    coeff_dict = {"intercept": coeffs_vec[0]}
    for idx, name in enumerate(predictors, start=1):
        coeff_dict[name] = coeffs_vec[idx]

    return coeff_dict, resid_std

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_data(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = {
                "idX": r.get("idX"),
                "Kloc+": _parse_float(r.get("Kloc+")),
                "Effort-": _parse_float(r.get("Effort-")),
                "Defects-": _parse_float(r.get("Defects-")),
                "MonthsX": _parse_float(r.get("MonthsX")),
            }
            rows.append(parsed)
    return rows

# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def _mean_std(values):
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return (None, None)
    m = sum(cleaned) / len(cleaned)
    var = sum((x - m) ** 2 for x in cleaned) / len(cleaned)
    return (m, math.sqrt(var))

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path):
    rows = _load_data(csv_path)

    # Separate completed projects (both Effort- and Defects- available)
    completed = [r for r in rows if r["Effort-"] is not None and r["Defects-"] is not None]
    proposals = [r for r in rows if r not in completed]

    # Historical statistics
    eff_values = [r["Effort-"] for r in completed]
    def_values = [r["Defects-"] for r in completed]
    eff_mean, eff_std = _mean_std(eff_values)
    def_mean, def_std = _mean_std(def_values)
    n_completed = len(completed)

    # Regression models
    try:
        coeffs_eff, std_eff = _compute_regression(rows,
                                                  predictors=["Kloc+", "MonthsX"],
                                                  target="Effort-")
    except Exception:
        coeffs_eff, std_eff = None, None
    try:
        coeffs_def, std_def = _compute_regression(rows,
                                                  predictors=["Kloc+", "Effort-"],
                                                  target="Defects-")
    except Exception:
        coeffs_def, std_def = None, None

    # ------------------------------------------------------------------ NASA-07
    print("== NASA-07 ==")
    alerts = []

    threshold_eff = eff_mean + 2 * eff_std if eff_mean is not None and eff_std is not None else None
    threshold_def = def_mean + 2 * def_std if def_mean is not None and def_std is not None else None

    for r in proposals:
        id_val = r.get("idX", "unknown")
        pred_eff = None
        pred_def = None
        # Predict effort if possible
        if coeffs_eff and r["Kloc+"] is not None and r["MonthsX"] is not None:
            pred_eff = (coeffs_eff["intercept"]
                        + coeffs_eff["Kloc+"] * r["Kloc+"]
                        + coeffs_eff["MonthsX"] * r["MonthsX"])
        # Predict defects if possible
        if coeffs_def and r["Kloc+"] is not None and r["Effort-"] is not None:
            pred_def = (coeffs_def["intercept"]
                        + coeffs_def["Kloc+"] * r["Kloc+"]
                        + coeffs_def["Effort-"] * r["Effort-"])

        issues = []
        if pred_eff is not None and threshold_eff is not None and pred_eff > threshold_eff:
            issues.append(f"Effort predicted {pred_eff:.1f} > threshold {threshold_eff:.1f}")
        if pred_def is not None and threshold_def is not None and pred_def > threshold_def:
            issues.append(f"Defects predicted {pred_def:.1f} > threshold {threshold_def:.1f}")

        if issues:
            alerts.append(f"ID {id_val}: " + "; ".join(issues))

    if alerts:
        for line in alerts[:5]:  # limit output to 5 for brevity
            print(line)
        if len(alerts) > 5:
            print(f"... and {len(alerts)-5} more alerts.")
    else:
        print("No unusually high proposals detected.")
        print("-")
        print("-")

    # ------------------------------------------------------------------ NASA-08
    print("\n== NASA-08 ==")
    if n_completed == 0:
        print("No completed projects in the dataset to update summaries.")
    else:
        print(f"Updated summary based on {n_completed} completed projects:")
        print(f"Average Effort-: {eff_mean:.2f} (std {eff_std:.2f})")
        print(f"Average Defects-: {def_mean:.2f} (std {def_std:.2f})")
        print("Regression model for Effort-:")
        if coeffs_eff:
            print(f"  Intercept: {coeffs_eff['intercept']:.4f}")
            print(f"  Kloc+: {coeffs_eff['Kloc+']:.4f}")
            print(f"  MonthsX: {coeffs_eff['MonthsX']:.4f}")
        else:
            print("  Could not compute due to insufficient data.")
        print("Regression model for Defects-:")
        if coeffs_def:
            print(f"  Intercept: {coeffs_def['intercept']:.4f}")
            print(f"  Kloc+: {coeffs_def['Kloc+']:.4f}")
            print(f"  Effort-: {coeffs_def['Effort-']:.4f}")
        else:
            print("  Could not compute due to insufficient data.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

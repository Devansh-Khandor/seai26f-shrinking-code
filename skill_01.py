#!/usr/bin/env python3
"""
Small data‑analysis script for the Auto93 dataset.

This script implements two client requests:

    CAR-01 – Show summary statistics for vehicle weight, fuel efficiency, and acceleration.
    CAR-02 – Rank vehicles on a composite score based on low weight, high mpg, and fast
             acceleration, and list the top 3.

Usage:
    python skill_01.py <path_to_auto93.csv>
"""

import csv
import sys
import statistics
from typing import List, Dict, Any, Optional, Tuple

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> List[Dict[str, Any]]:
    """
    Read the CSV file and return a list of dicts with numeric values converted.
    Missing numeric values are represented as None.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            cleaned: Dict[str, Any] = {}
            for k, v in rec.items():
                if v == '' or v is None:
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = None
            rows.append(cleaned)
    return rows

def compute_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """
    Compute mean, min, and max for a numeric key across all rows.
    Ignores rows where the key is None.
    """
    values = [row[key] for row in rows if row[key] is not None]
    return {
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values)
    }

def compute_score(row: Dict[str, Any],
                  weight_bounds: Dict[str, float],
                  mpg_bounds: Dict[str, float],
                  accel_bounds: Dict[str, float]) -> float:
    """
    Compute a composite score for a single vehicle.
    Lower weight, higher mpg, lower acceleration time are better.
    The score is the sum of each metric's normalized value (0..1).
    """
    weight_norm = (weight_bounds['max'] - row['Lbs-']) / (weight_bounds['max'] - weight_bounds['min'])
    mpg_norm = (row['Mpg+'] - mpg_bounds['min']) / (mpg_bounds['max'] - mpg_bounds['min'])
    accel_norm = (accel_bounds['max'] - row['Acc+']) / (accel_bounds['max'] - accel_bounds['min'])
    return weight_norm + mpg_norm + accel_norm

def format_float(value: float) -> str:
    """Return a string with up to 2 decimal places."""
    return f"{value:.2f}"

# --------------------------------------------------------------------------- #
# Linear regression helpers
# --------------------------------------------------------------------------- #

def build_design_matrix(rows: List[Dict[str, Any]],
                        feature_keys: List[str]) -> Tuple[List[List[float]], List[float]]:
    """
    Build the design matrix X (with intercept) and target vector y
    from the provided rows and feature keys.
    Rows with missing values for any of the features or target are skipped.
    """
    X: List[List[float]] = []
    y: List[float] = []
    for row in rows:
        if any(row[k] is None for k in feature_keys + ['target']):
            continue
        X.append([1.0] + [row[k] for k in feature_keys])  # intercept
        y.append(row['target'])
    return X, y

def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    result = [[sum(a * b for a, b in zip(row_a, col_b))
               for col_b in zip(*B)] for row_a in A]
    return result

def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve Ax = b for x using Gaussian elimination with partial pivoting.
    A must be a square matrix.
    """
    n = len(A)
    # Create augmented matrix
    M = [row[:] + [b_val] for row, b_val in zip(A, b)]
    # Forward elimination
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
        # Eliminate
        for j in range(i + 1, n):
            factor = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

def compute_regression_parameters(rows: List[Dict[str, Any]],
                                   feature_keys: List[str],
                                   target_key: str) -> Tuple[List[float], float]:
    """
    Compute regression coefficients for predicting target_key from feature_keys.
    Returns the coefficient vector (including intercept) and RMSE on training data.
    """
    # Prepare rows with target
    for row in rows:
        row['target'] = row[target_key]
    X, y = build_design_matrix(rows, feature_keys)
    if not X:
        raise ValueError("No training data available")
    XT = transpose(X)
    XTX = mat_mul(XT, X)
    XTy = [sum(row[i] * y_val for row, y_val in zip(XT, y)) for i in range(len(XT))]
    beta = solve_linear_system(XTX, XTy)
    # Compute residuals for RMSE
    residuals = []
    for xi, yi in zip(X, y):
        pred = sum(b * x for b, x in zip(beta, xi))
        residuals.append(yi - pred)
    rmse = (sum(r ** 2 for r in residuals) / len(residuals)) ** 0.5
    return beta, rmse

def predict(row: Dict[str, Any],
            beta: List[float],
            feature_keys: List[str]) -> Optional[float]:
    """
    Predict target value for a single row using coefficients beta.
    Returns None if any required feature is missing.
    """
    if any(row[k] is None for k in feature_keys):
        return None
    features = [1.0] + [row[k] for k in feature_keys]
    return sum(b * f for b, f in zip(beta, features))

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # CAR-01: Summary statistics
    # ----------------------------------------------------------------------- #
    print("== CAR-01 ==")
    weight_stats = compute_stats(rows, 'Lbs-')
    mpg_stats = compute_stats(rows, 'Mpg+')
    accel_stats = compute_stats(rows, 'Acc+')

    print(f"Weight (lbs): mean={format_float(weight_stats['mean'])}, "
          f"min={format_float(weight_stats['min'])}, "
          f"max={format_float(weight_stats['max'])}")

    print(f"Fuel efficiency (mpg): mean={format_float(mpg_stats['mean'])}, "
          f"min={format_float(mpg_stats['min'])}, "
          f"max={format_float(mpg_stats['max'])}")

    print(f"Acceleration (s): mean={format_float(accel_stats['mean'])}, "
          f"min={format_float(accel_stats['min'])}, "
          f"max={format_float(accel_stats['max'])}")

    # ----------------------------------------------------------------------- #
    # CAR-02: Composite ranking
    # ----------------------------------------------------------------------- #
    print("\n== CAR-02 ==")

    # Precompute bounds for normalization
    weight_bounds = {'min': min(row['Lbs-'] for row in rows if row['Lbs-'] is not None),
                     'max': max(row['Lbs-'] for row in rows if row['Lbs-'] is not None)}
    mpg_bounds = {'min': min(row['Mpg+'] for row in rows if row['Mpg+'] is not None),
                  'max': max(row['Mpg+'] for row in rows if row['Mpg+'] is not None)}
    accel_bounds = {'min': min(row['Acc+'] for row in rows if row['Acc+'] is not None),
                    'max': max(row['Acc+'] for row in rows if row['Acc+'] is not None)}

    # Compute scores
    for row in rows:
        if None in (row['Lbs-'], row['Mpg+'], row['Acc+']):
            continue
        row['score'] = compute_score(row, weight_bounds, mpg_bounds, accel_bounds)

    # Sort descending by score
    top_rows = sorted((r for r in rows if 'score' in r),
                      key=lambda r: r['score'], reverse=True)[:3]

    for i, v in enumerate(top_rows, start=1):
        print(f"{i}. Vehicle: Clndrs={int(v['Clndrs'])}, Volume={int(v['Volume'])}, "
              f"HpX={int(v['HpX'])}, Model={int(v['Model'])}, origin={int(v['origin'])}, "
              f"Lbs-={int(v['Lbs-'])}, Acc+= {format_float(v['Acc+'])}, "
              f"Mpg+= {int(v['Mpg+'])}, score={format_float(v['score'])}")

if __name__ == "__main__":
    main()

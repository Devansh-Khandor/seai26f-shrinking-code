#!/usr/bin/env python3
"""
Skill 10: Evaluation of vehicle selection system and performance estimation quality.

CAR-19: Compare the system’s top‑N vehicle selection against a random selection
        under the same evaluation budget, reporting the mean overall score
        difference.

CAR-20: Evaluate the accuracy of our regression‑based MPG prediction against
        a simple historical‑average estimator using leave‑one‑out cross‑validation,
        reporting RMSE for both methods and the percentage improvement.

The script is self‑contained and uses only the Python 3 standard library.
"""

import csv
import sys
import random
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
NUMERIC_KEYS = [
    'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
    'Lbs-', 'Acc+', 'Mpg+'
]

def read_data(csv_path):
    """Read CSV and convert numeric columns to float or None."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in NUMERIC_KEYS:
                try:
                    val = row.get(k, '')
                    row[k] = float(val) if val != '' else None
                except (ValueError, KeyError):
                    row[k] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Linear regression (normal equations)
# ----------------------------------------------------------------------
def transpose(A):
    return [list(col) for col in zip(*A)]

def mat_mult(A, B):
    """Multiply matrices A (m x n) and B (n x p)."""
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            s = 0.0
            for k in range(len(A[0])):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result

def mat_inv(A):
    """Inverse of a square matrix via Gaussian elimination."""
    n = len(A)
    AM = [A[i][:] + [float(i == j) for j in range(n)] for i in range(n)]
    for i in range(n):
        pivot = AM[i][i]
        if abs(pivot) < 1e-12:
            # Swap with a lower row that has a non‑zero pivot
            for r in range(i+1, n):
                if abs(AM[r][i]) > 1e-12:
                    AM[i], AM[r] = AM[r], AM[i]
                    pivot = AM[i][i]
                    break
            else:
                raise ValueError("Matrix not invertible")
        for j in range(2*n):
            AM[i][j] /= pivot
        for r in range(n):
            if r == i:
                continue
            factor = AM[r][i]
            for j in range(2*n):
                AM[r][j] -= factor * AM[i][j]
    return [row[n:] for row in AM]

def mat_vec_mult(A, v):
    """Multiply matrix A (m x n) with vector v (n)."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def solve_linear_system(A, b):
    """Solve A x = b."""
    return mat_vec_mult(mat_inv(A), b)

def linear_regression(X, y):
    """Compute coefficients β for Xβ = y using normal equations."""
    X_bias = [row + [1.0] for row in X]
    XTX = mat_mult(transpose(X_bias), X_bias)
    XTy = mat_vec_mult(transpose(X_bias), y)
    beta = solve_linear_system(XTX, XTy)
    return beta

def predict(beta, features):
    """Predict target from beta and features (no bias)."""
    return sum(beta[i] * features[i] for i in range(len(features))) + beta[-1]

# ----------------------------------------------------------------------
# Overall score calculation (same as previous skills)
# ----------------------------------------------------------------------
def compute_score_stats(rows):
    """Return min/max values for weight, mpg and acceleration."""
    weights = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    mpgs   = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    accs   = [r['Acc+'] for r in rows if r['Acc+'] is not None]
    return {
        'wmin': min(weights) if weights else 0.0,
        'wmax': max(weights) if weights else 0.0,
        'mmin': min(mpgs) if mpgs else 0.0,
        'mmax': max(mpgs) if mpgs else 0.0,
        'amin': min(accs) if accs else 0.0,
        'amax': max(accs) if accs else 0.0
    }

def overall_score(row, stats):
    """Return score in range 0–3 or None if any component is missing."""
    w, m, a = row.get('Lbs-'), row.get('Mpg+'), row.get('Acc+')
    if w is None or m is None or a is None:
        return None
    w_score = (stats['wmax'] - w) / (stats['wmax'] - stats['wmin']) if stats['wmax'] != stats['wmin'] else 1.0
    m_score = (m - stats['mmin']) / (stats['mmax'] - stats['mmin']) if stats['mmax'] != stats['mmin'] else 1.0
    a_score = (stats['amax'] - a) / (stats['amax'] - stats['amin']) if stats['amax'] != stats['amin'] else 1.0
    return w_score + m_score + a_score

# ----------------------------------------------------------------------
# CAR-19: selection vs random
# ----------------------------------------------------------------------
def car_19(rows, stats, budget=5):
    """Compare system selection against random selection under the same budget."""
    print("== CAR-19 ==")

    # Gather rows that can be scored
    scored_rows = [(idx + 1, r, overall_score(r, stats)) for idx, r in enumerate(rows) if overall_score(r, stats) is not None]
    if not scored_rows:
        print("No vehicles with complete data to score.")
        return

    # System selection: top‑N by score
    scored_rows.sort(key=lambda x: x[2], reverse=True)
    top_n = scored_rows[:budget]
    avg_system = sum(r[2] for r in top_n) / len(top_n)

    # Random selection: sample N unique indices
    random.seed(0)  # reproducible
    random_indices = random.sample(range(len(scored_rows)), min(budget, len(scored_rows)))
    random_scores = [scored_rows[i][2] for i in random_indices]
    avg_random = sum(random_scores) / len(random_scores)

    diff = avg_system - avg_random

    print(f"Evaluation budget (N): {budget}")
    print(f"Mean score of system’s top {budget} vehicles: {avg_system:.3f}")
    print(f"Mean score of {budget} random vehicles:     {avg_random:.3f}")
    print(f"Score improvement: {diff:.3f} points per vehicle")
    print(f"Percentage improvement: {100 * diff / avg_random:.1f}%")

# ----------------------------------------------------------------------
# CAR-20: regression vs average
# ----------------------------------------------------------------------
def car_20(rows):
    """Compare regression‑based MPG prediction to historical‑average estimator."""
    print("== CAR-20 ==")

    # Prepare data with available MPG
    X, y = [], []
    for r in rows:
        if r['Mpg+'] is not None and r['Lbs-'] is not None and r['Acc+'] is not None:
            X.append([r['Lbs-'], r['Acc+']])
            y.append(r['Mpg+'])
    if len(X) < 2:
        print("Insufficient data to build regression.")
        return

    # Historical average
    mean_mpg = sum(y) / len(y)

    # Leave‑one‑out regression
    errors_reg = []
    errors_avg = []

    for i in range(len(X)):
        # Train on all except i
        X_train = X[:i] + X[i+1:]
        y_train = y[:i] + y[i+1:]

        beta = linear_regression(X_train, y_train)
        pred = predict(beta, X[i])

        errors_reg.append((pred - y[i]) ** 2)
        errors_avg.append((mean_mpg - y[i]) ** 2)

    rmse_reg = math.sqrt(sum(errors_reg) / len(errors_reg))
    rmse_avg = math.sqrt(sum(errors_avg) / len(errors_avg))
    improvement = ((rmse_avg - rmse_reg) / rmse_avg) * 100.0 if rmse_avg != 0 else 0.0

    print(f"Number of rows with MPG: {len(X)}")
    print(f"Regression RMSE: {rmse_reg:.3f}")
    print(f"Historical‑average RMSE: {rmse_avg:.3f}")
    print(f"Improvement over average: {improvement:.1f}%")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    stats = compute_score_stats(rows)

    car_19(rows, stats, budget=5)
    print()
    car_20(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

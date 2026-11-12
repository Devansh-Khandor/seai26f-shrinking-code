#!/usr/bin/env python3
"""
skill_10.py

Implements:
  * APACHE-19: Compare a model‑guided configuration selection strategy against
    random selection under a fixed benchmarking budget, and report the
    best‑and‑average performance discovered by each.
  * APACHE-20: Compare predictions from a linear regression model against
    a simple historical‑average estimator on unseen configurations,
    reporting the mean absolute error (MAE) of both approaches.

Usage:
    python skill_10.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import random
import math
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, int]], List[str]]:
    """Read the CSV and return list of row dictionaries and field names."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, int]] = []
        for row in reader:
            parsed: Dict[str, int] = {}
            try:
                for k, v in row.items():
                    parsed[k] = int(v)
                rows.append(parsed)
            except ValueError:
                # skip malformed rows
                continue
        return rows, reader.fieldnames


# ----------------------------------------------------------------------
# Linear regression helpers (same as previous skills)
# ----------------------------------------------------------------------
def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # Pivot selection
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Singular or ill‑conditioned matrix")
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
        pivot = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= pivot
        for i in range(k + 1, n):
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x


def normal_equation_solve(X: List[List[float]], y: List[float]) -> List[float]:
    n = len(X)
    p = len(X[0])
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * y[i]
            for k in range(p):
                XtX[j][k] += X[i][j] * X[i][k]
    return gaussian_elimination(XtX, Xty)


def build_design_matrix(rows: List[Dict[str, int]], feature_keys: List[str]
                       ) -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []
    y: List[float] = []
    for r in rows:
        X.append([1] + [r[k] for k in feature_keys])
        y.append(r['Performance-'])
    return X, y


def predict_row(row: Dict[str, int], beta: List[float], feature_keys: List[str]) -> float:
    pred = beta[0]
    for idx, k in enumerate(feature_keys, start=1):
        pred += beta[idx] * row[k]
    return pred


# ----------------------------------------------------------------------
# APACHE-19: Model‑guided vs random selection
# ----------------------------------------------------------------------
def evaluate_selection(rows: List[Dict[str, int]], beta: List[float],
                       feature_keys: List[str], budget: int
                      ) -> Tuple[float, float, float]:
    """
    Return best performance, mean performance, and average predicted performance
    for the selected configurations.
    """
    selected = rows[:budget]
    best = max(r['Performance-'] for r in selected)
    mean = sum(r['Performance-'] for r in selected) / len(selected)
    pred_mean = sum(predict_row(r, beta, feature_keys) for r in selected) / len(selected)
    return best, mean, pred_mean


def apache_19(rows: List[Dict[str, int]]) -> List[str]:
    """Run APACHE‑19 and return a list of output strings."""
    # Determine feature keys
    perf_key = 'Performance-'
    feature_keys = [k for k in rows[0].keys() if k != perf_key]

    # Fit regression on all data (used only for prediction)
    X, y = build_design_matrix(rows, feature_keys)
    beta = normal_equation_solve(X, y)

    # Budget: 10% of total or at least 10
    total = len(rows)
    budget = max(10, total // 10)

    # ---------------- model‑guided selection ----------------
    # Sort by predicted performance descending
    sorted_by_pred = sorted(rows, key=lambda r: predict_row(r, beta, feature_keys), reverse=True)
    best_model, mean_model, pred_mean_model = evaluate_selection(
        sorted_by_pred, beta, feature_keys, budget
    )

    # ---------------- random selection ----------------
    random_rows = random.sample(rows, budget)
    best_rand, mean_rand, pred_mean_rand = evaluate_selection(
        random_rows, beta, feature_keys, budget
    )

    # ---------------- output ----------------
    out = [f"Budget of {budget} configurations"]
    out.append(f"Model‑guided best performance: {best_model}")
    out.append(f"Model‑guided mean performance: {mean_model:.2f}")
    out.append(f"Model‑guided average predicted performance: {pred_mean_model:.2f}")
    out.append("")
    out.append(f"Random best performance: {best_rand}")
    out.append(f"Random mean performance: {mean_rand:.2f}")
    out.append(f"Random average predicted performance: {pred_mean_rand:.2f}")
    out.append("")
    if best_model > best_rand:
        out.append("Model‑guided selection found a better best performance.")
    else:
        out.append("Random selection matched or beat model‑guided best.")
    return out


# ----------------------------------------------------------------------
# APACHE-20: Prediction vs baseline
# ----------------------------------------------------------------------
def apache_20(rows: List[Dict[str, int]]) -> List[str]:
    """Run APACHE‑20 and return a list of output strings."""
    perf_key = 'Performance-'
    feature_keys = [k for k in rows[0].keys() if k != perf_key]

    # Shuffle and split 70/30
    indices = list(range(len(rows)))
    random.shuffle(indices)
    split = int(len(rows) * 0.7)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    # Train regression on training set
    X_train, y_train = build_design_matrix(train_rows, feature_keys)
    beta = normal_equation_solve(X_train, y_train)

    # Baseline: mean performance from training set
    baseline_value = sum(y_train) / len(y_train)

    # Evaluate on test set
    mae_model = sum(
        abs(predict_row(r, beta, feature_keys) - r[perf_key]) for r in test_rows
    ) / len(test_rows)
    mae_baseline = sum(abs(baseline_value - r[perf_key]) for r in test_rows) / len(test_rows)

    out = []
    out.append(f"Test set size: {len(test_rows)}")
    out.append(f"Training set size: {len(train_rows)}")
    out.append(f"Mean absolute error (regression): {mae_model:.2f}")
    out.append(f"Mean absolute error (historical average): {mae_baseline:.2f}")
    if mae_model < mae_baseline:
        out.append("Regression model provides a meaningful improvement on unseen configurations.")
    else:
        out.append("Regression model does not outperform the baseline.")
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str):
    rows, _ = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # APACHE-19
    print("== APACHE-19 ==")
    for line in apache_19(rows):
        print(line)

    # APACHE-20
    print("\n== APACHE-20 ==")
    for line in apache_20(rows):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

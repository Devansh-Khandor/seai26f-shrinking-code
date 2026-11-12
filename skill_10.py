#!/usr/bin/env python3
"""
Skill 10: Performance comparison of vehicle‑selection and prediction systems
============================================================================

This script implements two reports:

* CAR-19 – Compare the quality of the vehicle‑selection system to a purely
           random choice of vehicles when both approaches have the same
           evaluation budget.

* CAR-20 – Evaluate the accuracy of the linear‑regression based
           performance‑prediction models against a naïve historical‑average
           baseline, and show whether the new system gives a meaningful
           improvement on unseen records.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import sys
import math
import random
from statistics import mean

# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """Return a list of rows with numeric values or None for missing data."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for key in (
                "Clndrs",
                "Volume",
                "HpX",
                "Model",
                "origin",
                "Lbs-",
                "Acc+",
                "Mpg+",
            ):
                try:
                    parsed[key] = float(r[key])
                except (KeyError, ValueError, TypeError):
                    parsed[key] = None
            rows.append(parsed)
    return rows

# ----------------------------------------------------------------------
# Matrix utilities (re‑used from skill_02)
# ----------------------------------------------------------------------
def transpose(m):
    return [list(row) for row in zip(*m)]

def matmul(a, b):
    result = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(a[0])):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result

def invert_matrix(m):
    n = len(m)
    a = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(m)]
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(a[r][i]) > abs(a[pivot][i]):
                pivot = r
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("Singular matrix")
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
        pv = a[i][i]
        for j in range(2 * n):
            a[i][j] /= pv
        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            for j in range(2 * n):
                a[r][j] -= factor * a[i][j]
    return [row[n:] for row in a]

# ----------------------------------------------------------------------
# Linear regression helpers
# ----------------------------------------------------------------------
def build_design_matrix(rows, feature_keys, target_key=None):
    X = []
    y = []
    for r in rows:
        if target_key is not None and r[target_key] is None:
            continue
        row = [1.0]  # intercept
        for k in feature_keys:
            row.append(r.get(k, 0.0) or 0.0)
        X.append(row)
        if target_key is not None:
            y.append([r[target_key]])
    return X, y

def linear_regression(X, y):
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    try:
        XtX_inv = invert_matrix(XtX)
    except ValueError:
        raise RuntimeError("Regression matrix is singular.")
    Xty = matmul(Xt, y)
    beta_mat = matmul(XtX_inv, Xty)
    return [row[0] for row in beta_mat]

def predict(X, beta):
    return matmul(X, [[b] for b in beta])

# ----------------------------------------------------------------------
# Scoring utilities (used by CAR‑19)
# ----------------------------------------------------------------------
def normalize(val, low, high, reverse=False):
    if high == low or val is None:
        return 0.5
    norm = (val - low) / (high - low)
    return 1.0 - norm if reverse else norm

def compute_score(row, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max):
    nw = normalize(row["Lbs-"], w_min, w_max, reverse=True)
    nm = normalize(row["Mpg+"], mpg_min, mpg_max)
    na = normalize(row["Acc+"], acc_min, acc_max, reverse=True)
    return (nw + nm + na) / 3.0

# ----------------------------------------------------------------------
# CAR-19: Selection comparison
# ----------------------------------------------------------------------
def car_19(rows, budget=5):
    """Compare our scoring system with random selection."""
    # Filter for complete records
    good = [r for r in rows if None not in (r["Lbs-"], r["Acc+"], r["Mpg+"])]
    if not good:
        return ["No complete data for CAR‑19."]
    # Compute min/max for score normalization
    w_min, w_max = min(r["Lbs-"] for r in good), max(r["Lbs-"] for r in good)
    mpg_min, mpg_max = min(r["Mpg+"] for r in good), max(r["Mpg+"] for r in good)
    acc_min, acc_max = min(r["Acc+"] for r in good), max(r["Acc+"] for r in good)

    # Scores for all vehicles
    scored = [(idx, r, compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max))
              for idx, r in enumerate(good, start=1)]
    scored.sort(key=lambda x: x[2], reverse=True)

    # Our best choices
    our_top = scored[:budget]
    our_avg = mean(s[2] for s in our_top)

    # Random picks – deterministic via seed
    random.seed(42)
    random_indices = random.sample(range(len(good)), budget)
    random_top = [scored[i] for i in random_indices]
    random_avg = mean(s[2] for s in random_top)

    diff = our_avg - random_avg
    diff_pct = (diff / random_avg) * 100 if random_avg != 0 else float('inf')

    out = [
        f"Evaluation budget: {budget} vehicles",
        f"Our system average score: {our_avg:.3f}",
        f"Random selection average score: {random_avg:.3f}",
        f"Score improvement: {diff:.3f} ({diff_pct:+.1f}%)",
        "",
        "Top vehicles chosen by our system:",
    ]
    for rank, (idx, r, scr) in enumerate(our_top, start=1):
        out.append(f"  {rank}. Vehicle #{idx:>3} – Score={scr:.3f}")
    out.append("")
    out.append("Randomly chosen vehicles:")
    for rank, (idx, r, scr) in enumerate(random_top, start=1):
        out.append(f"  {rank}. Vehicle #{idx:>3} – Score={scr:.3f}")
    return out

# ----------------------------------------------------------------------
# CAR-20: Prediction accuracy comparison
# ----------------------------------------------------------------------
def car_20(rows):
    """Compare linear‑regression predictions with a historical‑average baseline."""
    # Split into training (70%) and testing (30%)
    n_total = len(rows)
    if n_total < 10:
        return ["Not enough data to perform train/test split for CAR‑20."]

    indices = list(range(n_total))
    random.seed(123)  # deterministic split
    random.shuffle(indices)
    split = int(0.7 * n_total)
    train_idx, test_idx = indices[:split], indices[split:]

    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]

    out = [
        f"Training on {len(train)} records, testing on {len(test)} records."
    ]

    # ---------- MPG prediction ----------
    mpg_features = ["Lbs-", "HpX", "Acc+"]
    X_train, y_train = build_design_matrix(train, mpg_features, target_key="Mpg+")
    X_test, _ = build_design_matrix(test, mpg_features)
    if not X_train:
        out.append("Insufficient MPG training data.")
    else:
        beta_mpg = linear_regression(X_train, y_train)
        pred_train = predict(X_train, beta_mpg)
        pred_test = predict(X_test, beta_mpg)

        # MAE of model
        mae_model = mean(abs(t[0] - p[0]) for t, p in zip(y_train, pred_train))
        mae_test = mean(abs(t[0] - p[0]) for t, p in zip(
            [r["Mpg+"] for r in test], pred_test))

        # Baseline MAE (mean of training MPG)
        mean_mpg_train = mean(r["Mpg+"] for r in train)
        mae_baseline = mean(abs(mean_mpg_train - r["Mpg+"]) for r in test)

        improvement = ((mae_baseline - mae_test) / mae_baseline) * 100 if mae_baseline else float('inf')

        out.append(f"MPG prediction MAE (model): {mae_test:.2f}")
        out.append(f"MPG prediction MAE (baseline): {mae_baseline:.2f}")
        out.append(f"Improvement over baseline: {improvement:+.1f}%")

    # ---------- Weight prediction ----------
    weight_features = ["HpX", "Acc+", "Volume", "Clndrs", "Model", "origin"]
    X_train_w, y_train_w = build_design_matrix(train, weight_features, target_key="Lbs-")
    X_test_w, _ = build_design_matrix(test, weight_features)
    if not X_train_w:
        out.append("Insufficient weight training data.")
    else:
        beta_w = linear_regression(X_train_w, y_train_w)
        pred_train_w = predict(X_train_w, beta_w)
        pred_test_w = predict(X_test_w, beta_w)

        mae_model_w = mean(abs(t[0] - p[0]) for t, p in zip(y_train_w, pred_train_w))
        mae_test_w = mean(abs(t[0] - p[0]) for t, p in zip(
            [r["Lbs-"] for r in test], pred_test_w))

        mean_w_train = mean(r["Lbs-"] for r in train)
        mae_baseline_w = mean(abs(mean_w_train - r["Lbs-"]) for r in test)

        improvement_w = ((mae_baseline_w - mae_test_w) / mae_baseline_w) * 100 if mae_baseline_w else float('inf')

        out.append(f"Weight prediction MAE (model): {mae_test_w:.2f} lbs")
        out.append(f"Weight prediction MAE (baseline): {mae_baseline_w:.2f} lbs")
        out.append(f"Improvement over baseline: {improvement_w:+.1f}%")

    return out

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # CAR‑19
    print("== CAR-19 ==")
    for line in car_19(rows, budget=5):
        print(line)

    # CAR‑20
    print("\n== CAR-20 ==")
    for line in car_20(rows):
        print(line)

if __name__ == "__main__":
    main()

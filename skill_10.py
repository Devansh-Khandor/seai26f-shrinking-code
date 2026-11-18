#!/usr/bin/env python3
"""
skill_10.py

Implements CAR-19 and CAR-20 for the auto93 dataset.
"""

import csv
import json
import sys
import math
import random
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def read_data(path: str) -> List[Dict[str, Any]]:
    """Read CSV into list of dicts, converting numbers to float and
    converting '?' or '' to None."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v == '?' or v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def goal_columns(rows: List[Dict[str, Any]]) -> Set[str]:
    """Return set of column names that are goals."""
    if not rows:
        return set()
    return {k for k in rows[0] if k.endswith('-') or k.endswith('+')}


# ------------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------------
def normalized_value(val: float, mn: float, mx: float, minimize: bool) -> float:
    if mx == mn:
        return 0.0
    return (mx - val) / (mx - mn) if minimize else (val - mn) / (mx - mn)


def compute_mins_maxs(rows: List[Dict[str, Any]], cols: Set[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for col in cols:
        vals = [r[col] for r in rows if isinstance(r[col], float)]
        mins[col] = min(vals) if vals else 0.0
        maxs[col] = max(vals) if vals else 0.0
    return mins, maxs


def performance_score(row: Dict[str, Any],
                      goals: Set[str],
                      mins: Dict[str, float],
                      maxs: Dict[str, float]) -> float | None:
    if any(row[g] is None for g in goals):
        return None
    score = 0.0
    for g in goals:
        val = row[g]
        mn, mx = mins[g], maxs[g]
        score += normalized_value(val, mn, mx, minimize=g.endswith('-'))
    return score


# ------------------------------------------------------------------
# Linear regression utilities for CAR-20
# ------------------------------------------------------------------
def build_design_matrix(rows: List[Dict[str, Any]],
                        predictors: List[str]) -> List[List[float]]:
    X = []
    for row in rows:
        vec = [row[p] for p in predictors]
        X.append(vec)
    return X


def mat_transpose(m: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*m)]


def mat_mult(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    m, n = len(A), len(A[0])
    p = len(B[0])
    res = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            res[i][j] = s
    return res


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        # pivot
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot][i]) < 1e-12:
            raise ValueError("Singular matrix")
        M[i], M[pivot] = M[pivot], M[i]
        # normalize
        piv = M[i][i]
        for k in range(i, n + 1):
            M[i][k] /= piv
        # eliminate
        for j in range(n):
            if j == i:
                continue
            factor = M[j][i]
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    return [M[i][n] for i in range(n)]


def normal_equation_solution(X: List[List[float]], y: List[float]) -> List[float]:
    Xt = mat_transpose(X)
    XtX = mat_mult(Xt, X)
    Xty = [sum(Xt[i][j] * y[j] for j in range(len(y))) for i in range(len(Xt))]
    beta = solve_linear(XtX, Xty)
    return beta


def mean_abs_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


# ------------------------------------------------------------------
# CAR-19: evaluation budget comparison
# ------------------------------------------------------------------
def car_19(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    goals = goal_columns(rows)
    mins, maxs = compute_mins_maxs(rows, goals)

    # --------- Our selection ----------
    # pick 30 rows with lowest Lbs- (if available), else first rows
    goal_list = sorted(goals)
    # find rows with complete goals
    complete_rows = [row for row in rows if all(row[g] is not None for g in goals)]
    if complete_rows:
        # sort by Lbs- ascending
        complete_rows.sort(key=lambda r: r["Lbs-"])
        ours_rows = complete_rows[:30]
    else:
        ours_rows = rows[:30]

    ours_best_row = None
    ours_best_score = -1.0
    for idx, row in enumerate(rows, start=1):
        if row in ours_rows:
            sc = performance_score(row, goals, mins, maxs)
            if sc is not None and (sc > ours_best_score or (sc == ours_best_score and idx < ours_best_row)):
                ours_best_score = sc
                ours_best_row = idx

    # --------- Random selection ----------
    random.seed(0)
    repeats = 20
    ours_better = 0
    random_better = 0
    ours_best_list = []
    random_best_list = []
    all_indices = list(range(1, len(rows) + 1))
    for _ in range(repeats):
        rand_rows = random.sample(all_indices, min(30, len(rows)))
        rand_best_row = None
        rand_best_score = -1.0
        for idx in rand_rows:
            row = rows[idx - 1]
            sc = performance_score(row, goals, mins, maxs)
            if sc is not None and (sc > rand_best_score or (sc == rand_best_score and idx < rand_best_row)):
                rand_best_score = sc
                rand_best_row = idx
        ours_best_list.append(ours_best_row)
        random_best_list.append(rand_best_row)
        if ours_best_row is not None and rand_best_row is not None:
            if ours_best_score > rand_best_score:
                ours_better += 1
            elif ours_best_score < rand_best_score:
                random_better += 1
        elif ours_best_row is not None:
            ours_better += 1
        elif rand_best_row is not None:
            random_better += 1
        # ties or both None: no count

    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": 30,
        "repeats": repeats,
        "ours": ours_best_list,
        "random": random_best_list,
        "verdict": verdict
    }


# ------------------------------------------------------------------
# CAR-20: model vs baseline
# ------------------------------------------------------------------
def car_20(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Identify held‑out rows
    total = len(rows)
    heldout_indices = [i + 1 for i in range(total) if (i + 1) % 5 == 0]
    heldout_set = set(heldout_indices)

    # Train on non‑held‑out rows
    predictors = ["Clndrs", "Volume", "Model", "origin", "Lbs-", "Acc+"]
    train_rows = [row for idx, row in enumerate(rows, start=1)
                  if idx not in heldout_set and
                  row["Mpg+"] is not None and
                  all(row[p] is not None for p in predictors)]
    if not train_rows:
        # fallback: use mean of all non‑held‑out rows
        train_rows = [row for idx, row in enumerate(rows, start=1)
                      if idx not in heldout_set and row["Mpg+"] is not None]

    # Means for imputation
    predictor_means = {}
    for p in predictors:
        vals = [r[p] for r in train_rows if r[p] is not None]
        predictor_means[p] = sum(vals) / len(vals) if vals else 0.0

    # Build matrix
    X_train = build_design_matrix(train_rows, predictors)
    y_train = [r["Mpg+"] for r in train_rows]
    beta = normal_equation_solution(X_train, y_train)

    # Predictions for held‑out rows
    predictions: Dict[str, float] = {}
    y_true_list: List[float] = []
    y_pred_list: List[float] = []

    for idx in heldout_indices:
        row = rows[idx - 1]
        # skip if true value missing
        if row["Mpg+"] is None:
            continue
        vec = [row[p] if row[p] is not None else predictor_means[p] for p in predictors]
        est = sum(v * b for v, b in zip(vec, beta))
        predictions[str(idx)] = round(est, 2)
        y_true_list.append(row["Mpg+"])
        y_pred_list.append(est)

    mae_model = round(mean_abs_error(y_true_list, y_pred_list), 4) if y_true_list else 0.0

    # Baseline mean
    non_held_vals = [row["Mpg+"] for idx, row in enumerate(rows, start=1)
                     if idx not in heldout_set and row["Mpg+"] is not None]
    baseline_mean = sum(non_held_vals) / len(non_held_vals) if non_held_vals else 0.0
    baseline_mean = round(baseline_mean, 2)

    # Baseline MAE
    mae_baseline = round(mean_abs_error(y_true_list, [baseline_mean]*len(y_true_list)), 4) if y_true_list else 0.0

    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)

    # CAR-19
    print("== CAR-19 ==")
    res19 = car_19(rows)
    # Provide at least three lines of output
    print(f"Budget per evaluation: {res19['budget']}")
    print(f"Repeats: {res19['repeats']}")
    print(f"Our best row (first repeat): {res19['ours'][0]}")
    print(f"Random best row (first repeat): {res19['random'][0]}")
    print(f"Counts ours better: {sum(1 for i, r in enumerate(res19['ours']) if r is not None and res19['random'][i] is not None and (r == res19['ours'][0] and r > res19['random'][i])}")
    print(f"Counts random better: {sum(1 for i, r in enumerate(res19['random']) if r is not None and res19['ours'][i] is not None and (r == res19['random'][i] and r > res19['ours'][i])}")
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER CAR-19 {json.dumps(res19, ensure_ascii=False)}")

    # CAR-20
    print("\n== CAR-20 ==")
    res20 = car_20(rows)
    print(f"Baseline Mpg+ mean: {res20['baseline']}")
    print(f"MAE model: {res20['mae_model']}")
    print(f"MAE baseline: {res20['mae_baseline']}")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER CAR-20 {json.dumps(res20, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

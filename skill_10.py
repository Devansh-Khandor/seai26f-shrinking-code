#!/usr/bin/env python3
"""
Skill 10: Evaluation of selection system vs random and model vs baseline.

Usage:
    python skill_10.py <csv_path>
"""

import sys
import csv
import json
import random
import math
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, normalize
except Exception:
    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except Exception:
            return math.nan

    def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
        if max_val == min_val:
            return 1.0
        if maximize:
            return (val - min_val) / (max_val - min_val)
        return (max_val - val) / (max_val - min_val)

# --------------------------------------------------------------------------- #
# Reuse regression helpers from skill_02
# --------------------------------------------------------------------------- #

try:
    from skill_02 import build_design_matrix, compute_regression, predict
except Exception:
    # Minimal reimplementation if skill_02 not available
    def build_design_matrix(rows: List[Dict[str, str]],
                            feature_cols: List[str],
                            target_col: str,
                            missing_ok: bool = False) -> Tuple[List[List[float]], List[float]]:
        X = []
        y = []
        for row in rows:
            vals = []
            missing = False
            for col in feature_cols:
                v = row[col]
                if is_missing(v):
                    missing = True
                    break
                vals.append(to_float(v))
            target_val = row[target_col]
            if is_missing(target_val):
                missing = True
            if missing:
                if missing_ok:
                    X.append([1.0] + [float('nan')] * len(feature_cols))
                    y.append(float('nan'))
                else:
                    continue
            else:
                X.append([1.0] + vals)
                y.append(to_float(target_val))
        return X, y

    def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
        return [list(row) for row in zip(*A)]

    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for k in range(len(B)):
                aik = A[i][k]
                if aik == 0:
                    continue
                for j in range(len(B[0])):
                    result[i][j] += aik * B[k][j]
        return result

    def matrix_vector_multiply(A: List[List[float]], v: List[float]) -> List[float]:
        return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

    def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
        n = len(A)
        AI = [row[:] + [float(i == j) for i in range(n)] for j, row in enumerate(A)]
        for i in range(n):
            pivot = i
            for r in range(i, n):
                if abs(AI[r][i]) > abs(AI[pivot][i]):
                    pivot = r
            if abs(AI[pivot][i]) < 1e-12:
                raise ValueError("Matrix singular")
            AI[i], AI[pivot] = AI[pivot], AI[i]
            div = AI[i][i]
            AI[i] = [x / div for x in AI[i]]
            for r in range(n):
                if r != i:
                    factor = AI[r][i]
                    AI[r] = [AI[r][c] - factor * AI[i][c] for c in range(2 * n)]
        return [row[n:] for row in AI]

    def compute_regression(X: List[List[float]], y: List[float]) -> Tuple[List[float], float]:
        X_clean = []
        y_clean = []
        for xi, yi in zip(X, y):
            if any(math.isnan(val) for val in xi) or math.isnan(yi):
                continue
            X_clean.append(xi)
            y_clean.append(yi)
        n = len(X_clean)
        m = len(X_clean[0])
        Xt = matrix_transpose(X_clean)
        XtX = matrix_multiply(Xt, X_clean)
        Xty = matrix_vector_multiply(Xt, y_clean)
        XtX_inv = matrix_inverse(XtX)
        beta = matrix_vector_multiply(XtX_inv, Xty)
        preds = matrix_vector_multiply(X_clean, beta)
        residuals = [yi - pi for yi, pi in zip(y_clean, preds)]
        rss = sum(r ** 2 for r in residuals)
        sigma = math.sqrt(rss / (n - m)) if (n - m) > 0 else 0.0
        return beta, sigma

    def predict(beta: List[float], feature_row: List[float]) -> float:
        return sum(b * f for b, f in zip(beta, feature_row))

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def compute_goal_min_max(rows: List[Dict[str, str]],
                         goal_cols: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    col_min = {}
    col_max = {}
    for col in goal_cols:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        if vals:
            col_min[col] = min(vals)
            col_max[col] = max(vals)
        else:
            col_min[col] = col_max[col] = 0.0
    return col_min, col_max

def compute_composite_scores(rows: List[Dict[str, str]],
                             goal_cols: Dict[str, str],
                             col_min: Dict[str, float],
                             col_max: Dict[str, float]) -> List[Tuple[int, float]]:
    scores = []
    for idx, row in enumerate(rows, start=1):
        if any(is_missing(row[col]) for col in goal_cols):
            continue
        score = 0.0
        for col, dirn in goal_cols.items():
            v = to_float(row[col])
            score += normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
        scores.append((idx, score))
    return scores

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify goal columns and their direction
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}

    # ---------------- CAR-19 ----------------
    # Budget and repeats
    budget = min(30, len(rows) // 5)
    repeats = 20
    random.seed(0)

    # Pre‑compute composite scores for all rows
    col_min, col_max = compute_goal_min_max(rows, goal_columns)
    all_scores = compute_composite_scores(rows, goal_columns, col_min, col_max)
    score_dict = {idx: score for idx, score in all_scores}

    # Our selection: top budget rows by score (deterministic)
    sorted_all = sorted(all_scores, key=lambda kv: (-kv[1], kv[0]))
    ours_selected = [idx for idx, _ in sorted_all[:budget]]
    ours_best_idx = sorted_all[0][0]
    ours_best_score = sorted_all[0][1]

    # Random selection repeats
    random_bests = []
    for _ in range(repeats):
        sample = random.sample(range(1, len(rows) + 1), budget)
        best_idx = max(sample, key=lambda i: score_dict.get(i, -math.inf))
        random_bests.append(best_idx)

    # Compare
    better_count = 0
    worse_count = 0
    tie_count = 0
    for rnd_idx in random_bests:
        rnd_score = score_dict.get(rnd_idx, -math.inf)
        if ours_best_score > rnd_score:
            better_count += 1
        elif ours_best_score < rnd_score:
            worse_count += 1
        else:
            tie_count += 1

    if better_count > worse_count:
        verdict19 = "ours better"
    elif worse_count > better_count:
        verdict19 = "random better"
    else:
        verdict19 = "no clear difference"

    # Print CAR-19
    print("== CAR-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Ours best row: {ours_best_idx}")
    print(f"Random best rows: {random_bests[:5]} ...")
    print()

    # Machine‑checked answer
    car19_answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": [ours_best_idx] * repeats,
        "random": random_bests,
        "verdict": verdict19
    }
    print(f"ANSWER CAR-19 {json.dumps(car19_answer, separators=(',', ':'))}")

    # ---------------- CAR-20 ----------------
    # Identify held‑out rows (divisible by 5)
    held_out_indices = {i for i in range(1, len(rows) + 1) if i % 5 == 0}
    training_rows = [row for idx, row in enumerate(rows, start=1) if idx not in held_out_indices]
    held_out_rows = [row for idx, row in enumerate(rows, start=1) if idx in held_out_indices]

    target_col = 'Mpg+'
    feature_cols = [c for c in header if c != target_col and not c.endswith('X')]

    # Build regression
    X_train, y_train = build_design_matrix(training_rows, feature_cols, target_col, missing_ok=False)
    beta, sigma = compute_regression(X_train, y_train)

    # Predictions for held‑out rows
    predictions = {}
    mae_model_sum = 0.0
    mae_model_count = 0
    actuals = []
    for idx, row in enumerate(rows, start=1):
        if idx not in held_out_indices:
            continue
        # Check features
        feature_vals = []
        missing = False
        for col in feature_cols:
            v = row[col]
            if is_missing(v):
                missing = True
                break
            feature_vals.append(to_float(v))
        if missing:
            continue
        pred = predict(beta, [1.0] + feature_vals)
        actual = to_float(row[target_col])
        predictions[str(idx)] = round(pred, 4)
        mae_model_sum += abs(pred - actual)
        mae_model_count += 1
        actuals.append(actual)

    mae_model = mae_model_sum / mae_model_count if mae_model_count else float('nan')

    # Baseline: mean of non‑held‑out Mpg+
    baseline_vals = [to_float(row[target_col]) for idx, row in enumerate(rows, start=1) if idx not in held_out_indices and not is_missing(row[target_col])]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else float('nan')
    mae_baseline_sum = sum(abs(v - baseline_mean) for v in baseline_vals)
    mae_baseline = mae_baseline_sum / len(baseline_vals) if baseline_vals else float('nan')

    # Verdict
    if mae_model < mae_baseline:
        verdict20 = "better"
    elif mae_model > mae_baseline:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    # Print CAR-20
    print("== CAR-20 ==")
    print(f"Baseline mean Mpg+: {baseline_mean:.4f}")
    print(f"Model MAE: {mae_model:.4f}, Baseline MAE: {mae_baseline:.4f}")
    print()

    # Machine‑checked answer
    car20_answer = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 4),
        "mae_model": round(mae_model, 4),
        "mae_baseline": round(mae_baseline, 4),
        "verdict": verdict20
    }
    print(f"ANSWER CAR-20 {json.dumps(car20_answer, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

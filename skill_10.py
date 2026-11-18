#!/usr/bin/env python3
"""
skill_10.py: Comparison of vehicle-selection and prediction systems for auto93 dataset.
Usage: python skill_10.py <csv_path>
"""

import json
import random
import sys
from typing import Dict, List, Tuple

import ezr

# ------------------------------------------------------------------
# Read CSV and build table (common helper)

def read_csv(path: str) -> Tuple[List[str], List[List[object]]]:
    """Read CSV and return header and rows as atoms."""
    with open(path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
        rows: List[List[object]] = []
        for line in f:
            if line.strip():
                rows.append([ezr.atom(s) for s in line.split(",")])
    return header, rows


def build_table(header, rows):
    """Create an ezr table from header and rows."""
    return ezr.Tbl([header] + rows)


# ------------------------------------------------------------------
# Helper: overall score for a row (sum of normalized goal values)

def goal_columns(tbl: ezr.TBL) -> List[ezr.COL]:
    return [col for col in tbl.cols.values() if col.goal != 0]


def compute_min_max(tbl: ezr.TBL, goal_cols: List[ezr.COL]) -> Dict[int, Tuple[float, float]]:
    result: Dict[int, Tuple[float, float]] = {}
    for col in goal_cols:
        vals = [r[col.at] for r in tbl.rows if r[col.at] != "?"]
        if vals:
            result[col.at] = (min(vals), max(vals))
        else:
            result[col.at] = (0.0, 0.0)
    return result


def overall_score(tbl: ezr.TBL, row: List[object]) -> float:
    goal_cols = goal_columns(tbl)
    mins_maxs = compute_min_max(tbl, goal_cols)
    score = 0.0
    for col in goal_cols:
        val = row[col.at]
        if val == "?":
            continue
        mn, mx = mins_maxs[col.at]
        if mx == mn:
            comp = 1.0
        else:
            if col.goal == 1:  # maximize
                comp = (val - mn) / (mx - mn)
            else:  # minimize
                comp = (mx - val) / (mx - mn)
        score += comp
    return score


# ------------------------------------------------------------------
# Linear regression helpers (copied from skill_02)

def gauss_jordan(A, b):
    n = len(A)
    for i in range(n):
        A[i].append(b[i])
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
        div = A[col][col]
        for j in range(col, n + 1):
            A[col][j] /= div
        for r in range(n):
            if r != col:
                factor = A[r][col]
                for j in range(col, n + 1):
                    A[r][j] -= factor * A[col][j]
    return [A[i][n] for i in range(n)]


def linear_regression(features, targets):
    n = len(features)
    p = len(features[0]) if n > 0 else 0
    XtX = [[0.0] * (p + 1) for _ in range(p + 1)]
    Xty = [0.0] * (p + 1)
    for i in range(n):
        row = [1.0] + features[i]
        for a in range(p + 1):
            Xty[a] += row[a] * targets[i]
            for b in range(p + 1):
                XtX[a][b] += row[a] * row[b]
    coeffs = gauss_jordan([row[:] for row in XtX], Xty)
    return coeffs


def predict_row(coeffs, feature_vals):
    return coeffs[0] + sum(coeffs[i + 1] * feature_vals[i] for i in range(len(feature_vals)))


# ------------------------------------------------------------------
# CAR-19: Selection comparison

def car_19(tbl: ezr.TBL, repeats: int = 20) -> Dict:
    budget = min(30, len(tbl.rows) // 5)
    random.seed(42)
    ours_best_rows: List[int] = []
    ours_best_scores: List[float] = []
    random_best_rows: List[int] = []
    random_best_scores: List[float] = []

    ranker = ezr.ranker(tbl)  # tree ranker

    all_indices = list(range(len(tbl.rows)))  # 0-based

    for _ in range(repeats):
        # OURS: top budget by ranker
        top_idx = sorted(all_indices, key=lambda i: ranker(tbl.rows[i]), reverse=True)[:budget]
        # find best overall among these
        best_o = max(top_idx, key=lambda i: overall_score(tbl, tbl.rows[i]))
        ours_best_rows.append(best_o + 1)
        ours_best_scores.append(overall_score(tbl, tbl.rows[best_o]))

        # RANDOM: random sample of same size
        rnd_idx = random.sample(all_indices, budget)
        best_r = max(rnd_idx, key=lambda i: overall_score(tbl, tbl.rows[i]))
        random_best_rows.append(best_r + 1)
        random_best_scores.append(overall_score(tbl, tbl.rows[best_r]))

    # Verdict
    avg_ours = sum(ours_best_scores) / repeats
    avg_rand = sum(random_best_scores) / repeats
    eps = 1e-6
    if avg_ours - avg_rand > eps:
        verdict = "ours better"
    elif avg_rand - avg_ours > eps:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }


# ------------------------------------------------------------------
# CAR-20: Prediction comparison

def car_20(tbl: ezr.TBL) -> Dict:
    # Identify goal column Mpg+
    mpg_col = tbl.cols["Mpg+"]
    # Held‑out rows: divisible by 5 (1-based)
    held_out_idx = [i for i, _ in enumerate(tbl.rows) if (i + 1) % 5 == 0]
    train_idx = [i for i in range(len(tbl.rows)) if i not in held_out_idx]

    # Non‑held‑out Mpg+ values for baseline
    mpg_vals = [tbl.rows[i][mpg_col.at] for i in train_idx if tbl.rows[i][mpg_col.at] != "?"]
    baseline = sum(mpg_vals) / len(mpg_vals) if mpg_vals else 0.0

    # Predict using linear regression on all non‑goal columns
    predictors = [col for col in tbl.cols.values() if col.goal == 0]
    predictor_at = [col.at for col in predictors]

    # Build training data
    X_train = []
    y_train = []
    for i in train_idx:
        row = tbl.rows[i]
        if row[mpg_col.at] == "?":
            continue
        feat = []
        skip = False
        for at in predictor_at:
            v = row[at]
            if v == "?":
                # impute with mean of that predictor from training
                # compute means once
                pass
        # We'll compute means first
    # Compute predictor means for imputation
    means = {}
    for at in predictor_at:
        vals = [tbl.rows[i][at] for i in train_idx if tbl.rows[i][at] != "?"]
        means[at] = sum(vals) / len(vals) if vals else 0.0

    # Now build X_train, y_train
    for i in train_idx:
        row = tbl.rows[i]
        if row[mpg_col.at] == "?":
            continue
        feat = [row[at] if row[at] != "?" else means[at] for at in predictor_at]
        X_train.append(feat)
        y_train.append(row[mpg_col.at])

    coeffs = linear_regression(X_train, y_train)

    # Predictions for held‑out rows
    predictions: Dict[str, float] = {}
    mae_sum = 0.0
    mae_cnt = 0
    mae_baseline_sum = 0.0
    mae_baseline_cnt = 0

    for i in held_out_idx:
        row = tbl.rows[i]
        # skip if actual unknown
        actual = row[mpg_col.at]
        if actual == "?":
            continue
        feat = [row[at] if row[at] != "?" else means[at] for at in predictor_at]
        pred = predict_row(coeffs, feat)
        predictions[str(i + 1)] = pred
        mae_sum += abs(pred - actual)
        mae_cnt += 1
        mae_baseline_sum += abs(baseline - actual)
        mae_baseline_cnt += 1

    mae_model = mae_sum / mae_cnt if mae_cnt else 0.0
    mae_baseline = mae_baseline_sum / mae_baseline_cnt if mae_baseline_cnt else 0.0

    eps = 1e-6
    if mae_model + eps < mae_baseline:
        verdict = "better"
    elif mae_baseline + eps < mae_model:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }


# ------------------------------------------------------------------
# Main driver

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # CAR-19
    print("== CAR-19 ==")
    res19 = car_19(tbl)
    print(f"Budget: {res19['budget']}, Repeats: {res19['repeats']}")
    print(f"Best row numbers (ours) in first 5 repeats: {res19['ours'][:5]}")
    print(f"Best row numbers (random) in first 5 repeats: {res19['random'][:5]}")
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER CAR-19 {json.dumps(res19)}")

    # CAR-20
    print("\n== CAR-20 ==")
    res20 = car_20(tbl)
    print(f"Mean baseline Mpg+: {res20['baseline']:.3f}")
    print(f"MAE model: {res20['mae_model']:.3f}, MAE baseline: {res20['mae_baseline']:.3f}")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER CAR-20 {json.dumps(res20)}")


if __name__ == "__main__":
    main()

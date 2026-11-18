#!/usr/bin/env python3
"""
skill_10.py <csv_path>
Implements CAR-19 (selection vs random) and CAR-20 (model vs baseline prediction).
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Common utilities
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def load_rows(csv_path: str) -> Tuple[List[Tuple[int, Dict[str, Optional[float]]]], List[str]]:
    """Return list of (row_number, parsed_dict) and the header list."""
    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for i, row in enumerate(reader, start=1):
            parsed = {k: parse_value(v) for k, v in row.items()}
            rows.append((i, parsed))
    return rows, headers

def goal_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

def is_numeric(col: str) -> bool:
    return not col.endswith('X') and col not in goal_columns(col)  # simple heuristic

def numeric_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if is_numeric(h)]

def has_all_goals(data: Dict[str, Optional[float]], goals: List[str]) -> bool:
    return all(data.get(g) is not None for g in goals)

# --------------------------------------------------------------------------- #
# Normal equations for linear regression (no external libs)
# --------------------------------------------------------------------------- #

def transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]

def mat_mult(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(len(b[0])):
                result[i][j] += aik * b[k][j]
    return result

def mat_inverse(a: List[List[float]]) -> List[List[float]]:
    """Inverse of a square matrix using Gauss-Jordan elimination."""
    n = len(a)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            for r in range(i+1, n):
                if abs(aug[r][i]) > 1e-12:
                    aug[i], aug[r] = aug[r], aug[i]
                    pivot = aug[i][i]
                    break
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        for j in range(2*n):
            aug[i][j] /= pivot
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor != 0.0:
                for j in range(2*n):
                    aug[r][j] -= factor * aug[i][j]
    inv = [row[n:] for row in aug]
    return inv

def prepare_matrix(rows: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
    X = [[1.0] + r[:-1] for r in rows]
    y = [r[-1] for r in rows]
    return X, y

def normal_equation(X: List[List[float]], y: List[float]) -> List[float]:
    Xt = transpose(X)
    XtX = mat_mult(Xt, X)
    Xty = mat_mult(Xt, [[val] for val in y])
    XtX_inv = mat_inverse(XtX)
    beta_mat = mat_mult(XtX_inv, Xty)
    beta = [row[0] for row in beta_mat]
    return beta

def predict_row(beta: List[float], x: List[float]) -> float:
    return sum(b * val for b, val in zip(beta, [1.0] + x))

# --------------------------------------------------------------------------- #
# CAR-19: selection vs random
# --------------------------------------------------------------------------- #

def composite_score_row(data: Dict[str, Optional[float]], goals: List[str], minmax: Dict[str, Tuple[float, float]]) -> Optional[float]:
    norm = []
    for g in goals:
        v = data.get(g)
        if v is None:
            return None
        mn, mx = minmax[g]
        if mn == mx:
            n = 0.0
        else:
            n = (v - mn) / (mx - mn) if g.endswith('+') else (mx - v) / (mx - mn)
        norm.append(n)
    return sum(norm) / len(norm)

def car_19(rows: List[Tuple[int, Dict[str, Optional[float]]]], headers: List[str]) -> Dict:
    goals = goal_columns(headers)
    minmax: Dict[str, Tuple[float, float]] = {}
    for g in goals:
        vals = [data[g] for _, data in rows if data.get(g) is not None]
        if vals:
            minmax[g] = (min(vals), max(vals))
    # All rows that can be evaluated
    eval_rows = [(num, data) for num, data in rows if has_all_goals(data, goals)]
    # Sort by composite score (best first)
    eval_rows.sort(key=lambda nd: -composite_score_row(nd[1], goals, minmax))

    budget = min(30, len(rows) // 5)
    repeats = 20
    ours_best: List[int] = []
    random_best: List[int] = []

    rng = random.Random(42)

    for _ in range(repeats):
        # OURS: top budget rows
        our_set = eval_rows[:budget]
        if our_set:
            best_row_num = our_set[0][0]
        else:
            best_row_num = None
        ours_best.append(best_row_num)

        # RANDOM: sample budget rows with all goals
        random_indices = []
        while len(random_indices) < budget:
            idx = rng.randrange(len(rows))
            num, data = rows[idx]
            if has_all_goals(data, goals):
                random_indices.append(num)
        random_best_row = None
        best_score = -1.0
        for num in random_indices:
            data = next(d for n, d in rows if n == num)
            score = composite_score_row(data, goals, minmax)
            if score is not None and score > best_score:
                best_score = score
                random_best_row = num
        random_best.append(random_best_row)

    ours_better = sum(1 for o, r in zip(ours_best, random_best) if o is not None and (r is None or o != r and composite_score_row(rows[ours_best.index(o)-1][1], goals, minmax) > composite_score_row(rows[random_best.index(r)-1][1], goals, minmax)))
    # Simplify verdict logic: compare composite values
    ours_vals = []
    random_vals = []
    for o, r in zip(ours_best, random_best):
        o_val = composite_score_row(next(d for n, d in rows if n == o), goals, minmax) if o else None
        r_val = composite_score_row(next(d for n, d in rows if n == r), goals, minmax) if r else None
        ours_vals.append(o_val)
        random_vals.append(r_val)
    better_ours = sum(1 for ov, rv in zip(ours_vals, random_vals) if ov is not None and (rv is None or ov > rv))
    better_random = sum(1 for ov, rv in zip(ours_vals, random_vals) if rv is not None and (ov is None or rv > ov))
    if better_ours > better_random:
        verdict = "ours better"
    elif better_random > better_ours:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }

# --------------------------------------------------------------------------- #
# CAR-20: model vs baseline
# --------------------------------------------------------------------------- #

def car_20(rows: List[Tuple[int, Dict[str, Optional[float]]]], headers: List[str]) -> Dict:
    goals = goal_columns(headers)
    mpg_col = next(g for g in goals if g == "Mpg+")  # target
    # Identify held-out rows
    held_out = {num for num, _ in rows if num % 5 == 0}
    train_rows = [(num, data) for num, data in rows if num not in held_out]
    # Features: all numeric columns except the target goal
    feature_cols = numeric_columns(headers)
    if mpg_col in feature_cols:
        feature_cols.remove(mpg_col)
    # Compute column means on training set for imputation
    col_means: Dict[str, float] = {}
    for f in feature_cols:
        vals = [data[f] for _, data in train_rows if data.get(f) is not None]
        if vals:
            col_means[f] = sum(vals) / len(vals)
        else:
            col_means[f] = 0.0

    # Build training matrix
    X_train = []
    y_train = []
    for _, data in train_rows:
        if data.get(mpg_col) is None:
            continue
        row = []
        for f in feature_cols:
            val = data.get(f)
            row.append(val if val is not None else col_means[f])
        X_train.append(row)
        y_train.append(data[mpg_col])

    if not X_train:
        # No training data
        predictions = {str(num): None for num, _ in rows if num in held_out}
        return {
            "predictions": predictions,
            "baseline": None,
            "mae_model": None,
            "mae_baseline": None,
            "verdict": "no clear difference"
        }

    beta = normal_equation(prepare_matrix(X_train)[0], prepare_matrix(X_train)[1])
    # Predict for held-out rows
    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count_model = 0
    count_baseline = 0
    # Baseline: mean Mpg+ of training set
    baseline_val = sum(y_train) / len(y_train)

    for num, data in rows:
        if num not in held_out:
            continue
        mpg_true = data.get(mpg_col)
        # predict using model
        x_vec = [data.get(f) if data.get(f) is not None else col_means[f] for f in feature_cols]
        pred = predict_row(beta, x_vec)
        predictions[str(num)] = pred
        if mpg_true is not None:
            mae_model += abs(pred - mpg_true)
            mae_baseline += abs(baseline_val - mpg_true)
            count_model += 1
            count_baseline += 1

    if count_model > 0:
        mae_model /= count_model
    if count_baseline > 0:
        mae_baseline /= count_baseline

    if mae_model is not None and mae_baseline is not None:
        if mae_model < mae_baseline:
            verdict = "better"
        elif mae_model > mae_baseline:
            verdict = "worse"
        else:
            verdict = "no clear difference"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": baseline_val,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_10.py <csv_path>\n")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, headers = load_rows(csv_path)

    # CAR-19
    print("== CAR-19 ==")
    res19 = car_19(rows, headers)
    print(f"Budget: {res19['budget']}")
    print(f"Repeats: {res19['repeats']}")
    print(f"Ours best rows: {res19['ours']}")
    print(f"Random best rows: {res19['random']}")
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER CAR-19 {json.dumps(res19)}")

    # CAR-20
    print("\n== CAR-20 ==")
    res20 = car_20(rows, headers)
    print(f"Predictions (row->Mpg+): {res20['predictions']}")
    print(f"Baseline mean Mpg+: {res20['baseline']}")
    print(f"MAE model: {res20['mae_model']}")
    print(f"MAE baseline: {res20['mae_baseline']}")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER CAR-20 {json.dumps(res20)}")

if __name__ == "__main__":
    main()

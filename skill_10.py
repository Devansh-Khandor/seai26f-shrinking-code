#!/usr/bin/env python3
"""
skill_10.py – CAR-19 and CAR-20
"""

import csv
import json
import sys
import random
import math

# ------------------------------------------------------------------
# Reuse helpers from earlier skills
# ------------------------------------------------------------------
try:
    from skill_01 import parse_value, compute_stats
except Exception:  # fallback if skill_01 not importable
    def parse_value(val):
        if val == '?' or val == '':
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def compute_stats(values):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = sum(sorted_vals) / n
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        return {
            'mean': mean,
            'median': median,
            'min': sorted_vals[0],
            'max': sorted_vals[-1],
        }

# Linear regression helper from skill_02
try:
    from skill_02 import train_linear_regression
except Exception:
    # Minimal reimplementation of train_linear_regression
    import math as _math
    def train_linear_regression(target_col, predictor_cols, rows):
        n = 0
        Sxx_cross = [[0.0 for _ in predictor_cols] for _ in predictor_cols]
        Sxy = [0.0 for _ in predictor_cols]
        sum_y = 0.0
        sum_x = [0.0 for _ in predictor_cols]
        y_vals = []
        x_mat = []
        for row in rows:
            y = row.get(target_col)
            if y is None:
                continue
            xs = []
            missing = False
            for col in predictor_cols:
                x = row.get(col)
                if x is None:
                    missing = True
                    break
                xs.append(x)
            if missing:
                continue
            n += 1
            sum_y += y
            y_vals.append(y)
            x_mat.append(xs)
            for i, xi in enumerate(xs):
                sum_x[i] += xi
                Sxx_cross[i][i] += xi * xi
                Sxy[i] += xi * y
                for j in range(i+1, len(predictor_cols)):
                    xj = xs[j]
                    Sxx_cross[i][j] += xi * xj
                    Sxx_cross[j][i] += xi * xj
        if n == 0:
            return 0.0, [0.0]*len(predictor_cols), 0.0
        # Solve XTX * beta = XTy
        # Gaussian elimination
        A = [row[:] for row in Sxx_cross]
        b = Sxy[:]
        N = len(predictor_cols)
        # Forward
        for i in range(N):
            # pivot
            piv = A[i][i]
            if abs(piv) < 1e-12:
                for k in range(i+1, N):
                    if abs(A[k][i]) > 1e-12:
                        A[i], A[k] = A[k], A[i]
                        b[i], b[k] = b[k], b[i]
                        piv = A[i][i]
                        break
            if abs(piv) < 1e-12:
                coeffs = [0.0]*N
                intercept = 0.0
                residual_std = 0.0
                return intercept, coeffs, residual_std
            for j in range(i+1, N):
                factor = A[j][i] / piv
                for k in range(i, N):
                    A[j][k] -= factor * A[i][k]
                b[j] -= factor * b[i]
        # Backward
        coeffs = [0.0]*N
        for i in range(N-1, -1, -1):
            s = b[i]
            for j in range(i+1, N):
                s -= A[i][j] * coeffs[j]
            coeffs[i] = s / A[i][i]
        # Intercept
        mean_y = sum_y / n
        mean_x = [sx / n for sx in sum_x]
        intercept = mean_y
        for i, bx in enumerate(coeffs):
            intercept -= bx * mean_x[i]
        # Residual std
        resid_sq = 0.0
        for idx in range(n):
            pred = intercept
            for i, bx in enumerate(coeffs):
                pred += bx * x_mat[idx][i]
            resid_sq += (y_vals[idx] - pred) ** 2
        residual_std = _math.sqrt(resid_sq / (n - len(predictor_cols) - 1)) if n > len(predictor_cols) + 1 else 0.0
        return intercept, coeffs, residual_std

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']

def performance_score(row):
    """Return overall performance score if all goal columns present."""
    try:
        return -row['Lbs-'] + row['Acc+'] + row['Mpg+']
    except (TypeError, KeyError):
        return None

def round3(x):
    return round(x, 3)

# ------------------------------------------------------------------
# CAR-19 implementation
# ------------------------------------------------------------------
def car_19(rows, budget):
    """Compare our selection method with random selection."""
    # Candidate rows: those with complete goal data
    candidate_indices = [i for i, r in enumerate(rows, start=1) if all(r.get(col) is not None for col in GOAL_COLUMNS)]
    scores = {i: performance_score(rows[i-1]) for i in candidate_indices}
    # Sort candidates by score descending
    sorted_candidates = sorted(candidate_indices, key=lambda i: (-scores[i], i))
    # Ours: top budget rows
    ours_best = sorted_candidates[0] if sorted_candidates else None
    repeats = 20
    ours_best_rows = [ours_best] * repeats
    random_best_rows = []
    for _ in range(repeats):
        chosen = random.sample(candidate_indices, min(budget, len(candidate_indices)))
        best = max(chosen, key=lambda i: scores[i])
        random_best_rows.append(best)
    # Verdict
    ours_wins = sum(1 for o, r in zip(ours_best_rows, random_best_rows) if o > r or (o == r and o == ours_best))
    random_wins = repeats - ours_wins
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"
    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    return answer

# ------------------------------------------------------------------
# CAR-20 implementation
# ------------------------------------------------------------------
def car_20(rows, held_out_rows):
    """Predict held‑out Mpg+ using linear regression and compare to baseline."""
    # Training rows: not held out
    train_rows = [r for idx, r in enumerate(rows, start=1) if idx not in held_out_rows]
    # Train linear regression on Lbs- and Acc+ to predict Mpg+
    intercept, coeffs, _ = train_linear_regression('Mpg+', ['Lbs-', 'Acc+'], train_rows)
    # Mean of predictors for imputation
    mean_preds = {}
    for col in ['Lbs-', 'Acc+']:
        vals = [r[col] for r in train_rows if r[col] is not None]
        mean_preds[col] = sum(vals) / len(vals) if vals else 0.0
    predictions = {}
    mae_sum = 0.0
    mae_count = 0
    for idx in held_out_rows:
        r = rows[idx-1]
        x_vals = [r.get(col) if r.get(col) is not None else mean_preds[col] for col in ['Lbs-', 'Acc+']]
        pred = intercept + coeffs[0] * x_vals[0] + coeffs[1] * x_vals[1]
        predictions[str(idx)] = round3(pred)
        actual = r.get('Mpg+')
        if actual is not None:
            mae_sum += abs(pred - actual)
            mae_count += 1
    mae_model = mae_sum / mae_count if mae_count else 0.0
    # Baseline: mean Mpg+ of training rows
    base_vals = [r['Mpg+'] for r in train_rows if r['Mpg+'] is not None]
    baseline_mean = sum(base_vals) / len(base_vals) if base_vals else 0.0
    mae_base_sum = sum(abs(baseline_mean - v) for v in base_vals)
    mae_baseline = mae_base_sum / len(base_vals) if base_vals else 0.0
    # Verdict
    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"
    answer = {
        "predictions": predictions,
        "baseline": round3(baseline_mean),
        "mae_model": round3(mae_model),
        "mae_baseline": round3(mae_baseline),
        "verdict": verdict
    }
    return answer

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    # Parse numeric values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for k, v in r.items():
            parsed[k] = parse_value(v)
        parsed_rows.append(parsed)

    total_rows = len(parsed_rows)
    budget = min(30, total_rows // 5)

    # Held‑out rows: numbers divisible by 5
    held_out_rows = set(i for i in range(1, total_rows + 1) if i % 5 == 0)

    # CAR-19
    print("== CAR-19 ==")
    answer_19 = car_19(parsed_rows, budget)
    # Provide at least three lines of results
    print(f"Ours best row (deterministic): {answer_19['ours'][0]}")
    print(f"Random best rows (sample 20): {answer_19['random'][:5]} ...")
    print(f"Verdict: {answer_19['verdict']}")
    print(f"ANSWER CAR-19 {json.dumps(answer_19, separators=(',', ':'))}")

    # CAR-20
    print("\n== CAR-20 ==")
    answer_20 = car_20(parsed_rows, held_out_rows)
    print(f"Baseline mean Mpg+: {answer_20['baseline']}")
    print(f"Model MAE: {answer_20['mae_model']}, Baseline MAE: {answer_20['mae_baseline']}")
    print(f"Verdict: {answer_20['verdict']}")
    print(f"ANSWER CAR-20 {json.dumps(answer_20, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

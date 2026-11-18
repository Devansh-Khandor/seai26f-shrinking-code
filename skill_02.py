#!/usr/bin/env python3
import csv
import json
import sys
import math

# ------------------------------------------------------------------
# Utility functions (copied from skill_01)
# ------------------------------------------------------------------
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']
DIRECTION = {
    'Lbs-': 'min',   # minimize
    'Acc+': 'max',   # maximize
    'Mpg+': 'max',   # maximize
}

def parse_value(val):
    """Parse a CSV cell value. Return float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(values):
    """Return dict with mean, median, min, max."""
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

# ------------------------------------------------------------------
# Linear regression helpers
# ------------------------------------------------------------------
def train_linear_regression(target_col, predictor_cols, rows):
    """
    Train a simple linear regression with one intercept and
    len(predictor_cols) predictors.
    Returns (intercept, [coeff1, coeff2, ...], residual_std)
    """
    n = 0
    # sums for normal equations
    Sxx = [0.0 for _ in predictor_cols]  # diagonal
    Sxy = [0.0 for _ in predictor_cols]
    Sxx_cross = [[0.0 for _ in predictor_cols] for _ in predictor_cols]
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

        # accumulate sums
        n += 1
        sum_y += y
        y_vals.append(y)
        x_mat.append(xs)
        for i, xi in enumerate(xs):
            sum_x[i] += xi
            Sxx[i] += xi * xi
            Sxy[i] += xi * y
            for j, xj in enumerate(xs):
                Sxx_cross[i][j] += xi * xj

    if n == 0:
        return 0.0, [0.0]*len(predictor_cols), 0.0

    # Build normal equations: (X^T X) * beta = X^T y
    # We solve for beta (coefficients). Intercept will be handled later.
    # Compute matrix (X^T X)
    XTX = [[0.0 for _ in predictor_cols] for _ in predictor_cols]
    for i in range(len(predictor_cols)):
        for j in range(len(predictor_cols)):
            XTX[i][j] = Sxx_cross[i][j]

    # Compute X^T y vector
    XTy = [Sxy[i] for i in range(len(predictor_cols))]

    # Solve linear system XTX * beta = XTy
    # For up to 3 predictors we can use Gaussian elimination
    coeffs = _solve_linear_system(XTX, XTy)
    # Compute intercept
    mean_y = sum_y / n
    mean_x = [sx / n for sx in sum_x]
    intercept = mean_y
    for i, bx in enumerate(coeffs):
        intercept -= bx * mean_x[i]

    # Compute residual standard deviation
    resid_sq_sum = 0.0
    for idx in range(n):
        pred = intercept
        for i, bx in enumerate(coeffs):
            pred += bx * x_mat[idx][i]
        resid_sq_sum += (y_vals[idx] - pred) ** 2
    if n > len(predictor_cols):
        residual_std = math.sqrt(resid_sq_sum / (n - len(predictor_cols) - 1))
    else:
        residual_std = 0.0

    return intercept, coeffs, residual_std

def _solve_linear_system(A, b):
    """
    Solve Ax = b for small square A using Gaussian elimination.
    Return list of solutions.
    """
    n = len(b)
    # Augmented matrix
    M = [row[:] + [b_val] for row, b_val in zip(A, b)]

    # Forward elimination
    for i in range(n):
        # pivot
        piv = M[i][i]
        if abs(piv) < 1e-12:
            # Find non-zero pivot and swap
            for k in range(i+1, n):
                if abs(M[k][i]) > 1e-12:
                    M[i], M[k] = M[k], M[i]
                    piv = M[i][i]
                    break
        if abs(piv) < 1e-12:
            # Singular matrix
            return [0.0]*n
        for j in range(i+1, n):
            factor = M[j][i] / piv
            for k in range(i, n+1):
                M[j][k] -= factor * M[i][k]
    # Back substitution
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = M[i][n]
        for j in range(i+1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Parse numeric values
    parsed_rows = []
    for r in rows:
        parsed = {}
        for key, val in r.items():
            parsed[key] = parse_value(val)
        parsed_rows.append(parsed)

    total_rows = len(parsed_rows)

    # Determine held-out rows (1-based)
    held_out_rows = set(i+1 for i in range(total_rows) if (i+1) % 5 == 0)

    # ------------------------------------------------------------------
    # CAR-03: predict missing Mpg+ for held-out rows
    # ------------------------------------------------------------------
    print("== CAR-03 ==")

    # Train on rows not held-out where target and predictors known
    train_rows = [parsed_rows[i] for i in range(total_rows) if (i+1) not in held_out_rows]
    mpg_predictors = ['Lbs-', 'Acc+']
    intercept_mpg, coeffs_mpg, _ = train_linear_regression('Mpg+', mpg_predictors, train_rows)

    # Compute mean values of predictors for imputation
    mean_preds_mpg = {}
    for col in mpg_predictors:
        vals = [r[col] for r in train_rows if r[col] is not None]
        mean_preds_mpg[col] = sum(vals)/len(vals) if vals else 0.0

    predictions_mpg = {}
    mae_sum = 0.0
    mae_count = 0

    for idx, r in enumerate(parsed_rows):
        row_num = idx + 1
        if row_num in held_out_rows:
            # gather predictors
            x_vals = []
            for col in mpg_predictors:
                v = r.get(col)
                if v is None:
                    v = mean_preds_mpg[col]
                x_vals.append(v)
            pred = intercept_mpg + sum(c * xv for c, xv in zip(coeffs_mpg, x_vals))
            predictions_mpg[row_num] = pred
            # compute MAE if actual known
            actual = r.get('Mpg+')
            if actual is not None:
                mae_sum += abs(pred - actual)
                mae_count += 1

    for row_num, pred in sorted(predictions_mpg.items()):
        print(f"Row {row_num}: predicted Mpg+ = {pred:.3f}")

    mae = mae_sum / mae_count if mae_count else 0.0
    print(f"MAE: {mae:.3f}")

    answer_03 = {"predictions": {str(k): round(v,3) for k,v in predictions_mpg.items()},
                 "mae": round(mae,3)}
    print(f"ANSWER CAR-03 {json.dumps(answer_03, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # CAR-04: estimate Lbs- with uncertainty
    # ------------------------------------------------------------------
    print("\n== CAR-04 ==")

    # Train on rows not held-out where target and predictors known
    weight_predictors = ['Mpg+', 'Acc+']
    intercept_lbs, coeffs_lbs, residual_std_lbs = train_linear_regression('Lbs-', weight_predictors, train_rows)

    # Compute mean values of predictors for imputation
    mean_preds_lbs = {}
    for col in weight_predictors:
        vals = [r[col] for r in train_rows if r[col] is not None]
        mean_preds_lbs[col] = sum(vals)/len(vals) if vals else 0.0

    predictions_lbs = {}
    for idx, r in enumerate(parsed_rows):
        row_num = idx + 1
        if row_num in held_out_rows:
            x_vals = []
            for col in weight_predictors:
                v = r.get(col)
                if v is None:
                    v = mean_preds_lbs[col]
                x_vals.append(v)
            estimate = intercept_lbs + sum(c * xv for c, xv in zip(coeffs_lbs, x_vals))
            # simple 95% interval: estimate +/- 2 * residual_std_lbs
            low = estimate - 2 * residual_std_lbs
            high = estimate + 2 * residual_std_lbs
            predictions_lbs[row_num] = {
                "estimate": estimate,
                "low": low,
                "high": high
            }
            print(f"Row {row_num}: estimate={estimate:.3f}, low={low:.3f}, high={high:.3f}")

    answer_04 = {"predictions": {str(k): {"estimate": round(v["estimate"],3),
                                         "low": round(v["low"],3),
                                         "high": round(v["high"],3)}
                 for k,v in predictions_lbs.items()}}
    print(f"ANSWER CAR-04 {json.dumps(answer_04, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

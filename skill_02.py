#!/usr/bin/env python3
import csv
import json
import sys
import math

# ---------- Data utilities ----------

def read_data(path):
    """
    Reads the CSV file and returns a list of dictionaries (rows).
    '?' or '' are converted to None, numeric values to float, others kept as string.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, val in row.items():
                if val == '?' or val == '':
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(val)
                    except ValueError:
                        parsed[key] = val
            rows.append(parsed)
    return rows

# ---------- Linear algebra utilities ----------

def transpose(matrix):
    return [list(col) for col in zip(*matrix)]

def mat_mult(A, B):
    """Multiply matrix A (m x n) with B (n x p)."""
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def solve_linear_equation(A, b):
    """
    Solve Ax = b via Gaussian elimination with partial pivoting.
    A is a square matrix (list of lists), b is list.
    Returns list x.
    """
    n = len(A)
    # Augment
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        # Pivot
        pivot = i
        for j in range(i+1, n):
            if abs(M[j][i]) > abs(M[pivot][i]):
                pivot = j
        if abs(M[pivot][i]) < 1e-12:
            raise ValueError("Singular matrix encountered.")
        M[i], M[pivot] = M[pivot], M[i]
        # Normalize pivot row
        pivot_val = M[i][i]
        for k in range(i, n+1):
            M[i][k] /= pivot_val
        # Eliminate other rows
        for j in range(n):
            if j == i:
                continue
            factor = M[j][i]
            for k in range(i, n+1):
                M[j][k] -= factor * M[i][k]
    return [M[i][n] for i in range(n)]

# ---------- Regression utilities ----------

def build_design_matrix(rows, predictors, intercept=True):
    """
    Builds the design matrix X (list of lists).
    predictors: list of column names to use as features.
    intercept: if True, adds a column of 1.0 at the beginning.
    """
    X = []
    for row in rows:
        vec = []
        if intercept:
            vec.append(1.0)
        for col in predictors:
            val = row[col]
            if val is None:
                # Should not happen if rows were pre-filtered
                val = 0.0
            vec.append(val)
        X.append(vec)
    return X

def normal_equation_solution(X, y):
    """Solves for beta in X * beta = y."""
    Xt = transpose(X)
    XtX = mat_mult(Xt, X)
    Xty = [sum(Xt[i][j] * y[j] for j in range(len(y))) for i in range(len(Xt))]
    beta = solve_linear_equation(XtX, Xty)
    return beta

def predict(X, beta):
    """Compute predictions for design matrix X given coefficients beta."""
    return [sum(X[i][j] * beta[j] for j in range(len(beta))) for i in range(len(X))]

def mean_abs_error(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

def standard_deviation(values):
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / n)

# ---------- Prediction functions ----------

def linear_regression_predict(rows, target_col, predictor_cols, heldout_indices):
    """
    Train on all rows except those whose 1-based index is in heldout_indices.
    Return predictions for heldout rows and mae.
    """
    train_rows = [row for idx, row in enumerate(rows, start=1) if idx not in heldout_indices]
    predict_rows = [row for idx, row in enumerate(rows, start=1) if idx in heldout_indices]

    # Filter training rows to those with no missing target or predictors
    filtered_train = []
    for row in train_rows:
        if row[target_col] is None:
            continue
        missing = False
        for col in predictor_cols:
            if row[col] is None:
                missing = True
                break
        if not missing:
            filtered_train.append(row)

    if not filtered_train:
        return {}, 0.0

    # Build design matrix for training
    X_train = build_design_matrix(filtered_train, predictor_cols)
    y_train = [row[target_col] for row in filtered_train]

    # Solve for coefficients
    beta = normal_equation_solution(X_train, y_train)

    # Build predictions for held-out rows
    predictions = {}
    y_true = []
    y_pred = []

    # Pre-compute mean for each predictor to handle missing values
    predictor_means = {}
    for col in predictor_cols:
        vals = [row[col] for row in filtered_train if row[col] is not None]
        predictor_means[col] = sum(vals) / len(vals) if vals else 0.0

    for row in predict_rows:
        vec = [1.0]  # intercept
        for col in predictor_cols:
            val = row[col]
            if val is None:
                val = predictor_means[col]
            vec.append(val)
        pred = sum(vec[j] * beta[j] for j in range(len(beta)))
        idx = rows.index(row) + 1
        predictions[str(idx)] = round(pred, 2)

        # For MAE calculation if true value exists
        true_val = row[target_col]
        if true_val is not None:
            y_true.append(true_val)
            y_pred.append(pred)

    mae = mean_abs_error(y_true, y_pred) if y_true else 0.0
    return predictions, round(mae, 4)

def linear_regression_uncertainty(rows, target_col, predictor_cols, heldout_indices):
    """
    Train on all rows except heldout, predict on heldout, and produce uncertainty.
    Uncertainty interval: estimate ± 2 * residual_std
    """
    train_rows = [row for idx, row in enumerate(rows, start=1) if idx not in heldout_indices]
    predict_rows = [row for idx, row in enumerate(rows, start=1) if idx in heldout_indices]

    filtered_train = []
    for row in train_rows:
        if row[target_col] is None:
            continue
        missing = False
        for col in predictor_cols:
            if row[col] is None:
                missing = True
                break
        if not missing:
            filtered_train.append(row)

    if not filtered_train:
        return {}

    X_train = build_design_matrix(filtered_train, predictor_cols)
    y_train = [row[target_col] for row in filtered_train]
    beta = normal_equation_solution(X_train, y_train)

    # Residuals
    y_hat = predict(X_train, beta)
    residuals = [y_train[i] - y_hat[i] for i in range(len(y_train))]
    res_std = standard_deviation(residuals)

    # Predictor means for imputation
    predictor_means = {}
    for col in predictor_cols:
        vals = [row[col] for row in filtered_train if row[col] is not None]
        predictor_means[col] = sum(vals) / len(vals) if vals else 0.0

    predictions = {}
    for row in predict_rows:
        vec = [1.0]
        for col in predictor_cols:
            val = row[col]
            if val is None:
                val = predictor_means[col]
            vec.append(val)
        estimate = sum(vec[j] * beta[j] for j in range(len(beta)))
        low = estimate - 2 * res_std
        high = estimate + 2 * res_std
        idx = rows.index(row) + 1
        predictions[str(idx)] = {
            "estimate": round(estimate, 2),
            "low": round(low, 2),
            "high": round(high, 2)
        }
    return predictions

# ---------- Main routine ----------

def main(csv_path):
    rows = read_data(csv_path)

    # Identify held-out rows: divisible by 5 (1-based)
    heldout_indices = [idx for idx in range(1, len(rows)+1) if idx % 5 == 0]

    # ----- CAR-03 -----
    print("== CAR-03 ==")
    # Predict Mpg+
    mpg_predictors = ["Clndrs", "Volume", "Model", "origin", "Lbs-", "Acc+"]
    mpg_preds, mpg_mae = linear_regression_predict(rows, "Mpg+", mpg_predictors, heldout_indices)
    for rid, val in sorted(mpg_preds.items(), key=lambda x: int(x[0])):
        print(f"Row {rid} predicted Mpg+: {val}")
    print(f"MAE: {mpg_mae}")

    answer_car03 = {
        "predictions": mpg_preds,
        "mae": mpg_mae
    }
    print(f"ANSWER CAR-03 {json.dumps(answer_car03, ensure_ascii=False)}")

    # ----- CAR-04 -----
    print("\n== CAR-04 ==")
    # Predict Lbs-
    lbs_predictors = ["Clndrs", "Volume", "Model", "origin", "Acc+", "Mpg+"]
    lbs_preds = linear_regression_uncertainty(rows, "Lbs-", lbs_predictors, heldout_indices)
    for rid, stats in sorted(lbs_preds.items(), key=lambda x: int(x[0])):
        est = stats["estimate"]
        low = stats["low"]
        high = stats["high"]
        print(f"Row {rid} predicted Lbs-: estimate={est}, low={low}, high={high}")
    answer_car04 = {
        "predictions": lbs_preds
    }
    print(f"ANSWER CAR-04 {json.dumps(answer_car04, ensure_ascii=False)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

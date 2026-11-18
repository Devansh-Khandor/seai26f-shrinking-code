#!/usr/bin/env python3
"""
skill_02.py <csv_path>

Implements NASA-03 (Effort prediction) and NASA-04 (Defect prediction with uncertainty)
"""

import sys
import csv
import json
import math
from statistics import mean

# ---------- Utility functions ----------
def parse_csv(path):
    """Read CSV into list of dict rows (header omitted)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    """Convert cell to float, return None for missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_held_out(idx, nrows):
    """Row numbers are 1‑based. Divisible by 5."""
    return (idx % 5) == 0

# ---------- NASA‑03: Effort prediction ----------
def fit_single_regression(X, Y):
    """Fit y = a * x. Returns a."""
    n = len(X)
    if n == 0:
        return 0.0
    x_mean = sum(X) / n
    y_mean = sum(Y) / n
    num = sum((X[i] - x_mean) * (Y[i] - y_mean) for i in range(n))
    den = sum((X[i] - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0

def predict_effort(rows, held_out_set):
    """Return predictions dict (row_number -> predicted Effort-) and MAE."""
    X = []
    Y = []
    for idx, r in enumerate(rows, start=1):
        if is_held_out(idx, len(rows)) or idx in held_out_set:
            continue
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        if k is None or e is None:
            continue
        X.append(k)
        Y.append(e)
    a = fit_single_regression(X, Y)
    predictions = {}
    abs_errors = []
    for idx, r in enumerate(rows, start=1):
        if not is_held_out(idx, len(rows)):
            continue
        k = to_float(r["Kloc+"])
        e_true = to_float(r["Effort-"])
        if k is None or e_true is None:
            continue
        pred = a * k
        predictions[str(idx)] = pred
        abs_errors.append(abs(e_true - pred))
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return predictions, mae

# ---------- NASA‑04: Defect prediction with interval ----------
def matrix_multiply(A, B):
    """Multiply two matrices (list of lists)."""
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def transpose(A):
    return [list(col) for col in zip(*A)]

def invert_3x3(M):
    """Invert a 3x3 matrix. Returns None if singular."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    inv = [
        [(e*i - f*h) * inv_det, (c*h - b*i) * inv_det, (b*f - c*e) * inv_det],
        [(f*g - d*i) * inv_det, (a*i - c*g) * inv_det, (c*d - a*f) * inv_det],
        [(d*h - e*g) * inv_det, (b*g - a*h) * inv_det, (a*e - b*d) * inv_det],
    ]
    return inv

def fit_multiple_regression(X, Y):
    """
    Fit y = beta0 + beta1*x1 + beta2*x2
    X is list of [1, x1, x2]
    Y is list of y
    Returns beta as list [beta0, beta1, beta2]
    """
    n = len(X)
    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    XtY = [sum(Xt[i][j] * Y[j] for j in range(n)) for i in range(len(Xt))]
    inv_XtX = invert_3x3(XtX)
    if inv_XtX is None:
        return [0.0, 0.0, 0.0]
    beta = [sum(inv_XtX[i][j] * XtY[j] for j in range(3)) for i in range(3)]
    return beta

def predict_defects(rows, held_out_set):
    """Return predictions dict (row -> {estimate, low, high}) and MAE."""
    # Training data
    X = []
    Y = []
    for idx, r in enumerate(rows, start=1):
        if is_held_out(idx, len(rows)) or idx in held_out_set:
            continue
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        d = to_float(r["Defects-"])
        if k is None or e is None or d is None:
            continue
        X.append([1.0, k, e])
        Y.append(d)
    beta = fit_multiple_regression(X, Y)
    # Residuals and MSE
    residuals = []
    for i in range(len(Y)):
        pred = beta[0] + beta[1]*X[i][1] + beta[2]*X[i][2]
        residuals.append(Y[i] - pred)
    n = len(Y)
    p = 2  # predictors
    mse = sum(r**2 for r in residuals) / (n - p - 1) if n > p + 1 else 0.0
    # Invert XtX to get prediction variance factor for any x0
    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    inv_XtX = invert_3x3(XtX)
    # Predictions for held‑out
    predictions = {}
    abs_errors = []
    for idx, r in enumerate(rows, start=1):
        if not is_held_out(idx, len(rows)):
            continue
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        d_true = to_float(r["Defects-"])
        if k is None or e is None or d_true is None:
            continue
        x0 = [1.0, k, e]
        # Estimate
        est = beta[0] + beta[1]*k + beta[2]*e
        # Standard error of prediction
        # Compute x0' * inv_XtX * x0
        prod = [sum(inv_XtX[i][j] * x0[j] for j in range(3)) for i in range(3)]
        var_factor = sum(x0[i] * prod[i] for i in range(3))
        se = math.sqrt(mse * (1.0 + var_factor)) if mse > 0 else 0.0
        # 95% interval (normal approx)
        z = 1.96
        low = est - z * se
        high = est + z * se
        predictions[str(idx)] = {
            "estimate": est,
            "low": low,
            "high": high
        }
        abs_errors.append(abs(d_true - est))
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return predictions, mae

# ---------- Main ----------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_02.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    # Determine held-out set
    held_out = {idx for idx in range(1, len(rows)+1) if is_held_out(idx, len(rows))}
    # NASA-03
    print("== NASA-03 ==")
    effort_preds, effort_mae = predict_effort(rows, held_out)
    # Simple textual output
    print(f"Predicted Effort for {len(effort_preds)} held‑out rows.")
    avg_est = mean(effort_preds.values()) if effort_preds else 0.0
    print(f"Average predicted Effort: {avg_est:.4f}")
    print(f"MAE on held‑out: {effort_mae:.4f}")
    # Machine‑checked answer
    print("ANSWER NASA-03", json.dumps(
        {"predictions": effort_preds, "mae": effort_mae},
        separators=(',', ':')
    ))
    # NASA-04
    print("== NASA-04 ==")
    defect_preds, defect_mae = predict_defects(rows, held_out)
    # Textual summary
    print(f"Predicted Defects for {len(defect_preds)} held‑out rows.")
    avg_est = mean(item["estimate"] for item in defect_preds.values()) if defect_preds else 0.0
    print(f"Average predicted Defects: {avg_est:.4f}")
    print(f"MAE on held‑out: {defect_mae:.4f}")
    # Machine‑checked answer
    print("ANSWER NASA-04", json.dumps(
        {"predictions": defect_preds, "mae": defect_mae},
        separators=(',', ':')
    ))

if __name__ == "__main__":
    main()

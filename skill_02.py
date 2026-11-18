#!/usr/bin/env python3
"""
skill_02.py: Predict Effort- (NASA-03) and Defects- (NASA-04) for held‑out rows
using simple linear regression on Kloc+.
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# Helper functions – copied from skill_01 (parse_float, median, std_dev)
# ----------------------------------------------------------------------
def parse_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def median(lst):
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid-1] + s[mid]) / 2.0

def std_dev(values, mean):
    if len(values) == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)

# ----------------------------------------------------------------------
# Linear regression helpers
# ----------------------------------------------------------------------
def train_linear(X, y):
    """
    X: list of [1, feature] rows
    y: target values
    returns (beta0, beta1)
    """
    n = len(X)
    if n == 0:
        return (0.0, 0.0)

    # Compute X^T X
    a = sum(x[0]*x[0] for x in X)   # sum of ones = n
    b = sum(x[0]*x[1] for x in X)
    c = sum(x[1]*x[0] for x in X)
    d = sum(x[1]*x[1] for x in X)

    # Compute X^T y
    e = sum(x[0]*yi for x, yi in zip(X, y))
    f = sum(x[1]*yi for x, yi in zip(X, y))

    # Solve [[a,b],[c,d]] * beta = [e,f]
    det = a * d - b * c
    if abs(det) < 1e-12:
        # Singular – fall back to mean regression
        beta0 = sum(y) / n
        beta1 = 0.0
        return (beta0, beta1)

    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    beta0 = inv_a * e + inv_b * f
    beta1 = inv_c * e + inv_d * f
    return (beta0, beta1)

def predict(beta0, beta1, x):
    return beta0 + beta1 * x

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            rows.append((idx, r))

    # Identify held‑out rows (divisible by 5)
    held_out_nums = {idx for idx, _ in rows if idx % 5 == 0}
    history_rows = [(idx, r) for idx, r in rows if idx % 5 != 0]

    # ---------- NASA‑03 : Predict Effort‑ ----------
    # Build training set (history rows with valid Effort‑ and Kloc+)
    X_eff = []
    y_eff = []
    for idx, r in history_rows:
        eff = parse_float(r.get('Effort-'))
        kloc = parse_float(r.get('Kloc+'))
        if eff is None or kloc is None:
            continue
        X_eff.append([1.0, kloc])
        y_eff.append(eff)

    beta0_eff, beta1_eff = train_linear(X_eff, y_eff)

    # Predict for held‑out rows
    predictions_eff = {}
    abs_errs = []
    for idx, r in rows:
        if idx not in held_out_nums:
            continue
        eff_actual = parse_float(r.get('Effort-'))
        kloc = parse_float(r.get('Kloc+'))
        if kloc is None:
            # Fallback to mean of training y
            pred = sum(y_eff) / len(y_eff) if y_eff else 0.0
        else:
            pred = predict(beta0_eff, beta1_eff, kloc)
        predictions_eff[str(idx)] = round(pred, 4)
        if eff_actual is not None:
            abs_errs.append(abs(pred - eff_actual))

    mae_eff = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0
    mae_eff = round(mae_eff, 4)

    # ---------- NASA‑04 : Predict Defects‑ with uncertainty ----------
    # Build training set (history rows with valid Defects‑ and Kloc+)
    X_def = []
    y_def = []
    for idx, r in history_rows:
        defect = parse_float(r.get('Defects-'))
        kloc = parse_float(r.get('Kloc+'))
        if defect is None or kloc is None:
            continue
        X_def.append([1.0, kloc])
        y_def.append(defect)

    beta0_def, beta1_def = train_linear(X_def, y_def)

    # Compute residual standard deviation on training data
    if X_def:
        residuals = []
        for (x0, x1), y in zip(X_def, y_def):
            y_hat = predict(beta0_def, beta1_def, x1)
            residuals.append(y - y_hat)
        std_res = std_dev(residuals, 0.0)
    else:
        std_res = 0.0

    # Predict for held‑out rows
    predictions_def = {}
    for idx, r in rows:
        if idx not in held_out_nums:
            continue
        defect_actual = parse_float(r.get('Defects-'))
        kloc = parse_float(r.get('Kloc+'))
        if kloc is None:
            est = sum(y_def) / len(y_def) if y_def else 0.0
        else:
            est = predict(beta0_def, beta1_def, kloc)
        low = est - 1.96 * std_res
        high = est + 1.96 * std_res
        # Ensure low <= high
        if low > high:
            low, high = high, low
        predictions_def[str(idx)] = {
            "estimate": round(est, 4),
            "low": round(low, 4),
            "high": round(high, 4)
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print("== NASA-03 ==")
    for row_num, pred in predictions_eff.items():
        print(f"Row {row_num}: {pred}")
    print(f"MAE: {mae_eff}")
    nasa03_json = {"predictions": predictions_eff, "mae": mae_eff}
    print(f"ANSWER NASA-03 {nasa03_json}")

    print("\n== NASA-04 ==")
    for row_num, obj in predictions_def.items():
        print(f"Row {row_num}: estimate {obj['estimate']} low {obj['low']} high {obj['high']}")
    nasa04_json = {"predictions": predictions_def}
    print(f"ANSWER NASA-04 {nasa04_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

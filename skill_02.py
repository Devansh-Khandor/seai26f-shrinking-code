#!/usr/bin/env python3
"""
skill_02.py: Fuel‑efficiency & Weight prediction for auto93 dataset.
Usage: python skill_02.py <csv_path>
"""

import sys
import json
import random
import ezr

# ------------------------------------------------------------------
# Utility functions

def read_csv(path: str):
    """Read CSV and return header and rows (atoms)."""
    with open(path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
        rows = []
        for line in f:
            if line.strip():
                rows.append([ezr.atom(s) for s in line.split(",")])
    return header, rows


def build_table(header, rows):
    """Create an ezr table from header and rows."""
    return ezr.Tbl([header] + rows)


def column_index(tbl, name):
    """Return column index for column name."""
    for at, col in tbl.cols.items():
        if col.txt == name:
            return at
    raise KeyError(f"Column {name} not found")


# ------------------------------------------------------------------
# Linear regression utilities (no external libraries)

def gauss_jordan(A, b):
    """Solve A x = b using Gauss–Jordan elimination with partial pivoting.
    A is a list of lists, b a list. Returns solution list x."""
    n = len(A)
    # Augment matrix
    for i in range(n):
        A[i].append(b[i])
    # Forward elimination
    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
        # Normalize row
        div = A[col][col]
        for j in range(col, n + 1):
            A[col][j] /= div
        # Eliminate below
        for r in range(n):
            if r != col:
                factor = A[r][col]
                for j in range(col, n + 1):
                    A[r][j] -= factor * A[col][j]
    return [A[i][n] for i in range(n)]


def linear_regression(features, targets):
    """Fit linear regression on given feature matrix (list of lists)
    and target vector. Returns coefficients (including bias)."""
    n = len(features)
    p = len(features[0]) if n > 0 else 0
    # Build X^T X and X^T y
    XtX = [[0.0] * (p + 1) for _ in range(p + 1)]
    Xty = [0.0] * (p + 1)
    for i in range(n):
        row = [1.0] + features[i]
        for a in range(p + 1):
            Xty[a] += row[a] * targets[i]
            for b in range(p + 1):
                XtX[a][b] += row[a] * row[b]
    # Solve (X^T X) * coeffs = X^T y
    coeffs = gauss_jordan([row[:] for row in XtX], Xty)
    return coeffs  # list of length p+1


def predict_row(coeffs, feature_vals):
    """Predict target for a single row (feature values)."""
    return coeffs[0] + sum(coeffs[i + 1] * feature_vals[i] for i in range(len(feature_vals)))


# ------------------------------------------------------------------
# Model training and prediction for a single goal

def train_and_predict(tbl, target_name, predictors):
    """Return predictions dict for held‑out rows and mae for goal."""
    target_at = column_index(tbl, target_name)
    # Identify training rows (not held‑out, target present)
    all_rows = tbl.rows
    total_rows = len(all_rows)
    budget = min(30, total_rows // 5)
    training = [
        r for i, r in enumerate(all_rows, 1)
        if i % 5 != 0 and r[target_at] != "?"
    ]
    # Take at most budget rows
    training = training[:budget]
    # Compute means for predictors to impute missing values
    predictor_at = [column_index(tbl, name) for name in predictors]
    means = {}
    for at in predictor_at:
        vals = [r[at] for r in training if r[at] != "?"]
        means[at] = sum(vals) / len(vals) if vals else 0.0
    # Build training features and targets
    X = []
    y = []
    for r in training:
        row_feats = []
        for at in predictor_at:
            val = r[at]
            if val == "?":
                val = means[at]
            row_feats.append(val)
        X.append(row_feats)
        y.append(r[target_at])
    coeffs = linear_regression(X, y)
    # Compute residual std for uncertainty
    residuals = [y[i] - predict_row(coeffs, X[i]) for i in range(len(X))]
    if residuals:
        resid_std = (sum(v * v for v in residuals) / len(residuals)) ** 0.5
    else:
        resid_std = 0.0
    # Predict held‑out rows
    predictions = {}
    mae_n = 0
    mae_sum = 0.0
    for i, r in enumerate(all_rows, 1):
        if i % 5 == 0:  # held‑out
            # Prepare features
            row_feats = []
            skip = False
            for at in predictor_at:
                val = r[at]
                if val == "?":
                    if at in means:
                        val = means[at]
                    else:
                        skip = True
                row_feats.append(val)
            if skip:
                continue
            pred = predict_row(coeffs, row_feats)
            predictions[str(i)] = pred
            actual = r[target_at]
            if actual != "?":
                mae_sum += abs(pred - actual)
                mae_n += 1
    mae = mae_sum / mae_n if mae_n > 0 else 0.0
    return predictions, mae, resid_std


# ------------------------------------------------------------------
# Main driver

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # CAR-03: Fuel‑efficiency prediction (Mpg+)
    mpg_preds, mpg_mae, _ = train_and_predict(
        tbl,
        target_name="Mpg+",
        predictors=["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs-", "Acc+"],
    )
    print("== CAR-03 ==")
    # Print at least 3 lines of results
    for row_no, pred in list(mpg_preds.items())[:3]:
        actual = tbl.rows[int(row_no)-1]["Mpg+"]
        print(f"Row {row_no}: predicted Mpg+ = {pred:.3f} actual={actual}")
    print(f"Mean Absolute Error: {mpg_mae:.3f}")
    # JSON answer
    ans_mpg = {
        "predictions": {k: round(v, 10) for k, v in mpg_preds.items()},
        "mae": round(mpg_mae, 10),
    }
    print(f"ANSWER CAR-03 {json.dumps(ans_mpg)}")

    # CAR-04: Weight prediction (Lbs-)
    lbs_preds, _, lbs_std = train_and_predict(
        tbl,
        target_name="Lbs-",
        predictors=["Clndrs", "Volume", "HpX", "Model", "origin", "Acc+", "Mpg+"],
    )
    print("\n== CAR-04 ==")
    for row_no, pred in list(lbs_preds.items())[:3]:
        actual = tbl.rows[int(row_no)-1]["Lbs-"]
        low = pred - 2 * lbs_std
        high = pred + 2 * lbs_std
        print(f"Row {row_no}: estimate={pred:.2f} low={low:.2f} high={high:.2f} actual={actual}")
    print("Uncertainty range width (2σ):", f"{2 * lbs_std:.2f}")
    # JSON answer
    ans_lbs = {
        "predictions": {
            k: {"estimate": round(v, 10), "low": round(v - 2 * lbs_std, 10), "high": round(v + 2 * lbs_std, 10)}
            for k, v in lbs_preds.items()
        }
    }
    print(f"ANSWER CAR-04 {json.dumps(ans_lbs)}")


if __name__ == "__main__":
    main()

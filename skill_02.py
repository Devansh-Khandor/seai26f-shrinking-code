#!/usr/bin/env python3
"""
skill_02.py <csv path>

Implements:
    NASA-03: Estimate Effort- for held‑out rows and report MAE.
    NASA-04: Estimate Defects- and a 95% uncertainty range for held‑out rows.

The script prints a header for each request, a few human‑readable lines,
and then the machine‑checked JSON answer.
"""

import csv
import json
import sys

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def to_float(v):
    """Convert string to float or return None for missing."""
    return None if v == '?' else float(v)

def is_numeric(v):
    """Return True if v can be parsed as a float."""
    if v == '?':
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False

def build_categorical_mappings(rows, columns):
    """Return a dict mapping column->value->int."""
    mappings = {}
    for col in columns:
        if is_numeric(rows[0][col]):  # numeric column: no mapping needed
            continue
        mapping = {}
        idx = 0
        for row in rows:
            val = row[col]
            if val == '?':
                continue
            if val not in mapping:
                mapping[val] = idx
                idx += 1
        mappings[col] = mapping
    return mappings

def encode_row(row, cols, mappings):
    """Return list of numeric values for a row, using mappings for categorical."""
    vals = []
    for col in cols:
        v = row[col]
        if v == '?':
            vals.append(None)
            continue
        if col in mappings:
            vals.append(mappings[col][v])
        else:
            vals.append(float(v))
    return vals

def compute_means(rows, cols):
    """Compute mean for each column over rows, ignoring None."""
    sums = [0.0] * len(cols)
    counts = [0] * len(cols)
    for vals in rows:
        for i, v in enumerate(vals):
            if v is not None:
                sums[i] += v
                counts[i] += 1
    means = [s / c if c else 0.0 for s, c in zip(sums, counts)]
    return means

def replace_missing_with_mean(matrix, means):
    """Return a new matrix with None replaced by column means."""
    new = []
    for row in matrix:
        new.append([v if v is not None else m for v, m in zip(row, means)])
    return new

# Solve linear system A x = b via Gaussian elimination (partial pivot)
def solve_linear_system(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot
        for k in range(n):
            if k == i:
                continue
            factor = M[k][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]

def normal_equation(X, y):
    """Return regression coefficients solving (X^T X) beta = X^T y."""
    n_rows = len(X)
    n_feats = len(X[0])
    # Compute X^T X
    XtX = [[0.0] * n_feats for _ in range(n_feats)]
    XtY = [0.0] * n_feats
    for i in range(n_rows):
        xi = X[i]
        yi = y[i]
        for a in range(n_feats):
            XtY[a] += xi[a] * yi
            for b in range(n_feats):
                XtX[a][b] += xi[a] * xi[b]
    coeffs = solve_linear_system(XtX, XtY)
    return coeffs

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    # Determine feature columns (exclude those ending in 'X' or goal columns)
    feature_cols = [c for c in reader.fieldnames
                    if not c.endswith('X') and c not in ('Effort-', 'Defects-')]
    # Build categorical mappings
    cat_mappings = build_categorical_mappings(all_rows, feature_cols)
    # Encode all rows
    encoded_rows = [encode_row(r, feature_cols, cat_mappings) for r in all_rows]
    # Compute means for imputation
    means = compute_means(encoded_rows, feature_cols)
    # Prepare training data for Effort- regression (NASA-03)
    effort_targets = []
    effort_X = []
    effort_rows = []
    for idx, (row, enc) in enumerate(zip(all_rows, encoded_rows), start=1):
        if idx % 5 == 0:  # held-out row
            continue
        val = to_float(row['Effort-'])
        if val is None:
            continue
        effort_targets.append(val)
        effort_rows.append(enc)
    effort_X = replace_missing_with_mean(effort_rows, means)
    # Add intercept
    effort_X = [[1.0] + row for row in effort_X]
    # Train model
    if len(effort_X) == 0:
        effort_coeffs = [0.0]
    else:
        effort_coeffs = normal_equation(effort_X, effort_targets)

    # Prepare training data for Defects- regression (NASA-04)
    defect_targets = []
    defect_X = []
    defect_rows = []
    for idx, (row, enc) in enumerate(zip(all_rows, encoded_rows), start=1):
        if idx % 5 == 0:  # held-out
            continue
        val = to_float(row['Defects-'])
        if val is None:
            continue
        defect_targets.append(val)
        defect_rows.append(enc)
    defect_X = replace_missing_with_mean(defect_rows, means)
    defect_X = [[1.0] + row for row in defect_X]
    if len(defect_X) == 0:
        defect_coeffs = [0.0]
    else:
        defect_coeffs = normal_equation(defect_X, defect_targets)

    # Compute residual std dev for defects model
    defect_residuals = []
    if len(defect_X) > 0:
        for x, y in zip(defect_X, defect_targets):
            pred = sum(c * vx for c, vx in zip(defect_coeffs, x))
            defect_residuals.append(y - pred)
        dof = len(defect_X) - len(defect_coeffs)
        if dof > 0:
            sigma = (sum(r ** 2 for r in defect_residuals) / dof) ** 0.5
        else:
            sigma = 0.0
    else:
        sigma = 0.0

    # -----------------------------------------------
    # NASA-03 predictions
    # -----------------------------------------------
    nasa03_preds = {}
    mae_acc = 0.0
    count_mae = 0
    for idx, (row, enc) in enumerate(zip(all_rows, encoded_rows), start=1):
        if idx % 5 != 0:  # only held-out rows
            continue
        # Impute missing features
        imputed = [v if v is not None else means[i] for i, v in enumerate(enc)]
        vec = [1.0] + imputed
        pred = sum(c * vx for c, vx in zip(effort_coeffs, vec))
        nasa03_preds[str(idx)] = pred
        # Compute MAE
        actual = to_float(row['Effort-'])
        if actual is not None:
            mae_acc += abs(pred - actual)
            count_mae += 1
    mae = mae_acc / count_mae if count_mae else 0.0

    # -----------------------------------------------
    # NASA-04 predictions
    # -----------------------------------------------
    nasa04_preds = {}
    for idx, (row, enc) in enumerate(zip(all_rows, encoded_rows), start=1):
        if idx % 5 != 0:  # held-out
            continue
        imputed = [v if v is not None else means[i] for i, v in enumerate(enc)]
        vec = [1.0] + imputed
        est = sum(c * vx for c, vx in zip(defect_coeffs, vec))
        low = est - 1.96 * sigma
        high = est + 1.96 * sigma
        if low < 0.0:
            low = 0.0
        if high < 0.0:
            high = 0.0
        nasa04_preds[str(idx)] = {"estimate": est, "low": low, "high": high}

    # -----------------------------------------------
    # Output
    # -----------------------------------------------
    # NASA-03
    print("== NASA-03 ==")
    print(f"MAE of Effort- predictions on held‑out rows: {mae:.2f}")
    print(f"Number of held‑out rows evaluated: {len(nasa03_preds)}")
    print("Example predictions (first 3):")
    for r, val in list(nasa03_preds.items())[:3]:
        print(f"  Row {r}: {val:.2f}")
    print(f"ANSWER NASA-03 {json.dumps({'predictions': nasa03_preds, 'mae': mae})}")

    # NASA-04
    print("\n== NASA-04 ==")
    print(f"Uncertainty interval width (95%) using residual std={sigma:.2f}")
    print(f"Number of held‑out rows evaluated: {len(nasa04_preds)}")
    print("Example predictions (first 3):")
    for r, d in list(nasa04_preds.items())[:3]:
        print(f"  Row {r}: estimate={d['estimate']:.2f}, low={d['low']:.2f}, high={d['high']:.2f}")
    print(f"ANSWER NASA-04 {json.dumps({'predictions': nasa04_preds})}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

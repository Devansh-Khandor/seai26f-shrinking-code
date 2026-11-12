#!/usr/bin/env python3
"""
skill_02.py

This script implements NASA-03 (effort estimation) and NASA-04 (defect estimation) on the NASA software
project dataset.

Usage:
    python skill_02.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """Read CSV and return rows and numeric column names."""
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        numeric_cols = []
        for col in reader.fieldnames:
            try:
                for r in rows:
                    val = r[col]
                    if val != '':
                        float(val.replace(',', ''))
                        numeric_cols.append(col)
                        break
            except (ValueError, TypeError):
                continue
    return rows, numeric_cols


def to_numpy(rows, cols):
    """Return (n_samples, n_features) array of floats, NaN for missing."""
    arr = []
    for r in rows:
        row = []
        for c in cols:
            val = r[c]
            if val == '':
                row.append(np.nan)
            else:
                try:
                    row.append(float(val.replace(',', '')))
                except ValueError:
                    row.append(np.nan)
        arr.append(row)
    return np.array(arr, dtype=float)


def model_stats(X, y, model):
    """Return R^2, RMSE, residuals."""
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    residuals = y - y_pred
    return r2, rmse, residuals, y_pred


def predict_with_interval(model, X, residuals, x0):
    """
    Compute prediction for x0 and 95% prediction interval.
    Uses linear regression formula with inverse X^T X.
    """
    n, p = X.shape
    df = n - p - 1
    s = np.sqrt(np.sum(residuals ** 2) / df)

    # Build full design matrices with intercept
    Xfull = np.hstack([np.ones((n, 1)), X])
    inv_xtx = np.linalg.pinv(Xfull.T @ Xfull)

    x0_full = np.hstack([1.0, x0])
    se_pred = s * np.sqrt(1 + x0_full @ inv_xtx @ x0_full.T)

    # 95% t critical approx 2 for df large
    t_crit = 2.0
    return model.predict(x0_full.reshape(1, -1))[0], t_crit * se_pred


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # Identify target columns
    eff_col = 'Effort-'
    defect_col = 'Defects-'
    if eff_col not in numeric_cols or defect_col not in numeric_cols:
        print("Required numeric columns 'Effort-' and 'Defects-' not found.")
        sys.exit(1)

    # Features: all numeric columns except the two targets
    feature_cols = [c for c in numeric_cols if c not in (eff_col, defect_col)]

    if not feature_cols:
        print("No feature columns available for modeling.")
        sys.exit(1)

    # Build data matrices
    X_all = to_numpy(rows, feature_cols)
    y_eff = to_numpy(rows, [eff_col]).flatten()
    y_def = to_numpy(rows, [defect_col]).flatten()

    # Drop rows with NaNs in any of X or y
    mask_eff = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_eff)
    mask_def = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_def)

    X_eff = X_all[mask_eff]
    y_eff = y_eff[mask_eff]
    X_def = X_all[mask_def]
    y_def = y_def[mask_def]

    # Train models
    model_eff = LinearRegression(fit_intercept=True)
    model_eff.fit(X_eff, y_eff)
    r2_eff, rmse_eff, resid_eff, _ = model_stats(X_eff, y_eff, model_eff)

    model_def = LinearRegression(fit_intercept=True)
    model_def.fit(X_def, y_def)
    r2_def, rmse_def, resid_def, _ = model_stats(X_def, y_def, model_def)

    # Median feature vector (for a typical project)
    median_feat = np.median(X_all, axis=0)

    # Predict effort and defect for median project with interval
    pred_eff, unc_eff = predict_with_interval(model_eff, X_eff, resid_eff, median_feat)
    pred_def, unc_def = predict_with_interval(model_def, X_def, resid_def, median_feat)

    # ----------------- Output --------------------------------
    print("== NASA-03 ==")
    print(f"Effort model R²: {r2_eff:.3f}, RMSE: {rmse_eff:.3f}")
    print(f"Estimated effort for a median project: {pred_eff:.2f} ± {unc_eff:.2f} person‑months (95% PI)")
    print()

    print("== NASA-04 ==")
    print(f"Defect model R²: {r2_def:.3f}, RMSE: {rmse_def:.3f}")
    print(f"Estimated defect count for a median project: {pred_def:.2f} ± {unc_def:.2f} (95% PI)")
    print()


if __name__ == "__main__":
    main()

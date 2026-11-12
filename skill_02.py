#!/usr/bin/env python3
"""
Skill 02: NASA-03 and NASA-04 – Predictive modeling of effort and defects.
Author: Your Name
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

# ----------------------------------------------------------------------
# Utility: read CSV and convert numeric values where possible
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Reads a CSV file and returns:
        header: list of column names
        rows: list of dicts mapping column -> value (numeric where possible)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Helper: build feature matrix X and target vector y
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    """
    Given rows, produce:
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        feature_names: list of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []

    # Determine numeric columns
    numeric_cols = []
    for key in rows[0].keys():
        if key in exclude_cols or key == target_col:
            continue
        # Check if all values are numeric for this column
        if all(isinstance(r.get(key, None), (int, float)) for r in rows if isinstance(r.get(key, None), (int, float))):
            numeric_cols.append(key)

    # Build data arrays, skipping rows with missing target or any missing feature
    X_rows = []
    y_rows = []
    for r in rows:
        if not isinstance(r.get(target_col, None), (int, float)):
            continue  # skip rows without target
        feature_vals = []
        missing = False
        for col in numeric_cols:
            val = r.get(col, None)
            if isinstance(val, (int, float)):
                feature_vals.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(feature_vals)
        y_rows.append(r[target_col])

    if not X_rows:
        raise ValueError(f"No valid rows found for target '{target_col}'.")

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y, numeric_cols

# ----------------------------------------------------------------------
# NASA-03: Predict Effort
# ----------------------------------------------------------------------
def nasa_03(header, rows):
    """
    Fit a linear regression model to predict Effort- from numeric features
    and print model diagnostics.
    """
    target = 'Effort-'
    exclude = ['idX', 'YearX', 'MonthsX']  # columns not useful as predictors

    X, y, feat_names = build_matrix(rows, target_col=target, exclude_cols=exclude)
    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    preds = model.predict(X)
    residuals = y - preds
    residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    print("== NASA-03 ==")
    print(f"Training samples: {len(y)}")
    print(f"R²: {r2:.4f}")
    print("Coefficients:")
    for name, coef in zip(feat_names, model.coef_):
        print(f"  {name} -> {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual Std. Dev: {residual_std:.4f}")
    print()

# ----------------------------------------------------------------------
# NASA-04: Predict Defects
# ----------------------------------------------------------------------
def nasa_04(header, rows):
    """
    Fit a linear regression model to predict Defects- from numeric features
    and print model diagnostics, including uncertainty estimate.
    """
    target = 'Defects-'
    exclude = ['idX', 'YearX', 'MonthsX']

    X, y, feat_names = build_matrix(rows, target_col=target, exclude_cols=exclude)
    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    preds = model.predict(X)
    residuals = y - preds
    residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    print("== NASA-04 ==")
    print(f"Training samples: {len(y)}")
    print(f"R²: {r2:.4f}")
    print("Coefficients:")
    for name, coef in zip(feat_names, model.coef_):
        print(f"  {name} -> {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual Std. Dev (Uncertainty): {residual_std:.4f}")
    print()

# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        header, rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        nasa_03(header, rows)
    except Exception as e:
        print(f"NASA-03 error: {e}", file=sys.stderr)

    try:
        nasa_04(header, rows)
    except Exception as e:
        print(f"NASA-04 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

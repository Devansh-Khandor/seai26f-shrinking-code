#!/usr/bin/env python3
"""
skill_02.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-03 – Predict Effort- for held‑out projects.
NASA-04 – Predict Defects- and an uncertainty band for held‑out projects.

Outputs:
  == NASA-03 ==
    ... at least three lines of results ...
  ANSWER NASA-03 <JSON>

  == NASA-04 ==
    ... at least three lines of results ...
  ANSWER NASA-04 <JSON>
"""

import sys
import csv
import json
import math
from statistics import mean
from math import isnan

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_value(v):
    """Parse a cell value; return float or None for missing."""
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def mean_absolute_error(y_true, y_pred):
    """Compute MAE for two lists of numbers."""
    errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
    return mean(errors)

# --------------------------------------------------------------------------- #
# Main analysis functions
# --------------------------------------------------------------------------- #

def load_csv(csv_path):
    """Read CSV and return list of rows (dict) with parsed numeric values."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                if col.endswith('X'):   # non‑input columns
                    parsed[col] = None
                else:
                    parsed[col] = parse_value(raw[col])
            parsed['row_num'] = idx
            rows.append(parsed)
    return rows, all_columns

def get_feature_columns(all_columns, exclude=None):
    """Return column names to be used as features."""
    exclude = set(exclude or [])
    features = [c for c in all_columns if not c.endswith('X')
                and c not in exclude and c not in ('idX', 'YearX', 'MonthsX')]
    return features

def train_model(rows, feature_cols, target_col):
    """
    Train a linear regression model on rows with non‑None target values.
    Return the fitted model and the residual standard deviation on training data.
    """
    X = []
    y = []
    for r in rows:
        if r[target_col] is None:
            continue
        x = [r[c] for c in feature_cols]
        # Replace missing feature values with None; imputer will handle
        X.append(x)
        y.append(r[target_col])

    if not X:
        # No training data; return None
        return None, None

    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        LinearRegression()
    )
    X = np.array(X)
    y = np.array(y)
    pipeline.fit(X, y)

    # Compute residual std on training data
    y_pred = pipeline.predict(X)
    residuals = y - y_pred
    resid_std = np.sqrt(np.mean(residuals ** 2))
    return pipeline, resid_std

def predict_rows(model, rows, feature_cols):
    """Return predictions for the given rows (list of dict)."""
    X = []
    row_nums = []
    for r in rows:
        row_nums.append(r['row_num'])
        X.append([r[c] for c in feature_cols])
    X = np.array(X)
    preds = model.predict(X)
    return dict(zip(row_nums, preds))

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, all_columns = load_csv(csv_path)
    total_rows = len(rows)

    # Identify held‑out rows (divisible by 5)
    held_out_rows = [r for r in rows if r['row_num'] % 5 == 0]
    non_held_rows = [r for r in rows if r['row_num'] % 5 != 0]

    # ----------------------------------------------------------------------- #
    # NASA-03 – Predict Effort-
    # ----------------------------------------------------------------------- #
    effort_features = get_feature_columns(all_columns,
                                          exclude=('Effort-', 'Defects-'))
    effort_model, _ = train_model(non_held_rows, effort_features, 'Effort-')
    if effort_model is not None:
        effort_preds = predict_rows(effort_model, held_out_rows, effort_features)
        # Compute MAE against actual Effort- values
        y_true = [r['Effort-'] for r in held_out_rows if r['Effort-'] is not None]
        y_pred = [effort_preds[r['row_num']] for r in held_out_rows if r['Effort-'] is not None]
        mae_effort = mean_absolute_error(y_true, y_pred)
    else:
        effort_preds = {}
        mae_effort = 0.0

    print("== NASA-03 ==")
    # Show first few predictions
    for i, (row_num, val) in enumerate(effort_preds.items()):
        print(f"Row {row_num} Effort- predicted: {val:.2f}")
        if i >= 5:   # keep output short
            break
    print(f"Mean Absolute Error: {mae_effort:.2f}")
    print("ANSWER NASA-03", json.dumps(
        {"predictions": {str(k): float(v) for k, v in effort_preds.items()},
         "mae": float(mae_effort)},
        ensure_ascii=False))

    # ----------------------------------------------------------------------- #
    # NASA-04 – Predict Defects- with uncertainty
    # ----------------------------------------------------------------------- #
    defects_features = get_feature_columns(all_columns,
                                          exclude=('Defects-', 'Effort-'))
    defects_model, resid_std = train_model(non_held_rows, defects_features, 'Defects-')
    if defects_model is not None:
        defects_preds = predict_rows(defects_model, held_out_rows, defects_features)
        # Compute confidence interval (95%) using residual std
        ci = 1.96 * resid_std
        defects_predictions = {}
        for row_num, pred in defects_preds.items():
            low = max(0.0, pred - ci)
            high = pred + ci
            defects_predictions[row_num] = {
                "estimate": float(pred),
                "low": float(low),
                "high": float(high)
            }
    else:
        defects_predictions = {}

    print("\n== NASA-04 ==")
    # Show first few predictions
    for i, (row_num, d) in enumerate(defects_predictions.items()):
        print(f"Row {row_num} Defects- estimate: {d['estimate']:.2f}, "
              f"95% CI: [{d['low']:.2f}, {d['high']:.2f}]")
        if i >= 5:
            break
    print("ANSWER NASA-04", json.dumps(
        {"predictions": {str(k): v for k, v in defects_predictions.items()}},
        ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

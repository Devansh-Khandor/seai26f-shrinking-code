#!/usr/bin/env python3
import sys
import csv
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Utility functions for parsing numeric values
# ----------------------------------------------------------------------
def parse_float(val):
    """
    Convert a CSV string value to a float, handling missing data.
    Returns None if value is missing or cannot be parsed as float.
    """
    if val is None or val == '':
        return None
    val = val.strip()
    if val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Main analysis functions
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV file and return:
        - data: list of (row_index, row_dict) tuples, where row_index is 1-based.
        - fieldnames: list of column names from the header.
    """
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames

def filter_rows(data, heldout_mask, target, exclude_missing=True):
    """
    Build X (features) and y (target) arrays from data, excluding heldout rows.
    Parameters:
        - data: list of (row_index, row_dict)
        - heldout_mask: set of row indices that are held‑out
        - target: column name of target variable
        - exclude_missing: if True, rows with missing target or any feature are discarded
    Returns:
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,)
        indices: list of row indices corresponding to rows in X
    """
    X_list = []
    y_list = []
    idx_list = []

    for row_idx, row in data:
        if row_idx in heldout_mask:
            continue
        # target must be present
        t_val = parse_float(row[target])
        if t_val is None:
            continue
        # Build feature vector: all columns except those ending with 'X' and the target
        feat_vals = []
        missing_feature = False
        for col, val in row.items():
            if col == target or col.endswith('X'):
                continue
            f_val = parse_float(val)
            if f_val is None:
                missing_feature = True
                break
            feat_vals.append(f_val)
        if missing_feature:
            continue
        X_list.append(feat_vals)
        y_list.append(t_val)
        idx_list.append(row_idx)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y, idx_list

def predict_heldout(X, y, idx, data, heldout_mask, target, model_cls):
    """
    Train model on all non‑held‑out data (X, y), then predict for held‑out rows.
    Returns:
        predictions: dict mapping row_index -> predicted value (float)
        actuals: dict mapping row_index -> actual target value (float) for rows where present
    """
    model = model_cls()
    model.fit(X, y)

    predictions = {}
    actuals = {}
    for row_idx, row in data:
        if row_idx not in heldout_mask:
            continue
        # Prepare feature vector
        feat_vals = []
        missing_feature = False
        for col, val in row.items():
            if col == target or col.endswith('X'):
                continue
            f_val = parse_float(val)
            if f_val is None:
                missing_feature = True
                break
            feat_vals.append(f_val)
        if missing_feature or len(feat_vals) != X.shape[1]:
            # cannot predict due to missing feature
            continue
        pred = model.predict(np.array([feat_vals]))[0]
        predictions[row_idx] = pred

        # actual if present
        act_val = parse_float(row[target])
        if act_val is not None:
            actuals[row_idx] = act_val

    return predictions, actuals

def compute_mae(predictions, actuals):
    """
    Compute mean absolute error over the intersection of keys in predictions and actuals.
    """
    errors = []
    for key in predictions:
        if key in actuals:
            errors.append(abs(predictions[key] - actuals[key]))
    if not errors:
        return None
    return sum(errors) / len(errors)

def compute_confidence_intervals(model, X_train, y_train, X_test, confidence=0.95):
    """
    Compute prediction intervals assuming normal distribution of residuals.
    Returns list of (low, high) for each sample in X_test.
    """
    # Predictions on training data
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    std_err = np.sqrt(np.mean(residuals ** 2))
    # z-score for given confidence
    z = 1.96  # approximate for 95%
    y_pred_test = model.predict(X_test)
    low = y_pred_test - z * std_err
    high = y_pred_test + z * std_err
    return low, high

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    # Determine held‑out rows: divisible by 5
    heldout_mask = {idx for idx, _ in data if idx % 5 == 0}

    # ------------------------------------------------------------------
    # NASA‑03: Predict Effort-
    # ------------------------------------------------------------------
    print("== NASA-03 ==")
    # Build training data for Effort-
    X_eff, y_eff, eff_idx = filter_rows(data, heldout_mask, 'Effort-')
    # Predict on held‑out
    predictions_eff, actuals_eff = predict_heldout(
        X_eff, y_eff, eff_idx, data, heldout_mask, 'Effort-', LinearRegression)
    mae_eff = compute_mae(predictions_eff, actuals_eff)
    print(f"MAE on held‑out Effort- predictions: {mae_eff:.2f}")
    # machine‑checked answer
    pred_dict = {str(k): float(v) for k, v in predictions_eff.items()}
    ans_eff = {"predictions": pred_dict, "mae": mae_eff}
    print(f"ANSWER NASA-03 {json.dumps(ans_eff, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # NASA‑04: Predict Defects- with uncertainty
    # ------------------------------------------------------------------
    print("\n== NASA-04 ==")
    X_def, y_def, def_idx = filter_rows(data, heldout_mask, 'Defects-')
    # Train model
    model_def = LinearRegression()
    model_def.fit(X_def, y_def)

    # Predict and intervals for held‑out
    predictions_def = {}
    low_dict = {}
    high_dict = {}

    for row_idx, row in data:
        if row_idx not in heldout_mask:
            continue
        # Feature extraction
        feat_vals = []
        missing_feature = False
        for col, val in row.items():
            if col == 'Defects-' or col.endswith('X'):
                continue
            f_val = parse_float(val)
            if f_val is None:
                missing_feature = True
                break
            feat_vals.append(f_val)
        if missing_feature or len(feat_vals) != X_def.shape[1]:
            continue
        pred_val = model_def.predict(np.array([feat_vals]))[0]
        predictions_def[row_idx] = pred_val

    # Compute residual standard error from training set
    y_pred_train = model_def.predict(X_def)
    residuals = y_def - y_pred_train
    std_err = math.sqrt(np.mean(residuals ** 2))
    z = 1.96  # 95% CI
    # Compute intervals
    for row_idx, pred in predictions_def.items():
        low = pred - z * std_err
        high = pred + z * std_err
        low_dict[row_idx] = low
        high_dict[row_idx] = high

    # Actual Defects- values for held‑out rows (if present)
    actuals_def = {}
    for row_idx, row in data:
        if row_idx in heldout_mask:
            act_val = parse_float(row['Defects-'])
            if act_val is not None:
                actuals_def[row_idx] = act_val

    # Print sample output
    num_preds = len(predictions_def)
    print(f"Predicted Defects- for {num_preds} held‑out rows.")
    # machine‑checked answer
    pred_output = {}
    for row_idx in predictions_def:
        pred_output[str(row_idx)] = {
            "estimate": float(predictions_def[row_idx]),
            "low": float(low_dict[row_idx]),
            "high": float(high_dict[row_idx])
        }
    ans_def = {"predictions": pred_output}
    print(f"ANSWER NASA-04 {json.dumps(ans_def, separators=(',', ':'))}")

if __name__ == "__main__":
    main()

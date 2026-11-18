#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv path>

Extends the original analysis to provide:
- CAR-03: Predict missing Mpg+ (fuel‑efficiency) using regression.
- CAR-04: Predict missing Lbs- (vehicle weight) and provide a 95% uncertainty band.

The script prints each section header, a few descriptive lines, and a
machine‑checked ANSWER line in JSON format.
"""

import sys
import json
import numpy as np

# Reuse helpers from skill_01
from skill_01 import read_csv, parse_float, collect_column_values

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def get_col_indices(header, include=None, exclude=None):
    """
    Return list of indices for columns in `header`.
    If `include` is provided, only those names are considered.
    If `exclude` is provided, those names are omitted.
    """
    idxs = []
    for i, name in enumerate(header):
        if include is not None and name not in include:
            continue
        if exclude is not None and name in exclude:
            continue
        idxs.append(i)
    return idxs

def extract_features(row, idxs):
    """
    Extract values for the given column indices from `row`.
    Return None if any value is missing.
    """
    vals = []
    for i in idxs:
        v = parse_float(row[i])
        if v is None:
            return None
        vals.append(v)
    return vals

def train_regression(X, y):
    """
    Train a simple linear regression (no intercept handling)
    using scikit‑learn. Return the fitted model.
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    return model

def prediction_stats(preds, truths):
    """
    Compute mean absolute error between predictions and truths.
    """
    errors = [abs(p - t) for p, t in zip(preds, truths)]
    if not errors:
        return 0.0
    return sum(errors) / len(errors)

# --------------------------------------------------------------------------- #
# Main logic for CAR‑03 and CAR‑04
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)
    n_rows = len(data)

    # Identify indices
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Feature columns: all that are not ending in 'X', 'Mpg+', 'Lbs-'
    # (we use the same feature set for both predictions)
    feature_names = [name for name in header
                     if not name.endswith('X')
                     and name not in ('Mpg+', 'Lbs-')]

    feature_idx = get_col_indices(header, include=feature_names)

    # -------------------------------------------------- CAR-03 --------------------------------------------------
    # Train to predict Mpg+
    X_mpg = []
    y_mpg = []
    for i, row in enumerate(data, start=1):
        if i % 5 == 0:          # held‑out
            continue
        feat = extract_features(row, feature_idx)
        if feat is None:
            continue
        mpg_val = parse_float(row[idx_mpg])
        if mpg_val is None:
            continue
        X_mpg.append(feat)
        y_mpg.append(mpg_val)

    model_mpg = None
    if X_mpg:  # ensure we have data to train
        model_mpg = train_regression(np.array(X_mpg), np.array(y_mpg))

    predictions_mpg = {}
    mae = 0.0
    true_vals_mpg = []
    pred_vals_mpg = []

    if model_mpg:
        for i, row in enumerate(data, start=1):
            if i % 5 == 0:          # held‑out
                feat = extract_features(row, feature_idx)
                if feat is None:
                    continue
                pred = model_mpg.predict([feat])[0]
                predictions_mpg[str(i)] = pred
                true_val = parse_float(row[idx_mpg])
                if true_val is not None:
                    pred_vals_mpg.append(pred)
                    true_vals_mpg.append(true_val)
        mae = prediction_stats(pred_vals_mpg, true_vals_mpg)

    # Print CAR‑03 section
    print("== CAR-03 ==")
    print(f"Mean absolute error (MAE) on held‑out rows: {mae:.3f}")
    if predictions_mpg:
        example_row = next(iter(predictions_mpg))
        example_pred = predictions_mpg[example_row]
        example_true = parse_float(data[int(example_row)-1][idx_mpg])
        print(f"Example: Row {example_row} predicted mpg={example_pred:.2f} (actual={example_true})")
    else:
        print("No predictions could be made.")

    # Machine‑checked answer
    answer_car03 = {
        "predictions": predictions_mpg,
        "mae": mae
    }
    print(f"ANSWER CAR-03 {json.dumps(answer_car03, separators=(',', ':'), sort_keys=True)}")

    # -------------------------------------------------- CAR-04 --------------------------------------------------
    # Train to predict Lbs-
    X_lbs = []
    y_lbs = []
    for i, row in enumerate(data, start=1):
        if i % 5 == 0:          # held‑out
            continue
        feat = extract_features(row, feature_idx)
        if feat is None:
            continue
        lbs_val = parse_float(row[idx_lbs])
        if lbs_val is None:
            continue
        X_lbs.append(feat)
        y_lbs.append(lbs_val)

    model_lbs = None
    std_residual = 0.0
    predictions_lbs = {}

    if X_lbs:
        model_lbs = train_regression(np.array(X_lbs), np.array(y_lbs))
        # Compute residuals to estimate uncertainty
        y_pred_train = model_lbs.predict(np.array(X_lbs))
        residuals = np.array(y_lbs) - y_pred_train
        std_residual = float(np.std(residuals, ddof=0))

        # Predict for held‑out rows
        for i, row in enumerate(data, start=1):
            if i % 5 == 0:  # held‑out
                feat = extract_features(row, feature_idx)
                if feat is None:
                    continue
                est = model_lbs.predict([feat])[0]
                low = est - 1.96 * std_residual
                high = est + 1.96 * std_residual
                predictions_lbs[str(i)] = {
                    "estimate": est,
                    "low": low,
                    "high": high
                }

    # Print CAR‑04 section
    print("\n== CAR-04 ==")
    print(f"Standard deviation of residuals on training set: {std_residual:.2f}")
    if predictions_lbs:
        example_row = next(iter(predictions_lbs))
        example_est = predictions_lbs[example_row]["estimate"]
        example_low = predictions_lbs[example_row]["low"]
        example_high = predictions_lbs[example_row]["high"]
        print(f"Example: Row {example_row} estimate lbs={example_est:.2f} "
              f"(95% range: {example_low:.2f} – {example_high:.2f})")
    else:
        print("No weight predictions could be made.")

    # Machine‑checked answer
    answer_car04 = {
        "predictions": predictions_lbs
    }
    print(f"ANSWER CAR-04 {json.dumps(answer_car04, separators=(',', ':'), sort_keys=True)}")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

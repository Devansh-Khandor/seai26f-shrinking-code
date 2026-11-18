#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑03 and CAR‑04).

Usage:
    python skill_02.py <path_to_csv>
"""

import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Import common utilities from the previous skill
from skill_01 import load_data, GOAL_COLUMNS

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def get_held_out_rows(total_rows):
    """
    Return a sorted list of 1‑based row numbers that are divisible by 5.
    """
    return [i for i in range(1, total_rows + 1) if i % 5 == 0]


def prepare_features(data, feature_cols, target_col):
    """
    Build X and y matrices for regression.
    Only rows that have non‑None values for the target and for all feature columns
    are considered for training.
    Missing feature values are imputed with the column mean (computed on training data).
    """
    X_list, y_list = [], []
    # First collect raw data and compute means for each feature
    feature_means = {col: [] for col in feature_cols}
    for row in data:
        val = row["values"][target_col]
        if val is None:
            continue
        skip = False
        features = []
        for col in feature_cols:
            f_val = row["values"][col]
            if f_val is None:
                skip = True
                break
            features.append(f_val)
            feature_means[col].append(f_val)
        if skip:
            continue
        X_list.append(features)
        y_list.append(val)

    if not X_list:
        return None, None, None  # No training data

    X = np.array(X_list)
    y = np.array(y_list)

    # Impute means for any remaining missing in training set (unlikely)
    for i, col in enumerate(feature_cols):
        mean_val = np.mean(feature_means[col])
        X[:, i] = np.where(np.isnan(X[:, i]), mean_val, X[:, i])

    # Compute feature means for imputation during prediction
    feature_means_vals = {col: np.mean(feature_means[col]) for col in feature_cols}
    return X, y, feature_means_vals


def predict_with_model(model, data, feature_cols, feature_means, target_col, held_out_rows):
    """
    Return a dictionary mapping row number (string) to predicted value.
    Missing features in a held‑out row are imputed with the training feature means.
    """
    predictions = {}
    for row_num in held_out_rows:
        # Find row in data
        row = data[row_num - 1]  # 0‑based index
        X_row = []
        for col in feature_cols:
            f_val = row["values"][col]
            if f_val is None:
                f_val = feature_means[col]
            X_row.append(f_val)
        X_row = np.array([X_row])
        pred = model.predict(X_row)[0]
        predictions[str(row_num)] = float(pred)
    return predictions


def compute_mae(predictions, data, target_col, held_out_rows):
    """
    Compute mean absolute error between predictions and true target values
    for those held‑out rows where the true value is available.
    """
    errors = []
    for row_num_str, pred in predictions.items():
        row_num = int(row_num_str)
        true_val = data[row_num - 1]["values"][target_col]
        if true_val is None:
            continue
        errors.append(abs(pred - true_val))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def predict_with_intervals(model, data, feature_cols, feature_means, target_col, held_out_rows, n_std=2.0):
    """
    Predict values and return estimate, low, high for each held‑out row.
    The interval is estimate ± n_std * residual_std from training data.
    """
    # Build training set again to compute residuals
    X_train = []
    y_train = []
    for row in data:
        val = row["values"][target_col]
        if val is None:
            continue
        skip = False
        features = []
        for col in feature_cols:
            f_val = row["values"][col]
            if f_val is None:
                skip = True
                break
            features.append(f_val)
        if skip:
            continue
        X_train.append(features)
        y_train.append(val)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Impute training missing if any
    for i, col in enumerate(feature_cols):
        mean_val = np.mean([row["values"][col] for row in data if row["values"][col] is not None])
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), mean_val, X_train[:, i])

    # Predict on training data
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    residual_std = np.sqrt(np.mean(residuals ** 2))

    predictions = {}
    for row_num in held_out_rows:
        row = data[row_num - 1]
        X_row = []
        for col in feature_cols:
            f_val = row["values"][col]
            if f_val is None:
                f_val = feature_means[col]
            X_row.append(f_val)
        X_row = np.array([X_row])
        est = model.predict(X_row)[0]
        low = est - n_std * residual_std
        high = est + n_std * residual_std
        predictions[str(row_num)] = {
            "estimate": float(est),
            "low": float(low),
            "high": float(high),
        }
    return predictions


# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)
    total_rows = len(data)
    held_out_rows = get_held_out_rows(total_rows)

    # ---- CAR-03: Predict Mpg+ for held-out rows ----
    mpg_feature_cols = ["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs-", "Acc+"]
    X_mpg, y_mpg, mpg_means = prepare_features(data, mpg_feature_cols, "Mpg+")
    if X_mpg is None:
        mpg_predictions = {}
        mpg_mae = 0.0
    else:
        mpg_model = LinearRegression().fit(X_mpg, y_mpg)
        mpg_predictions = predict_with_model(mpg_model, data, mpg_feature_cols, mpg_means, "Mpg+", held_out_rows)
        mpg_mae = compute_mae(mpg_predictions, data, "Mpg+", held_out_rows)

    print("== CAR-03 ==")
    # Provide at least three lines of output
    print(f"Number of held‑out rows: {len(held_out_rows)}")
    print(f"Mean absolute error on held‑out rows: {mpg_mae:.4f}")
    # Sample predictions
    sample = list(mpg_predictions.items())[:3]
    for row_num, pred in sample:
        print(f"Row {row_num} predicted Mpg+: {pred:.2f}")

    # Machine‑checked answer
    answer_json = json.dumps({"predictions": mpg_predictions, "mae": mpg_mae}, separators=(",", ":"))
    print(f"ANSWER CAR-03 {answer_json}")

    # ---- CAR-04: Predict Lbs- with uncertainty ----
    lbs_feature_cols = ["Clndrs", "Volume", "HpX", "Model", "origin", "Acc+", "Mpg+"]
    X_lbs, y_lbs, lbs_means = prepare_features(data, lbs_feature_cols, "Lbs-")
    if X_lbs is None:
        lbs_predictions = {}
    else:
        lbs_model = LinearRegression().fit(X_lbs, y_lbs)
        lbs_predictions = predict_with_intervals(lbs_model, data, lbs_feature_cols, lbs_means, "Lbs-", held_out_rows)

    print("\n== CAR-04 ==")
    # Provide at least three lines of output
    print(f"Number of held‑out rows: {len(held_out_rows)}")
    print(f"Sample weight estimate for row {held_out_rows[0]}: {lbs_predictions[str(held_out_rows[0])]['estimate']:.2f}")
    print(f"Sample low/high range for row {held_out_rows[0]}: {lbs_predictions[str(held_out_rows[0])]['low']:.2f} / {lbs_predictions[str(held_out_rows[0])]['high']:.2f}")

    # Machine‑checked answer
    answer_json = json.dumps({"predictions": lbs_predictions}, separators=(",", ":"))
    print(f"ANSWER CAR-04 {answer_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

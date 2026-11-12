#!/usr/bin/env python3
"""
Skill 02: Apache performance estimation & comparison
"""

import sys
import csv
import numpy as np
from sklearn.linear_model import LinearRegression

# Import helpers from the original skill_01
from skill_01 import read_csv_to_numpy, summarize_performance, identify_favorable


# --------------------------------------------------------------------------- #
# Model training and prediction helpers
# --------------------------------------------------------------------------- #

def train_linear_model(data, header):
    """
    Train a simple linear regression model (features: all columns except last).
    Returns the fitted model, the inverse (X^T X)^-1 matrix, and the residual variance.
    """
    X = data[:, :-1]  # all features
    y = data[:, -1]   # performance

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Residuals and variance
    residuals = y - model.predict(X)
    resid_var = np.mean(residuals ** 2)

    # Compute (X^T X)^-1 for prediction variance
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)

    return model, XtX_inv, resid_var


def predict_with_confidence(model, XtX_inv, resid_var, x_row, confidence=0.95):
    """
    Predict the performance for a single configuration row (numpy array of features).
    Returns the point estimate and the lower/upper bounds of the confidence interval.
    Uses the standard linear regression prediction variance formula.
    """
    pred = model.predict(x_row.reshape(1, -1))[0]

    # Compute prediction variance
    x = x_row.reshape(1, -1)
    var_pred = resid_var * (1 + (x @ XtX_inv @ x.T)[0, 0])

    # Confidence interval using z-score
    import scipy.stats
    z = scipy.stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z * np.sqrt(var_pred)

    return pred, pred - margin, pred + margin


# --------------------------------------------------------------------------- #
# APACHE-03: Estimate performance for sample configurations
# --------------------------------------------------------------------------- #

def apache_03(header, data, model, XtX_inv, resid_var):
    print("\n== APACHE-03 ==")
    # We'll demonstrate on the first three configurations
    n_show = min(3, data.shape[0])
    for i in range(n_show):
        x_row = data[i, :-1]
        true_perf = data[i, -1]
        pred, low, high = predict_with_confidence(model, XtX_inv, resid_var, x_row)
        print(f"Config {i} (actual: {true_perf}) -> estimated: {pred:.2f} ± {high - pred:.2f} (95% CI: [{low:.2f}, {high:.2f}])")


# --------------------------------------------------------------------------- #
# APACHE-04: Compare pairs differing by one setting
# --------------------------------------------------------------------------- #

def find_one_diff_pairs(data):
    """
    Find all pairs of rows that differ in exactly one feature (excluding performance).
    Returns a list of tuples (idx1, idx2, feature_index, diff).
    """
    pairs = []
    n = data.shape[0]
    feats = data[:, :-1]
    for i in range(n):
        for j in range(i + 1, n):
            diff_mask = np.abs(feats[i] - feats[j])  # difference per feature
            if diff_mask.sum() == 1:
                feature_idx = np.argmax(diff_mask)
                diff = data[j, -1] - data[i, -1]
                pairs.append((i, j, feature_idx, diff))
    return pairs


def apache_04(header, data, model, XtX_inv, resid_var):
    print("\n== APACHE-04 ==")
    pairs = find_one_diff_pairs(data)
    if not pairs:
        print("No configuration pairs differ by exactly one setting.")
        return

    # Compute predicted differences and actual differences
    preds = model.predict(data[:, :-1])
    actual_diffs = []
    predicted_diffs = []

    for i, j, feat_idx, _ in pairs:
        actual_diff = data[j, -1] - data[i, -1]
        predicted_diff = preds[j] - preds[i]
        actual_diffs.append(actual_diff)
        predicted_diffs.append(predicted_diff)

    actual_diffs = np.array(actual_diffs)
    predicted_diffs = np.array(predicted_diffs)

    mean_pred_diff = predicted_diffs.mean()
    std_pred_diff = predicted_diffs.std(ddof=1)

    mean_actual_diff = actual_diffs.mean()
    std_actual_diff = actual_diffs.std(ddof=1)

    print(f"Found {len(pairs)} configuration pairs differing by one setting.")
    print(f"Mean actual performance difference: {mean_actual_diff:.2f} (std: {std_actual_diff:.2f})")
    print(f"Mean predicted performance difference: {mean_pred_diff:.2f} (std: {std_pred_diff:.2f})")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Summary from previous skill
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Train model
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # APACHE-03
    apache_03(header, data, model, XtX_inv, resid_var)

    # APACHE-04
    apache_04(header, data, model, XtX_inv, resid_var)


if __name__ == "__main__":
    main()

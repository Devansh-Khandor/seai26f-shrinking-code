#!/usr/bin/env python3
"""
skill_02.py – Performance prediction and pairwise comparison for Apache configurations.

Usage:
    python skill_02.py <path-to-Apache_AllMeasurements.csv>
"""

import sys
import csv
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import skill_01 as s1  # Import existing utilities

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_csv(path):
    """Load CSV into a list of dict rows and return column names."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    return rows, headers

def feature_matrix(rows, headers):
    """Return numpy array of features for each row (excluding the target)."""
    target_col = 'Performance-'
    feature_cols = [h for h in headers if h != target_col]
    X = []
    for row in rows:
        # Convert all feature values to int (binary flags)
        X.append([int(row[col]) for col in feature_cols])
    return np.array(X), feature_cols

def target_vector(rows, headers):
    """Return numpy array of target values (Performance-)"""
    target_col = 'Performance-'
    y = [int(row[target_col]) for row in rows]
    return np.array(y)

def train_regressor(X, y):
    """Train a simple linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_performance(model, X):
    """Return predicted performance values."""
    return model.predict(X)

def residual_std(model, X, y):
    """Return standard deviation of residuals (observed - predicted)."""
    preds = model.predict(X)
    residuals = y - preds
    return np.std(residuals)

def find_one_setting_pairs(rows, feature_cols):
    """Yield tuples (idx1, idx2, differing_col) for pairs that differ by one setting."""
    n = len(rows)
    for i, j in itertools.combinations(range(n), 2):
        diff_count = 0
        diff_col = None
        for col in feature_cols:
            if rows[i][col] != rows[j][col]:
                diff_count += 1
                diff_col = col
            if diff_count > 1:
                break
        if diff_count == 1:
            yield (i, j, diff_col)

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)
    # Build feature matrix and target vector
    X, feature_cols = feature_matrix(rows, headers)
    y = target_vector(rows, headers)

    # Train model
    model = train_regressor(X, y)
    preds = predict_performance(model, X)
    resid_std = residual_std(model, X, y)

    # ------------------------------------------------------------------
    # APACHE-03: Performance prediction for configurations
    # ------------------------------------------------------------------
    print("== APACHE-03 ==")
    median_actual = np.median(y)
    median_predicted = np.median(preds)
    mse = mean_squared_error(y, preds)
    print(f"Median actual performance: {median_actual:.2f}")
    print(f"Median predicted performance: {median_predicted:.2f}")
    print(f"Mean squared error of predictions: {mse:.2f}")

    # Show predictions for the first 3 configurations
    print("\nSample predictions (first 3 configurations):")
    for idx in range(min(3, len(rows))):
        print(f"Config #{idx+1} | Actual: {y[idx]:.2f} | Predicted: {preds[idx]:.2f}")

    # ------------------------------------------------------------------
    # APACHE-04: Pairwise comparison for one-setting differences
    # ------------------------------------------------------------------
    print("\n== APACHE-04 ==")
    pairs = list(find_one_setting_pairs(rows, feature_cols))
    print(f"Total pairs differing by one setting: {len(pairs)}")

    if pairs:
        print("\nFirst 3 pairs and their estimated performance differences:")
        for pair_idx, (i, j, diff_col) in enumerate(pairs[:3], start=1):
            actual_diff = abs(y[i] - y[j])
            pred_diff = abs(preds[i] - preds[j])
            uncertainty = resid_std  # Simplified uncertainty estimate
            print(f"Pair #{pair_idx}: differing column '{diff_col}'")
            print(f"  Actual performance difference: {actual_diff:.2f}")
            print(f"  Estimated performance difference: {pred_diff:.2f}")
            print(f"  Uncertainty (std of residuals): ±{uncertainty:.2f}\n")
    else:
        print("No pairs found that differ by exactly one setting.")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_02.py <csv_path>")
    main(sys.argv[1])

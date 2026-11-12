#!/usr/bin/env python3
"""
skill_02.py

Entry point for CAR-03 and CAR-04 skill requests.

Usage:
    python skill_02.py <path/to/auto93.csv>
"""

import argparse
import csv
import sys
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# Utility functions (reuse from skill_01 where possible)
# --------------------------------------------------------------------------- #

def read_csv_as_array(path: str) -> np.ndarray:
    """
    Reads a CSV file containing only numeric values into a NumPy array.
    Treats empty fields as np.nan.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            processed_row = []
            for cell in row:
                if cell.strip() == "":
                    processed_row.append(np.nan)
                else:
                    try:
                        processed_row.append(float(cell))
                    except ValueError:
                        processed_row.append(np.nan)
            data.append(processed_row)
    return np.array(data, dtype=float)

def compute_summary_stats(arr: np.ndarray, col_index: int) -> Tuple[float, float, float]:
    """
    Computes mean, min, and max for a specified column, ignoring NaNs.
    """
    col = arr[:, col_index]
    mean = float(np.nanmean(col))
    min_val = float(np.nanmin(col))
    max_val = float(np.nanmax(col))
    return mean, min_val, max_val

def compute_composite_score(arr: np.ndarray) -> Tuple[np.ndarray, List[int], List[float]]:
    """
    Computes a composite score for each vehicle based on
    - Weight (column 5): lower is better
    - Fuel Efficiency (column 7): higher is better
    - Acceleration (column 6): lower is better
    Returns:
        scores: array of composite scores (higher is better)
        top_indices: list of indices of top 5 vehicles (descending order)
        top_scores: list of scores for those vehicles
    """
    weight = arr[:, 5].reshape(-1, 1)
    fuel = arr[:, 7].reshape(-1, 1)
    acc = arr[:, 6].reshape(-1, 1)

    # Normalize each feature to [0, 1]
    scaler_w = MinMaxScaler()
    scaler_f = MinMaxScaler()
    scaler_a = MinMaxScaler()

    norm_weight = 1 - scaler_w.fit_transform(weight)  # lower is better
    norm_fuel = scaler_f.fit_transform(fuel)          # higher is better
    norm_acc = 1 - scaler_a.fit_transform(acc)        # lower is better

    # Composite score: equal weights
    scores = (norm_weight + norm_fuel + norm_acc).flatten()

    # Get top 5 indices (descending)
    top_indices = np.argsort(scores)[::-1][:5]
    top_scores = scores[top_indices]
    return scores, list(top_indices), list(top_scores)

# --------------------------------------------------------------------------- #
# CAR-03: Predict missing fuel efficiency
# --------------------------------------------------------------------------- #

def predict_missing_fuel(arr: np.ndarray) -> None:
    """
    Detects rows with missing fuel efficiency (column 7) and estimates them
    using linear regression trained on available data.
    Prints results.
    """
    print("== CAR-03 ==")
    # Identify rows with known and missing fuel
    known_mask = ~np.isnan(arr[:, 7])
    missing_mask = np.isnan(arr[:, 7])

    if not np.any(missing_mask):
        print("No missing fuel efficiency values found.")
        return

    # Prepare training data
    X_train = arr[known_mask]
    y_train = arr[known_mask, 7]
    # Use all columns except fuel (column 7) as features
    X_train_feat = np.delete(X_train, 7, axis=1)

    # Train linear regression
    lr = LinearRegression()
    lr.fit(X_train_feat, y_train)

    # Predict for missing rows
    X_missing = arr[missing_mask]
    X_missing_feat = np.delete(X_missing, 7, axis=1)
    y_pred = lr.predict(X_missing_feat)

    # Compute training MSE
    y_train_pred = lr.predict(X_train_feat)
    mse = mean_squared_error(y_train, y_train_pred)

    # Print predictions
    missing_indices = np.where(missing_mask)[0]
    for idx, pred in zip(missing_indices, y_pred):
        print(f"Index {idx}: predicted mpg = {pred:.2f}")

    # Print metric
    print(f"Training MSE (fuel): {mse:.3f}")

# --------------------------------------------------------------------------- #
# CAR-04: Predict weight and uncertainty
# --------------------------------------------------------------------------- #

def predict_weight_with_uncertainty(arr: np.ndarray) -> None:
    """
    Estimates vehicle weight (column 5) using linear regression based on
    all other available characteristics, and reports the standard deviation
    of prediction errors as a measure of uncertainty.
    Prints results.
    """
    print("== CAR-04 ==")
    # Prepare training data
    X = np.delete(arr, 5, axis=1)  # all columns except weight
    y = arr[:, 5]

    # Train linear regression
    lr = LinearRegression()
    lr.fit(X, y)

    # Predict on the training set to compute residuals
    y_pred_train = lr.predict(X)
    residuals = y - y_pred_train
    sigma = float(np.std(residuals))  # standard deviation of residuals

    # Predict for all rows (including those already known)
    y_pred = lr.predict(X)

    # Print example predictions
    # We'll print first 5 predictions as demonstration
    for idx in range(min(5, len(y_pred))):
        print(f"Index {idx}: predicted weight = {y_pred[idx]:.1f} lbs ± {sigma:.1f} lbs")

    # Print overall uncertainty
    print(f"Prediction standard deviation (uncertainty): {sigma:.2f} lbs")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="CAR-03 & CAR-04 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    # Load data
    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # CAR-03
    predict_missing_fuel(data)

    # CAR-04
    predict_weight_with_uncertainty(data)

if __name__ == "__main__":
    main()

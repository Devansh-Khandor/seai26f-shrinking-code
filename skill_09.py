#!/usr/bin/env python3
"""
Skill 09: Anomaly Detection (CAR-17 & CAR-18)

Usage:
    python skill_09.py <path_to_auto93.csv>
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Data loading helper
# --------------------------------------------------------------------------- #

def load_csv(path: str) -> np.ndarray:
    """Load the CSV into a structured NumPy array with numeric columns as float."""
    try:
        data = np.genfromtxt(
            path,
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    numeric_cols = [
        'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
        'Lbs-', 'Acc+', 'Mpg+'
    ]
    for col in numeric_cols:
        if col in data.dtype.names:
            try:
                data[col] = data[col].astype(float)
            except Exception:
                pass
    return data


# --------------------------------------------------------------------------- #
# Utility: Standardised features
# --------------------------------------------------------------------------- #

def standardize_columns(data: np.ndarray, cols: list) -> np.ndarray:
    """Return a (n_samples, n_cols) array of standardised columns (z‑score)."""
    arr = np.column_stack([data[col].astype(float) for col in cols])
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    # Avoid division by zero
    std[std == 0] = 1.0
    z = (arr - mean) / std
    return z


# --------------------------------------------------------------------------- #
# CAR-17: Unusual combination of characteristics
# --------------------------------------------------------------------------- #

def car_17(data: np.ndarray):
    print("== CAR-17 ==")

    # Columns to consider for the anomaly score
    feature_cols = ['Lbs-', 'Mpg+', 'Acc+', 'HpX', 'Volume', 'Model', 'origin']

    z = standardize_columns(data, feature_cols)

    # Compute Euclidean distance from mean (in z‑space)
    # Handle missing values: ignore rows with any NaNs in the feature set
    mask_valid = ~np.isnan(z).any(axis=1)
    if not np.any(mask_valid):
        print("No vehicle has complete data for CAR‑17 analysis.")
        return

    distances = np.linalg.norm(z[mask_valid], axis=1)

    # Get indices in the original array
    indices = np.where(mask_valid)[0]
    sorted_idx = np.argsort(-distances)[:5]  # top 5

    for rank, idx in enumerate(sorted_idx, start=1):
        print(f"  {rank}. Vehicle index {indices[idx]}: distance = {distances[idx]:.2f}")


# --------------------------------------------------------------------------- #
# CAR-18: Unexpected fuel efficiency or weight
# --------------------------------------------------------------------------- #

def car_18(data: np.ndarray):
    print("\n== CAR-18 ==")

    # Prepare data for regression
    # Exclude the target column from features
    cols_all = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                'Lbs-', 'Acc+', 'Mpg+']

    # Build feature matrix
    features = standardize_columns(data, cols_all)

    # Weight model (target weight)
    weight = data['Lbs-'].astype(float)
    weight_mask = ~np.isnan(weight)
    if np.any(weight_mask):
        X_weight = features[weight_mask]
        y_weight = weight[weight_mask]
        weight_model = LinearRegression()
        weight_model.fit(X_weight, y_weight)
        weight_pred = weight_model.predict(features)
        weight_resid = weight - weight_pred
        weight_std = np.nanstd(weight_resid[weight_mask], ddof=1)
        # Flag anomalies: |residual| > 2 * std
        weight_anom_mask = (np.abs(weight_resid) > 2 * weight_std) & weight_mask
        weight_anom_idx = np.where(weight_anom_mask)[0]
        print("  Weight anomalies:")
        if weight_anom_idx.size == 0:
            print("    None detected.")
        else:
            for idx in weight_anom_idx[:3]:
                actual = weight[idx]
                pred = weight_pred[idx]
                diff = actual - pred
                print(f"    Vehicle {idx}: weight = {actual:.1f} lbs, "
                      f"predicted = {pred:.1f} lbs, diff = {diff:+.1f} lbs")
    else:
        print("  No weight data available for anomaly detection.")

    # MPG model (target mpg)
    mpg = data['Mpg+'].astype(float)
    mpg_mask = ~np.isnan(mpg)
    if np.any(mpg_mask):
        X_mpg = features[mpg_mask]
        y_mpg = mpg[mpg_mask]
        mpg_model = LinearRegression()
        mpg_model.fit(X_mpg, y_mpg)
        mpg_pred = mpg_model.predict(features)
        mpg_resid = mpg - mpg_pred
        mpg_std = np.nanstd(mpg_resid[mpg_mask], ddof=1)
        mpg_anom_mask = (np.abs(mpg_resid) > 2 * mpg_std) & mpg_mask
        mpg_anom_idx = np.where(mpg_anom_mask)[0]
        print("\n  Fuel‑efficiency anomalies:")
        if mpg_anom_idx.size == 0:
            print("    None detected.")
        else:
            for idx in mpg_anom_idx[:3]:
                actual = mpg[idx]
                pred = mpg_pred[idx]
                diff = actual - pred
                print(f"    Vehicle {idx}: mpg = {actual:.2f}, "
                      f"predicted = {pred:.2f}, diff = {diff:+.2f}")
    else:
        print("  No MPG data available for anomaly detection.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    data = load_csv(csv_path)
    car_17(data)
    car_18(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)

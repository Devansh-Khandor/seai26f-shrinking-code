#!/usr/bin/env python3
"""
skill_09.py

Entry point for CAR-17 and CAR-18 skills.

Usage:
    python skill_09.py <path/to/auto93.csv>
"""

import argparse
import sys
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Import utilities from earlier modules
# --------------------------------------------------------------------------- #
try:
    from skill_02 import read_csv_as_array, compute_summary_stats
except Exception:
    # Minimal fall‑back implementations if skill_02 is missing
    def read_csv_as_array(path: str) -> np.ndarray:
        import csv

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

    def compute_summary_stats(arr: np.ndarray, col_index: int):
        col = arr[:, col_index]
        return float(np.nanmean(col)), float(np.nanmin(col)), float(np.nanmax(col))


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _complete_rows(arr: np.ndarray, exclude_idx: int = None) -> np.ndarray:
    """Return rows with no NaNs, optionally excluding a column."""
    if exclude_idx is None:
        mask = ~np.isnan(arr).any(axis=1)
    else:
        mask = ~np.isnan(arr).any(axis=1) & ~np.isnan(arr[:, exclude_idx])
    return arr[mask]


def _zscore_matrix(arr: np.ndarray, cols: List[int]) -> np.ndarray:
    sub = arr[:, cols]
    mean = np.nanmean(sub, axis=0)
    std = np.nanstd(sub, axis=0, ddof=0)
    std[std == 0] = 1.0
    return (sub - mean) / std


# --------------------------------------------------------------------------- #
# CAR-17: Flag records with unusually different combinations of characteristics
# --------------------------------------------------------------------------- #
def car_17(arr: np.ndarray) -> None:
    print("== CAR-17 ==")

    # Use all numeric columns that are present
    # Ensure no NaNs in any used column
    clean_arr = _complete_rows(arr)
    if clean_arr.size == 0:
        print("No complete records to evaluate.")
        return

    # Compute z‑scores for every feature
    num_cols = clean_arr.shape[1]
    z = _zscore_matrix(clean_arr, list(range(num_cols)))

    # Euclidean distance from the overall mean in z‑space
    distances = np.linalg.norm(z, axis=1)

    # Threshold: 95th percentile of distances
    thresh = np.percentile(distances, 95)

    outlier_indices = np.where(distances > thresh)[0]
    outlier_distances = distances[outlier_indices]

    if outlier_indices.size == 0:
        print("No records identified as unusual outliers.")
        return

    print(f"Total records: {clean_arr.shape[0]}")
    print(f"Outliers detected: {outlier_indices.size}")
    print("Indices and distance scores (top 5 if more than 5):")
    for idx, dist in zip(outlier_indices[:5], outlier_distances[:5]):
        print(f"  Index {idx}: distance = {dist:.3f}")


# --------------------------------------------------------------------------- #
# CAR-18: Flag vehicles with fuel efficiency or weight inconsistent with similar cars
# --------------------------------------------------------------------------- #
def car_18(arr: np.ndarray) -> None:
    print("\n== CAR-18 ==")

    # Prepare training data for mpg
    mpg_mask = ~np.isnan(arr[:, 7])
    mpg_train = arr[mpg_mask]
    X_mpg = np.delete(mpg_train, 7, axis=1)  # all columns except mpg
    y_mpg = mpg_train[:, 7]

    # Prepare training data for weight
    w_mask = ~np.isnan(arr[:, 5])
    w_train = arr[w_mask]
    X_w = np.delete(w_train, 5, axis=1)  # all columns except weight
    y_w = w_train[:, 5]

    # Train linear regressions if enough data
    if X_mpg.shape[0] < 2 or X_w.shape[0] < 2:
        print("Insufficient data to fit regression models.")
        return

    lr_mpg = LinearRegression().fit(X_mpg, y_mpg)
    lr_w = LinearRegression().fit(X_w, y_w)

    # Predict for all rows that have the target present
    pred_mpg = np.full(arr.shape[0], np.nan)
    pred_w = np.full(arr.shape[0], np.nan)

    # For mpg predictions
    mpg_available = ~np.isnan(arr[:, 7])
    X_all_mpg = np.delete(arr[mpg_available], 7, axis=1)
    pred_mpg[mpg_available] = lr_mpg.predict(X_all_mpg)

    # For weight predictions
    w_available = ~np.isnan(arr[:, 5])
    X_all_w = np.delete(arr[w_available], 5, axis=1)
    pred_w[w_available] = lr_w.predict(X_all_w)

    # Compute residuals
    resid_mpg = arr[:, 7] - pred_mpg
    resid_w = arr[:, 5] - pred_w

    # Standard deviations of residuals
    std_mpg = np.nanstd(resid_mpg)
    std_w = np.nanstd(resid_w)

    # Thresholds: 2 * std
    thresh_mpg = 2 * std_mpg
    thresh_w = 2 * std_w

    # Flag indices
    mpg_flag = np.where(np.abs(resid_mpg) > thresh_mpg)[0]
    w_flag = np.where(np.abs(resid_w) > thresh_w)[0]

    if mpg_flag.size == 0 and w_flag.size == 0:
        print("No vehicles flagged as having anomalous mpg or weight.")
        return

    print("Vehicles with anomalous fuel efficiency or weight:")
    for idx in np.unique(np.concatenate((mpg_flag, w_flag))):
        w_val = arr[idx, 5]
        mpg_val = arr[idx, 7]
        print(f"  Index {idx}: weight={w_val:.1f} lbs, mpg={mpg_val:.1f} mpg")
        if idx in mpg_flag:
            print(f"    • mpg residual {resid_mpg[idx]:.2f} exceeds ±{thresh_mpg:.2f} (±{std_mpg:.2f} SD)")
        if idx in w_flag:
            print(f"    • weight residual {resid_w[idx]:.2f} exceeds ±{thresh_w:.2f} (±{std_w:.2f} SD)")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-17 & CAR-18 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if data.size == 0:
        print("CSV contains no data.", file=sys.stderr)
        sys.exit(1)

    car_17(data)
    car_18(data)


if __name__ == "__main__":
    main()

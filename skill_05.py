#!/usr/bin/env python3
"""
Skill 05 – Diagnostic and Comparative Analysis

Usage:
    python skill_05.py <path-to-auto93.csv>
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

# Reuse helpers from previous skills
from skill_01 import load_csv, compute_summary, print_summary, compute_scores


def analyze_car_09(data: np.ndarray, header: list[str]) -> None:
    """
    CAR-09: Identify which characteristics most strongly associate with
    an unusually poor fuel efficiency for the vehicle with the lowest MPG.
    """
    print("\n== CAR-09 ==")

    mpg_idx = header.index("Mpg+")
    # Identify vehicle with lowest actual MPG
    lowest_mpg_idx = np.nanargmin(data[:, mpg_idx])
    lowest_mpg = data[lowest_mpg_idx, mpg_idx]

    # Prepare training data for linear regression (exclude MPG)
    feature_indices = [i for i in range(len(header)) if i != mpg_idx]
    X = data[:, feature_indices]
    y = data[:, mpg_idx]

    # Drop rows with missing values in either X or y
    mask = ~np.isnan(y)
    for col in range(X.shape[1]):
        mask = mask & ~np.isnan(X[:, col])
    X_train = X[mask]
    y_train = y[mask]

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Feature means for contribution calculation
    feature_means = np.mean(X_train, axis=0)

    # Contribution of each feature for the lowest-MPG vehicle
    vehicle_features = data[lowest_mpg_idx, feature_indices]
    # Handle any NaNs in this vehicle
    valid = ~np.isnan(vehicle_features)
    contrib = np.zeros_like(vehicle_features)
    contrib[valid] = (vehicle_features[valid] - feature_means[valid]) * model.coef_[valid]

    # Sort by contribution (most negative first)
    sorted_idx = np.argsort(contrib)  # ascending, so most negative first
    feature_names = [header[i] for i in feature_indices]

    # Show top 3 contributing features
    print(f"Vehicle #{lowest_mpg_idx + 1} has the lowest MPG: {lowest_mpg:.2f}")
    print("Most negative contributors to low MPG:")
    for rank, idx in enumerate(sorted_idx[:3], 1):
        name = feature_names[idx]
        val = vehicle_features[idx]
        coef = model.coef_[idx]
        cont = contrib[idx]
        print(f"  {rank}. {name}: value={val:.2f}, coef={coef:.3f}, "
              f"contrib={cont:.3f}")


def analyze_car_10(data: np.ndarray, header: list[str]) -> None:
    """
    CAR-10: Find common characteristics of vehicles with unfavorable
    overall performance and how they differ from better-performing vehicles.
    """
    print("\n== CAR-10 ==")

    scores = compute_scores(data, header)

    # Define poor and good groups as bottom/top 20% by score
    low_thresh = np.quantile(scores, 0.20)
    high_thresh = np.quantile(scores, 0.80)

    poor_mask = scores <= low_thresh
    good_mask = scores >= high_thresh

    if poor_mask.sum() == 0 or good_mask.sum() == 0:
        print("Insufficient data to compare poor vs good performance groups.")
        return

    # Compute mean of each numeric feature for both groups
    feature_indices = [i for i in range(len(header))]
    feature_names = header

    # Ensure we only consider numeric columns
    # (All columns are numeric in this dataset, but we keep the guard)
    means_poor = np.nanmean(data[poor_mask][:, feature_indices], axis=0)
    means_good = np.nanmean(data[good_mask][:, feature_indices], axis=0)

    # Compute absolute differences
    diff = np.abs(means_poor - means_good)

    # Sort features by magnitude of difference
    sorted_idx = np.argsort(-diff)  # descending

    # Show top 3 distinguishing characteristics
    print(f"Poor-performing vehicles: count={poor_mask.sum()}")
    print(f"Good-performing vehicles: count={good_mask.sum()}")
    print("\nTop 3 distinguishing characteristics:")

    for rank, idx in enumerate(sorted_idx[:3], 1):
        name = feature_names[idx]
        val_poor = means_poor[idx]
        val_good = means_good[idx]
        diff_val = val_poor - val_good
        direction = "higher" if diff_val > 0 else "lower"
        print(f"  {rank}. {name}: poor={val_poor:.2f}, good={val_good:.2f}, "
              f"Δ={diff_val:.2f} ({direction} in poor)")

    # Optional: show summary statistics for each group (optional for completeness)
    print("\nSummary of key columns for each group:")
    for col in ("Lbs-", "Mpg+", "Acc+"):
        idx = header.index(col)
        print(f"  {col}: poor mean={means_poor[idx]:.2f}, good mean={means_good[idx]:.2f}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    analyze_car_09(data, header)
    analyze_car_10(data, header)


if __name__ == "__main__":
    main()

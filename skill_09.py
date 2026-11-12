#!/usr/bin/env python3
"""
Skill 09 – Anomaly Detection and Consistency Checks

Usage:
    python skill_09.py <path-to-auto93.csv>
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

# Reuse helpers from earlier skills
from skill_01 import load_csv, compute_summary, print_summary


def flag_anomalies(data: np.ndarray, header: list[str], z_threshold: float = 3.0) -> list[int]:
    """
    Flag rows whose multivariate z‑score exceeds the specified threshold.
    The score is the Euclidean norm of per‑feature z‑scores.
    """
    # Compute per‑feature statistics
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0, ddof=1)

    # Avoid division by zero for constant columns
    stds_safe = np.where(stds == 0, 1.0, stds)

    # Compute z‑scores (NaNs stay NaN)
    z = (data - means) / stds_safe
    # Euclidean norm per row, ignoring NaNs
    norms = np.linalg.norm(np.nan_to_num(z), axis=1)

    # Flag where norm > threshold
    flagged = np.where(norms > z_threshold)[0]
    return flagged.tolist()


def predict_and_flag(
    data: np.ndarray,
    header: list[str],
    target_col: str,
    threshold_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a linear regression model to predict `target_col` from all other columns.
    Return indices of rows with residual magnitude > threshold_factor * residual_std.
    Also return the residuals themselves for further inspection.
    """
    target_idx = header.index(target_col)
    feature_indices = [i for i in range(len(header)) if i != target_idx]

    X = data[:, feature_indices]
    y = data[:, target_idx]

    # Remove rows with missing values in X or y
    mask = ~np.isnan(y)
    for col in range(X.shape[1]):
        mask = mask & ~np.isnan(X[:, col])
    X_clean = X[mask]
    y_clean = y[mask]

    # Fit model
    model = LinearRegression()
    model.fit(X_clean, y_clean)

    # Predict on full data (NaNs in X will become NaN predictions)
    y_pred = model.predict(X)

    # Compute residuals
    residuals = y - y_pred
    resid_std = np.nanstd(residuals)

    # Flag where |residual| > threshold_factor * resid_std
    threshold = threshold_factor * resid_std
    flagged = np.where(np.abs(residuals) > threshold)[0]

    return flagged, residuals, y_pred, resid_std


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-17: Flag anomalous vehicle records based on multivariate distance
    # ------------------------------------------------------------------
    print("\n== CAR-17 ==")

    anomaly_idx = flag_anomalies(data, header, z_threshold=3.0)

    if not anomaly_idx:
        print("No anomalies found (all records within 3σ).")
    else:
        # Show up to three flagged records
        print(f"Found {len(anomaly_idx)} anomalous records (threshold z > 3.0). Showing first 3:")
        for i, idx in enumerate(anomaly_idx[:3], 1):
            # Compute per‑feature z‑scores for display
            means = np.nanmean(data, axis=0)
            stds = np.nanstd(data, axis=0, ddof=1)
            z = (data[idx] - means) / np.where(stds == 0, 1.0, stds)
            # Show the three features with largest |z|
            abs_z = np.abs(z)
            top_features = np.argsort(abs_z)[-3:][::-1]
            feature_str = ", ".join(
                f"{header[j]}={data[idx, j]:.1f} ({z[j]:.2f})"
                for j in top_features
            )
            print(f"{i}. Vehicle #{idx + 1}: {feature_str}")

    # ------------------------------------------------------------------
    # CAR-18: Flag inconsistencies in reported MPG or Weight
    # ------------------------------------------------------------------
    print("\n== CAR-18 ==")

    # MPG inconsistencies
    mpg_flagged, mpg_residuals, mpg_pred, mpg_std = predict_and_flag(
        data, header, "Mpg+", threshold_factor=2.0
    )
    if not mpg_flagged.size:
        print("No significant MPG inconsistencies detected.")
    else:
        print(f"Found {len(mpg_flagged)} vehicles with MPG residuals > 2σ (σ={mpg_std:.2f}). Showing first 3:")
        for i, idx in enumerate(mpg_flagged[:3], 1):
            actual = data[idx, header.index("Mpg+")]
            predicted = mpg_pred[idx]
            resid = mpg_residuals[idx]
            print(f"{i}. Vehicle #{idx + 1}: Actual={actual:.2f}, Pred={predicted:.2f}, Resid={resid:.2f}")

    # Weight inconsistencies
    weight_flagged, weight_residuals, weight_pred, weight_std = predict_and_flag(
        data, header, "Lbs-", threshold_factor=2.0
    )
    if not weight_flagged.size:
        print("No significant Weight inconsistencies detected.")
    else:
        print(f"Found {len(weight_flagged)} vehicles with Weight residuals > 2σ (σ={weight_std:.2f}). Showing first 3:")
        for i, idx in enumerate(weight_flagged[:3], 1):
            actual = data[idx, header.index("Lbs-")]
            predicted = weight_pred[idx]
            resid = weight_residuals[idx]
            print(f"{i}. Vehicle #{idx + 1}: Actual={actual:.2f}, Pred={predicted:.2f}, Resid={resid:.2f}")


if __name__ == "__main__":
    main()

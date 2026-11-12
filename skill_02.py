#!/usr/bin/env python3
"""
Skill 02 – Fuel Efficiency and Weight Estimation

Usage:
    python skill_02.py <path-to-auto93.csv>
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from skill_01 import load_csv, compute_summary, print_summary  # reuse earlier code


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-03: Estimate missing fuel‑efficiency (MPG) values
    # ------------------------------------------------------------------
    print("\n== CAR-03 ==")

    mpg_idx = header.index("Mpg+")
    feature_indices = [i for i in range(len(header)) if i != mpg_idx]

    X = data[:, feature_indices]
    y = data[:, mpg_idx]

    # Use only rows where MPG is not missing
    valid = ~np.isnan(y)
    X_train = X[valid]
    y_train = y[valid]

    # If we have no complete MPG entries, we still build a model on all data
    if X_train.size == 0:
        X_train = X
        y_train = y

    mpg_model = LinearRegression()
    mpg_model.fit(X_train, y_train)

    # Predict for all vehicles
    mpg_pred = mpg_model.predict(X)

    # Estimate uncertainty as residual std. dev. on training data
    y_train_pred = mpg_model.predict(X_train)
    residuals = y_train - y_train_pred
    mpg_sigma = np.sqrt(np.mean(residuals ** 2))

    # Show results for first three vehicles (or fewer if dataset is small)
    n_show = min(3, len(data))
    for idx in range(n_show):
        pred = mpg_pred[idx]
        print(f"Vehicle #{idx + 1}: Predicted MPG = {pred:.2f} ± {mpg_sigma:.2f}")

    # ------------------------------------------------------------------
    # CAR-04: Estimate vehicle weight
    # ------------------------------------------------------------------
    print("\n== CAR-04 ==")

    weight_idx = header.index("Lbs-")
    feature_indices_w = [i for i in range(len(header)) if i != weight_idx]

    X_w = data[:, feature_indices_w]
    y_w = data[:, weight_idx]

    # Assume no missing weights
    weight_model = LinearRegression()
    weight_model.fit(X_w, y_w)

    weight_pred = weight_model.predict(X_w)
    y_w_pred = weight_model.predict(X_w)
    weight_sigma = np.sqrt(np.mean((y_w - y_w_pred) ** 2))

    n_show = min(3, len(data))
    for idx in range(n_show):
        pred = weight_pred[idx]
        print(f"Vehicle #{idx + 1}: Predicted Weight = {pred:.2f} ± {weight_sigma:.2f} lbs")


if __name__ == "__main__":
    main()

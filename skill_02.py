#!/usr/bin/env python3
"""
Skill 02: Fuel Efficiency & Weight Estimation (CAR-03 & CAR-04)

Usage:
    python skill_02.py <path_to_auto93.csv>
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression


def load_data(csv_path: str):
    """
    Load CSV into structured NumPy array and cast numeric fields.
    """
    data = np.genfromtxt(
        csv_path,
        delimiter=',',
        names=True,
        dtype=None,
        encoding='utf-8'
    )

    # Cast numeric fields to float
    weight = data['Lbs-'].astype(float)
    mpg = data['Mpg+'].astype(float)
    acc = data['Acc+'].astype(float)
    hp = data['HpX'].astype(float)
    volume = data['Volume'].astype(float)
    model_year = data['Model'].astype(float)
    origin = data['origin'].astype(float)

    return {
        'weight': weight,
        'mpg': mpg,
        'acc': acc,
        'hp': hp,
        'volume': volume,
        'model_year': model_year,
        'origin': origin
    }


def main(csv_path: str):
    # Load data
    data = load_data(csv_path)
    weight = data['weight']
    mpg = data['mpg']
    acc = data['acc']
    hp = data['hp']
    volume = data['volume']
    model_year = data['model_year']
    origin = data['origin']

    n_samples = len(weight)

    # Build feature matrix for mpg estimation (all features)
    X_mpg = np.column_stack([acc, mpg, hp, volume, model_year, origin])

    # Identify rows with missing MPG
    missing_mpg_mask = np.isnan(mpg)

    # Train model on rows with known MPG
    X_train_mpg = X_mpg[~missing_mpg_mask]
    y_train_mpg = mpg[~missing_mpg_mask]
    mpg_model = LinearRegression()
    mpg_model.fit(X_train_mpg, y_train_mpg)

    # Predict MPG for all vehicles (including missing)
    mpg_pred = mpg_model.predict(X_mpg)

    # === CAR-03 ===
    print("== CAR-03 ==")
    print("Estimated MPG for vehicles with missing data (first 3):")
    count = 0
    for idx in range(n_samples):
        if missing_mpg_mask[idx]:
            print(f"Vehicle {idx}: predicted mpg = {mpg_pred[idx]:.2f}")
            count += 1
            if count >= 3:
                break
    if count == 0:
        # If no missing MPG, show first 3 predictions anyway
        for idx in range(min(3, n_samples)):
            print(f"Vehicle {idx}: predicted mpg = {mpg_pred[idx]:.2f}")

    # Replace missing MPG with predicted values for weight model
    mpg_filled = mpg.copy()
    mpg_filled[missing_mpg_mask] = mpg_pred[missing_mpg_mask]

    # Build feature matrix for weight estimation (use same features but with filled MPG)
    X_weight = np.column_stack([acc, mpg_filled, hp, volume, model_year, origin])

    # Train weight model on all data
    weight_model = LinearRegression()
    weight_model.fit(X_weight, weight)

    # Predict weight for all vehicles
    weight_pred = weight_model.predict(X_weight)

    # Compute residual standard deviation (uncertainty)
    residuals = weight - weight_model.predict(X_weight)
    sigma = residuals.std(ddof=1)

    # === CAR-04 ===
    print("\n== CAR-04 ==")
    print(f"Estimated weight and uncertainty (sigma = {sigma:.2f} lbs) for first 3 vehicles:")
    for idx in range(min(3, n_samples)):
        print(f"Vehicle {idx}: predicted weight = {weight_pred[idx]:.2f} ± {sigma:.2f} lbs")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path_to_auto93.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

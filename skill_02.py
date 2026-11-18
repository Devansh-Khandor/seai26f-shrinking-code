#!/usr/bin/env python3
"""
Skill 02 – Missing values prediction and uncertainty.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import json
import sys
from math import isfinite
from typing import Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# Helpers – reuse from skill_01
# --------------------------------------------------------------------------- #
def parse_float(s: str):
    """Parse a string into float, return None for '?' or empty."""
    s = s.strip()
    if s == '' or s == '?':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Return mean, median, min, max for a list of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    # median
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return {
        "mean": mean,
        "median": median,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


# --------------------------------------------------------------------------- #
# Prediction utilities
# --------------------------------------------------------------------------- #
def train_regression(
    data: List[Dict[str, float]],
    target_col: str,
    feature_cols: List[str],
) -> Tuple[LinearRegression, SimpleImputer, np.ndarray, np.ndarray]:
    """Train a LinearRegression model on the given data."""
    X_raw = np.array(
        [
            [row.get(col) if row.get(col) is not None else np.nan for col in feature_cols]
            for row in data
        ]
    )
    y_raw = np.array([row.get(target_col) for row in data], dtype=float)

    # mask: keep rows where target is not missing and all features present
    mask = ~np.isnan(y_raw) & (~np.isnan(X_raw).any(axis=1))
    X = X_raw[mask]
    y = y_raw[mask]

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    model = LinearRegression()
    model.fit(X_imp, y)

    return model, imputer, X_imp, y


def predict(
    model: LinearRegression,
    imputer: SimpleImputer,
    data: List[Dict[str, float]],
    feature_cols: List[str],
) -> np.ndarray:
    X_raw = np.array(
        [
            [row.get(col) if row.get(col) is not None else np.nan for col in feature_cols]
            for row in data
        ]
    )
    X_imp = imputer.transform(X_raw)
    return model.predict(X_imp)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Read data
    data: List[Dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    # Identify held‑out rows (divisible by 5, 1‑based)
    held_out_indices = [i for i, _ in enumerate(data, start=1) if i % 5 == 0]
    held_out_rows = [data[i - 1] for i in held_out_indices]

    # ------------------------------------------------------------------ #
    # CAR-03: Predict missing Mpg+
    # ------------------------------------------------------------------ #
    mpg_features = ["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs-", "Acc+"]
    model_mpg, imputer_mpg, X_train_mpg, y_train_mpg = train_regression(
        data, "Mpg+", mpg_features
    )
    # Predict on held‑out
    preds_mpg = predict(model_mpg, imputer_mpg, held_out_rows, mpg_features)

    # Compute MAE on held‑out actual values
    actual_mpg = np.array([row["Mpg+"] for row in held_out_rows], dtype=float)
    mae_mpg = float(np.mean(np.abs(preds_mpg - actual_mpg)))
    mae_mpg = round(mae_mpg, 6)

    # Build JSON result
    predictions_mpg = {
        str(row_num): round(float(preds_mpg[idx]), 6) for idx, row_num in enumerate(held_out_indices)
    }
    ans_car_03 = json.dumps(
        {"predictions": predictions_mpg, "mae": mae_mpg},
        separators=(",", ":"),
    )

    # ------------------------------------------------------------------ #
    # CAR-04: Predict Lbs- with uncertainty
    # ------------------------------------------------------------------ #
    lbs_features = ["Clndrs", "Volume", "HpX", "Model", "origin", "Acc+", "Mpg+"]
    model_lbs, imputer_lbs, X_train_lbs, y_train_lbs = train_regression(
        data, "Lbs-", lbs_features
    )
    # Predict on held‑out
    preds_lbs = predict(model_lbs, imputer_lbs, held_out_rows, lbs_features)

    # Residual std for 95% CI
    residuals = y_train_lbs - model_lbs.predict(X_train_lbs)
    residual_std = float(np.std(residuals, ddof=1))
    ci_scale = 1.96 * residual_std

    # Build predictions dict with uncertainty
    predictions_lbs = {}
    for idx, row_num in enumerate(held_out_indices):
        est = round(float(preds_lbs[idx]), 6)
        low = round(est - ci_scale, 6)
        high = round(est + ci_scale, 6)
        predictions_lbs[str(row_num)] = {"estimate": est, "low": low, "high": high}

    mae_lbs = float(np.mean(np.abs(preds_lbs - np.array([row["Lbs-"] for row in held_out_rows], dtype=float))))
    mae_lbs = round(mae_lbs, 6)

    ans_car_04 = json.dumps(
        {"predictions": predictions_lbs, "mae": mae_lbs},
        separators=(",", ":"),
    )

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    print("== CAR-03 ==")
    print(f"Number of held‑out rows: {len(held_out_indices)}")
    print(f"MAE on held‑out Mpg+: {mae_mpg}")
    print("Predicted Mpg+ for each held‑out row:")
    for row_num in held_out_indices:
        print(f"{row_num}: {predictions_mpg[str(row_num)]}")

    print(f"ANSWER CAR-03 {ans_car_03}")

    print("\n== CAR-04 ==")
    print(f"Number of held‑out rows: {len(held_out_indices)}")
    print(f"MAE on held‑out Lbs-: {mae_lbs}")
    print("Predicted Lbs- with uncertainty for each held‑out row:")
    for row_num in held_out_indices:
        pred = predictions_lbs[str(row_num)]
        print(f"{row_num}: estimate={pred['estimate']}, low={pred['low']}, high={pred['high']}")

    print(f"ANSWER CAR-04 {ans_car_04}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

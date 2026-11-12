#!/usr/bin/env python3
"""
skill_10.py

Entry point for CAR-19 and CAR-20 skills.

Usage:
    python skill_10.py <path/to/auto93.csv>
"""

import argparse
import sys
import random
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Reuse utilities that handle NaNs gracefully
# --------------------------------------------------------------------------- #
try:
    from skill_02 import read_csv_as_array, compute_composite_score, compute_summary_stats
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

    def compute_composite_score(arr: np.ndarray) -> Tuple[np.ndarray, list, list]:
        weight = arr[:, 5].reshape(-1, 1)
        fuel = arr[:, 7].reshape(-1, 1)
        acc = arr[:, 6].reshape(-1, 1)

        from sklearn.preprocessing import MinMaxScaler
        scaler_w = MinMaxScaler()
        scaler_f = MinMaxScaler()
        scaler_a = MinMaxScaler()

        norm_weight = 1 - scaler_w.fit_transform(weight)
        norm_fuel = scaler_f.fit_transform(fuel)
        norm_acc = 1 - scaler_a.fit_transform(acc)

        scores = (norm_weight + norm_fuel + norm_acc).flatten()
        top_indices = np.argsort(scores)[::-1][:5]
        top_scores = scores[top_indices]
        return scores, list(top_indices), list(top_scores)

    def compute_summary_stats(arr: np.ndarray, col_index: int) -> Tuple[float, float, float]:
        col = arr[:, col_index]
        return float(np.nanmean(col)), float(np.nanmin(col)), float(np.nanmax(col))


# --------------------------------------------------------------------------- #
# CAR-19: Compare our system against random selection
# --------------------------------------------------------------------------- #
def car_19(arr: np.ndarray, budget: int = 5, trials: int = 1000) -> None:
    """
    Demonstrates that selecting the top-<budget> vehicles by composite score
    yields a higher average score than selecting <budget> vehicles at random.
    """
    print("== CAR-19 ==")

    # Compute composite scores for all vehicles
    scores, _, _ = compute_composite_score(arr)

    # Our system: top-k vehicles
    top_k_idx = np.argsort(scores)[::-1][:budget]
    top_k_mean = float(np.mean(scores[top_k_idx]))

    # Random baseline: average over many random draws
    rand_means = []
    n = len(scores)
    if n < budget:
        print("Not enough vehicles to draw from.")
        return
    for _ in range(trials):
        rand_idx = random.sample(range(n), budget)
        rand_means.append(np.mean(scores[rand_idx]))
    rand_mean_avg = float(np.mean(rand_means))

    # Results
    print(f"Budget: {budget} vehicles")
    print(f"Average composite score of top {budget} (our system): {top_k_mean:.4f}")
    print(f"Average composite score of random {budget} (average over {trials} trials): {rand_mean_avg:.4f}")

    improvement = (top_k_mean - rand_mean_avg) / rand_mean_avg * 100 if rand_mean_avg != 0 else 0.0
    print(f"Improvement over random selection: {improvement:.2f}%")

# --------------------------------------------------------------------------- #
# CAR-20: Evaluate predictive performance vs simple historical‑average baseline
# --------------------------------------------------------------------------- #
def car_20(arr: np.ndarray) -> None:
    """
    Uses linear regression to predict the composite score of each vehicle based
    on all other features, and compares the mean absolute error against a
    baseline that always predicts the historical average composite score.
    """
    print("\n== CAR-20 ==")

    # Filter rows that contain no NaNs in any of the 8 columns
    complete_mask = ~np.isnan(arr).any(axis=1)
    data = arr[complete_mask]
    if data.size == 0:
        print("No complete rows available for predictive analysis.")
        return

    # Compute true composite scores for complete rows
    true_scores, _, _ = compute_composite_score(data)

    # Features: all columns except the composite score (which is not part of data)
    X = data[:, :8]  # 8 columns

    n_samples = X.shape[0]
    if n_samples < 2:
        print("Insufficient data for training predictive model.")
        return

    # Leave‑one‑out cross‑validation
    mae_loocv = 0.0
    for i in range(n_samples):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(true_scores, i)
        X_test = X[i].reshape(1, -1)
        y_true = true_scores[i]

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)[0]
        mae_loocv += abs(y_pred - y_true)

    mae_loocv /= n_samples

    # Baseline: predict mean composite score for all
    mean_score = float(np.mean(true_scores))
    mae_baseline = float(np.mean(np.abs(true_scores - mean_score)))

    # Results
    print(f"Mean Absolute Error (LOOCV linear regression): {mae_loocv:.4f}")
    print(f"Mean Absolute Error (historical‑average baseline): {mae_baseline:.4f}")

    improvement = (mae_baseline - mae_loocv) / mae_baseline * 100 if mae_baseline != 0 else 0.0
    print(f"Reduction in prediction error relative to baseline: {improvement:.2f}%")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-19 & CAR-20 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    # Load data
    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if data.size == 0:
        print("CSV contains no data.", file=sys.stderr)
        sys.exit(1)

    # Execute CAR-19
    car_19(data)

    # Execute CAR-20
    car_20(data)


if __name__ == "__main__":
    main()

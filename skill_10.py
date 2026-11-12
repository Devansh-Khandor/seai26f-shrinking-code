#!/usr/bin/env python3
"""
Skill 10 – System Evaluation & Performance Benchmarking

This module implements the final two CARs:
    CAR-19: Compare the vehicle-selection system against random choice
    CAR-20: Compare model‑based performance estimates with a simple historical average

Both analyses are performed on the dataset supplied via the command line.
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

# Reuse helpers from the earlier skills
from skill_01 import load_csv, compute_scores


def compare_selection_system(data: np.ndarray, header: list[str], budget: int = 5,
                            trials: int = 10, seed: int = 42) -> None:
    """
    CAR-19: Evaluate whether the system's top‑budget selections outperform
    random selections when both have the same evaluation budget.
    """
    print("\n== CAR-19 ==")

    n = min(budget, data.shape[0])
    if n <= 0:
        print("No vehicles available for evaluation.")
        return

    # System: top‑n scores
    scores = compute_scores(data, header)
    top_indices = np.argsort(scores)[::-1][:n]
    system_avg = np.mean(scores[top_indices])

    # Random: average over several trials
    rng = np.random.default_rng(seed)
    random_avgs = []
    for _ in range(trials):
        rand_idx = rng.choice(data.shape[0], size=n, replace=False)
        random_avgs.append(np.mean(scores[rand_idx]))
    random_avg = np.mean(random_avgs)

    # Simple statistics
    diff = system_avg - random_avg
    improvement = diff / random_avg * 100 if random_avg != 0 else float('inf')

    print(f"Evaluation budget: {n} vehicles")
    print(f"Average score – System: {system_avg:.4f}")
    print(f"Average score – Random:  {random_avg:.4f}")
    print(f"Difference:       {diff:.4f}")
    print(f"Improvement:      {improvement:.2f}% (system > random)")
    print(f"Random trials: {trials} (seed={seed})")


def compare_prediction_improvement(data: np.ndarray, header: list[str],
                                   target_col: str = "Mpg+", seed: int = 123) -> None:
    """
    CAR-20: Compare model‑based MPG predictions with a simple historical average.
    The comparison is performed on a hold‑out test set.
    """
    print("\n== CAR-20 ==")

    if data.shape[0] < 2:
        print("Not enough data to split into training and testing sets.")
        return

    # Identify columns
    target_idx = header.index(target_col)
    feature_indices = [i for i in range(len(header)) if i != target_idx]

    X = data[:, feature_indices]
    y = data[:, target_idx]

    # Split indices
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.shape[0])
    split = int(0.8 * data.shape[0])
    train_idx, test_idx = perm[:split], perm[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit linear regression model on training data
    # Skip rows with missing values
    mask_train = ~np.isnan(y_train)
    for col in range(X_train.shape[1]):
        mask_train = mask_train & ~np.isnan(X_train[:, col])
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]

    model = LinearRegression()
    model.fit(X_train_clean, y_train_clean)

    # Predict on test set
    y_pred = model.predict(X_test)
    # Handle rows with NaNs in X_test
    nan_mask = np.isnan(y_pred)
    y_pred = np.where(nan_mask, np.nan, y_pred)

    # Compute RMSE for model
    resid_model = y_test - y_pred
    rmse_model = np.sqrt(np.nanmean(resid_model ** 2))

    # Baseline: historical average from training set
    mean_train = np.nanmean(y_train)
    resid_baseline = y_test - mean_train
    rmse_baseline = np.sqrt(np.mean(resid_baseline ** 2))

    improvement = (rmse_baseline - rmse_model) / rmse_baseline * 100 if rmse_baseline != 0 else float('inf')

    print(f"Training set: {len(train_idx)} records, Test set: {len(test_idx)} records")
    print(f"RMSE – Linear regression model: {rmse_model:.4f}")
    print(f"RMSE – Historical‑average baseline: {rmse_baseline:.4f}")
    print(f"Improvement over baseline: {improvement:.2f}%")
    print(f"Target column: {target_col}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    compare_selection_system(data, header, budget=5, trials=10)
    compare_prediction_improvement(data, header, target_col="Mpg+")


if __name__ == "__main__":
    main()

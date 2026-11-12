#!/usr/bin/env python3
"""
Skill 10: Performance Evaluation & Random Baseline Comparison (CAR-19 & CAR-20)

Usage:
    python skill_10.py <path_to_auto93.csv>
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
# Composite score calculation
# --------------------------------------------------------------------------- #

def composite_score(data: np.ndarray) -> np.ndarray:
    """Return an array of composite scores for each vehicle."""
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    def norm_minmax(arr: np.ndarray) -> np.ndarray:
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        return (arr - mn) / (mx - mn)

    weight_score = 1.0 - norm_minmax(weight)
    mpg_score    = norm_minmax(mpg)
    acc_score    = 1.0 - norm_minmax(acc)

    return weight_score + mpg_score + acc_score


# --------------------------------------------------------------------------- #
# CAR-19: System vs Random Baseline
# --------------------------------------------------------------------------- #

def car_19(data: np.ndarray):
    print("== CAR-19 ==")

    mpg = data['Mpg+'].astype(float)
    mask = ~np.isnan(mpg)
    if not mask.any():
        print("No MPG data available for evaluation.")
        return

    data_clean = data[mask]
    comp = composite_score(data_clean)

    N = min(5, len(data_clean))
    top_idx = np.argsort(-comp)[:N]
    top_mpg_mean = mpg[mask][top_idx].mean()

    rng = np.random.default_rng(0)
    random_means = []
    for _ in range(1000):
        rand_idx = rng.choice(len(data_clean), size=N, replace=False)
        random_means.append(mpg[mask][rand_idx].mean())
    rand_mean = np.mean(random_means)
    rand_std = np.std(random_means, ddof=1)

    print(f"Top {N} vehicles by composite score have average MPG = {top_mpg_mean:.2f}")
    print(f"Randomly selecting {N} vehicles yields average MPG = {rand_mean:.2f} (±{rand_std:.2f})")
    improvement = top_mpg_mean - rand_mean
    print(f"Improvement over random selection: {improvement:.2f} mpg\n")


# --------------------------------------------------------------------------- #
# CAR-20: Predictive vs Historical‑Average Estimates
# --------------------------------------------------------------------------- #

def car_20(data: np.ndarray):
    print("== CAR-20 ==")

    mpg = data['Mpg+'].astype(float)
    # Define feature columns (excluding mpg itself)
    feat_cols = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                 'Lbs-', 'Acc+']

    # Build a mask where all features and mpg are present
    mask = ~np.isnan(mpg)
    for col in feat_cols:
        mask &= ~np.isnan(data[col])

    if mask.sum() < 2:
        print("Not enough complete records for model training.")
        return

    # Prepare data
    X = np.column_stack([data[col][mask].astype(float) for col in feat_cols])
    y = mpg[mask]

    # Random split: 70% train, 30% test
    rng = np.random.default_rng(0)
    n = X.shape[0]
    idx = rng.permutation(n)
    train_size = int(0.7 * n)
    train_idx = idx[:train_size]
    test_idx  = idx[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    mae_model = np.mean(np.abs(pred_test - y_test))

    # Historical average estimate
    avg_train = y_train.mean()
    mae_avg   = np.mean(np.abs(y_test - avg_train))

    print(f"Linear regression MAE on test set: {mae_model:.2f} mpg")
    print(f"Historical average MAE on test set: {mae_avg:.2f} mpg")
    improvement = mae_avg - mae_model
    print(f"Improvement over historical average: {improvement:.2f} mpg\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    data = load_csv(csv_path)
    car_19(data)
    car_20(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)

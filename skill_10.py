#!/usr/bin/env python3
"""
Skill 10: Evaluation comparison and performance estimate validation.

This script implements:
  CAR-19 – Compare the vehicle‑selection system against random selection
           when both use the same evaluation budget.
  CAR-20 – Compare the regression‑based MPG predictions with a
           historical‑average baseline, using cross‑validation to show
           improvement on unseen records.
"""

import sys
import random
import math
import numpy as np
from collections import defaultdict

# Re‑use utilities from earlier skills
from skill_02 import load_vehicles, Vehicle
from skill_01 import best_combination, compute_z_scores

# --------------------------------------------------------------------------- #
# Helper: compute composite scores for all vehicles
# --------------------------------------------------------------------------- #
def all_composite_scores(vehicles):
    """
    Return a dict mapping vehicle index to its composite score.
    """
    scores = {}
    # best_combination with top_n=len(vehicles) returns all sorted
    for score, idx, v in best_combination(vehicles, top_n=len(vehicles)):
        scores[idx] = score
    return scores

# --------------------------------------------------------------------------- #
# CAR-19: system vs random selection
# --------------------------------------------------------------------------- #
def car_19(vehicles, budget=5, n_trials=1000):
    """
    Compare the mean composite score of the best 'budget' vehicles
    with the mean composite score of randomly selected vehicles
    using the same budget.  Returns a list of output lines.
    """
    lines = []

    # Compute all scores once
    scores = all_composite_scores(vehicles)

    # Get top budget vehicles
    top_indices = sorted(scores, key=scores.get, reverse=True)[:budget]
    best_avg = np.mean([scores[i] for i in top_indices])

    # Random sampling
    random_means = []
    for _ in range(n_trials):
        sample = random.sample(range(len(vehicles)), budget)
        random_means.append(np.mean([scores[i] for i in sample]))

    rand_avg = np.mean(random_means)
    rand_std = np.std(random_means, ddof=1)

    improvement = best_avg - rand_avg
    improvement_pct = (improvement / rand_avg * 100) if rand_avg != 0 else float('inf')

    lines.append(f"Evaluation budget: {budget} vehicles")
    lines.append(f"Best‑N average composite score: {best_avg:.4f}")
    lines.append(f"Random sample average composite score: {rand_avg:.4f} ± {rand_std:.4f}")
    lines.append(f"Improvement over random: {improvement:.4f} ({improvement_pct:.1f}%)")
    return lines

# --------------------------------------------------------------------------- #
# CAR-20: cross‑validated performance estimate
# --------------------------------------------------------------------------- #
def car_20(vehicles, test_ratio=0.2):
    """
    Compare the regression‑based MPG predictions with a historical‑average
    baseline using cross‑validation.  Returns a list of output lines.
    """
    lines = []

    # Filter only vehicles with known MPG
    known = [v for v in vehicles if v.Mpg is not None]
    if len(known) < 10:
        lines.append("Not enough records with known MPG to perform cross‑validation.")
        return lines

    # Shuffle and split
    indices = list(range(len(known)))
    random.shuffle(indices)
    split_point = int(len(known) * (1 - test_ratio))
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    # Prepare training data
    X_train = np.array([
        [v.Clndrs, v.Volume, v.HpX, v.Model, v.origin, v.Lbs, v.Acc]
        for v in [known[i] for i in train_idx]
    ])
    y_train = np.array([known[i].Mpg for i in train_idx])

    # Training data for test predictions
    X_test = np.array([
        [v.Clndrs, v.Volume, v.HpX, v.Model, v.origin, v.Lbs, v.Acc]
        for v in [known[i] for i in test_idx]
    ])
    y_test = np.array([known[i].Mpg for i in test_idx])

    # Fit regression model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test
    y_pred = model.predict(X_test)

    # MSE for regression
    mse_reg = np.mean((y_test - y_pred) ** 2)

    # Baseline: historical average MPG from training set
    mean_mpg_train = np.mean(y_train)
    y_baseline = np.full_like(y_test, mean_mpg_train)
    mse_base = np.mean((y_test - y_baseline) ** 2)

    improvement = mse_base - mse_reg
    improvement_pct = (improvement / mse_base * 100) if mse_base != 0 else float('inf')

    lines.append(f"Cross‑validation split: {len(train_idx)} train / {len(test_idx)} test")
    lines.append(f"Regression MSE on test set: {mse_reg:.2f}")
    lines.append(f"Historical‑average MSE on test set: {mse_base:.2f}")
    lines.append(f"Improvement (lower MSE): {improvement:.2f} ({improvement_pct:.1f}%)")
    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-19
    print("== CAR-19 ==")
    for line in car_19(vehicles, budget=5, n_trials=1000):
        print(line)

    # CAR-20
    print("\n== CAR-20 ==")
    for line in car_20(vehicles, test_ratio=0.2):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_10.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])

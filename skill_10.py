#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <csv_path>

The script prints:
    * CAR-19: Comparison between the composite‑score selection method and random selection
              under the same evaluation budget.
    * CAR-20: Comparison of regression‑based weight and MPG predictions versus
              simple historical‑average baselines on a held‑out test set.
"""

import sys
import math
import random
from pathlib import Path

# Import helpers from previous skills
from skill_01 import read_csv, compute_composite_scores, format_number, mean
from skill_02 import (
    linear_regression_coefficients,
    regression_statistics,
    predict_using_coefficients,
)

# --------------------------------------------------------------------------- #
# Helper functions for evaluation
# --------------------------------------------------------------------------- #

def mean_mse(preds, actuals):
    """Compute mean squared error between two lists of numbers."""
    if not preds:
        return float('nan')
    return sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(preds)

# --------------------------------------------------------------------------- #
# CAR-19: Composite score vs random selection
# --------------------------------------------------------------------------- #

def print_car_19_selection_comparison(data, budget=5):
    if len(data) < 2:
        print("\n== CAR-19 ==")
        print("  Not enough vehicle records to compare.")
        return

    # Composite scores for all vehicles
    scored = compute_composite_scores(data)
    scored.sort(key=lambda x: x[0])  # lower is better

    # Ensure budget does not exceed data size
    budget = min(budget, len(data))
    selected_scores = [score for score, _ in scored[:budget]]
    selected_mean = mean(selected_scores)

    # Random selection
    random.seed(42)  # reproducible
    random_indices = random.sample(range(len(data)), budget)
    random_scores = [compute_composite_scores([data[i]])[0][0] for i in random_indices]
    random_mean = mean(random_scores)

    print("\n== CAR-19 ==")
    print(f"  Evaluation budget (number of vehicles): {budget}")
    print(f"  Mean composite score of selected vehicles: {format_number(selected_mean)}")
    print(f"  Mean composite score of random vehicles:   {format_number(random_mean)}")
    diff = random_mean - selected_mean
    print(f"  Difference (random - selected): {format_number(diff)}")
    if diff > 0:
        print("  The composite‑score selection yields a lower (better) mean score.")
    else:
        print("  Random selection performed as well or better in this sample.")

# --------------------------------------------------------------------------- #
# CAR-20: Regression vs historical‑average baseline
# --------------------------------------------------------------------------- #

def print_car_20_prediction_improvement(data, train_frac=0.8):
    if len(data) < 5:
        print("\n== CAR-20 ==")
        print("  Not enough data for a meaningful train/test split.")
        return

    # Shuffle data deterministically
    random.seed(123)
    data_shuffled = data[:]
    random.shuffle(data_shuffled)

    split_idx = int(len(data_shuffled) * train_frac)
    train_data = data_shuffled[:split_idx]
    test_data  = data_shuffled[split_idx:]

    # Train regression models
    try:
        mpg_beta = linear_regression_coefficients(train_data, 'Mpg+', ['Lbs-', 'Acc+', 'Volume', 'HpX'])
        weight_beta = linear_regression_coefficients(train_data, 'Lbs-', ['Mpg+', 'Acc+', 'Volume', 'HpX'])
    except ValueError as exc:
        print("\n== CAR-20 ==")
        print(f"  Regression training failed: {exc}")
        return

    # Predict on test set
    mpg_preds = [predict_using_coefficients(mpg_beta, v, ['Lbs-', 'Acc+', 'Volume', 'HpX']) for v in test_data]
    weight_preds = [predict_using_coefficients(weight_beta, v, ['Mpg+', 'Acc+', 'Volume', 'HpX']) for v in test_data]

    # Baseline predictions: historical means from training set
    train_mpg_mean = mean([v['Mpg+'] for v in train_data])
    train_weight_mean = mean([v['Lbs-'] for v in train_data])

    baseline_mpg_preds = [train_mpg_mean] * len(test_data)
    baseline_weight_preds = [train_weight_mean] * len(test_data)

    # Compute MSE
    mpg_mse = mean_mse(mpg_preds, [v['Mpg+'] for v in test_data])
    weight_mse = mean_mse(weight_preds, [v['Lbs-'] for v in test_data])

    baseline_mpg_mse = mean_mse(baseline_mpg_preds, [v['Mpg+'] for v in test_data])
    baseline_weight_mse = mean_mse(baseline_weight_preds, [v['Lbs-'] for v in test_data])

    print("\n== CAR-20 ==")
    print(f"  Test set size: {len(test_data)} vehicles")
    print("  Prediction accuracy (MSE):")
    print(f"    MPG – Regression: {format_number(mpg_mse)} vs Baseline: {format_number(baseline_mpg_mse)}")
    print(f"    Weight – Regression: {format_number(weight_mse)} vs Baseline: {format_number(baseline_weight_mse)}")
    # Show improvement percentage if applicable
    mpg_improve = (baseline_mpg_mse - mpg_mse) / baseline_mpg_mse * 100 if baseline_mpg_mse != 0 else float('nan')
    weight_improve = (baseline_weight_mse - weight_mse) / baseline_weight_mse * 100 if baseline_weight_mse != 0 else float('nan')
    print(f"  MPG MSE improvement: {format_number(mpg_improve)} %")
    print(f"  Weight MSE improvement: {format_number(weight_improve)} %")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    print_car_19_selection_comparison(data, budget=5)
    print_car_20_prediction_improvement(data)

if __name__ == "__main__":
    main()

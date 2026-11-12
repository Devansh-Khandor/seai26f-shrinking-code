#!/usr/bin/env python3
"""
Skill 10: Evaluation of vehicle-selection quality and predictive accuracy.

CAR-19: Show whether the system finds better candidates than random selection
        when both approaches use the same number of evaluated vehicles.

CAR-20: Compare the system's performance estimates (linear regression) against
        a simple historical‑average estimate and report any improvement.

Usage:
    python skill_10.py <path-to-auto93.csv>
"""

import sys
import random
import statistics
from pathlib import Path

# Core utilities from previous skills
from skill_01 import load_data, compute_balanced_scores, format_value

# Regression helpers from skill_02
from skill_02 import _prepare_regression, _fit_regression, _predict


def _average_score(rows):
    """Return the mean score of the given list of scored rows."""
    if not rows:
        return 0.0
    return statistics.mean([r["score"] for r in rows])


def _car_19(rows):
    """
    Compare the average score of the top N vehicles (system) versus
    the average score of N randomly selected vehicles.
    """
    print("== CAR-19 ==")

    # Number of vehicles to evaluate – use top 5 as an example budget
    budget = min(5, len(rows))
    if budget == 0:
        print("No vehicles available for comparison.")
        return

    # Compute system‑selected vehicles (top N)
    scored_rows = compute_balanced_scores(rows)
    system_top = scored_rows[:budget]
    avg_system = _average_score(system_top)

    # Randomly selected vehicles
    random_rows = random.sample(rows, budget)
    # Use their score values – we need a score for each; compute it
    random_scored = []
    for r in random_rows:
        # Create a dummy row with same structure for score calculation
        # Compute score using the same logic as compute_balanced_scores
        # Instead of re‑implementing, use the original function
        # We can temporarily add the row to a list and score it
        temp = compute_balanced_scores([r])[0]
        random_scored.append(temp)
    avg_random = _average_score(random_scored)

    # Calculate improvement percentage
    improvement = ((avg_system - avg_random) / avg_random * 100) if avg_random else 0.0

    # Output results
    print(f"Average score of top {budget} vehicles (system): {format_value(avg_system)}")
    print(f"Average score of {budget} randomly selected vehicles: {format_value(avg_random)}")
    print(f"Improvement over random selection: {format_value(improvement, '{:.2f}')}%")


def _car_20(rows):
    """
    Compare regression predictions vs historical averages for MPG and weight
    on a held‑out test set. Report RMSE for each method and the improvement.
    """
    print("== CAR-20 ==")

    if not rows:
        print("No data available for evaluation.")
        return

    # Shuffle and split into train/test (70/30)
    random.seed(0)
    shuffled = rows[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.7)
    train_rows = shuffled[:split_idx]
    test_rows = shuffled[split_idx:]

    # Helper to evaluate a target
    def evaluate(target, predictors):
        # Train regression
        X_train, y_train = _prepare_regression(train_rows, target, predictors)
        beta, _ = _fit_regression(X_train, y_train)

        # Historical average from training
        train_vals = [r[target] for r in train_rows if isinstance(r[target], (int, float))]
        hist_mean = sum(train_vals) / len(train_vals) if train_vals else 0.0

        # Compute RMSE for test set
        mse_reg = 0.0
        mse_hist = 0.0
        n = 0
        for r in test_rows:
            true_val = r[target]
            if not isinstance(true_val, (int, float)):
                continue
            pred_reg = _predict(r, beta, predictors)
            pred_hist = hist_mean
            mse_reg += (true_val - pred_reg) ** 2
            mse_hist += (true_val - pred_hist) ** 2
            n += 1

        rmse_reg = (mse_reg / n) ** 0.5 if n else 0.0
        rmse_hist = (mse_hist / n) ** 0.5 if n else 0.0
        improvement = rmse_hist - rmse_reg  # positive means regression better
        return rmse_reg, rmse_hist, improvement

    # MPG evaluation
    mpg_preds = evaluate("Mpg+", ["Lbs-", "Acc+", "Volume", "HpX", "Clndrs"])
    # Weight evaluation
    weight_preds = evaluate("Lbs-", ["Mpg+", "Acc+", "Volume", "HpX", "Clndrs"])

    # Output
    print(f"MPG prediction (regression) RMSE: {format_value(mpg_preds[0])}")
    print(f"MPG prediction (historical average) RMSE: {format_value(mpg_preds[1])}")
    print(f"MPG RMSE improvement: {format_value(mpg_preds[2], '{:.2f}')}\n")

    print(f"Weight prediction (regression) RMSE: {format_value(weight_preds[0])}")
    print(f"Weight prediction (historical average) RMSE: {format_value(weight_preds[1])}")
    print(f"Weight RMSE improvement: {format_value(weight_preds[2], '{:.2f}')}")

    if mpg_preds[2] > 0 and weight_preds[2] > 0:
        print("\nThe regression‑based system provides a meaningful improvement over the historical‑average estimates on unseen records.")
    else:
        print("\nNo improvement detected; the simple historical‑average estimate is comparable to the regression model on unseen records.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    _car_19(rows)
    _car_20(rows)


if __name__ == "__main__":
    main()

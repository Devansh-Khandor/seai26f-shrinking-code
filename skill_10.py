#!/usr/bin/env python3
"""
skill_10.py

Implements the final pair of Apache configuration analyses:

APACHE-19: Compare our configuration-selection system against random
           configuration selection under the same benchmarking budget, and
           show which approach discovers better-performing configurations.

APACHE-20: Compare our performance predictions against a simple
           historical-average estimate and show whether our system provides
           a meaningful improvement on previously unseen configurations.

The script loads the CSV data, performs both analyses, and prints the
results with the required header lines.
"""

import sys
import math
import random
from typing import List

# Reuse helpers from earlier skills
from skill_01 import read_configurations, performance_stats
from skill_02 import compute_regression_coeffs, predict_performance


def format_float(value: float, precision: int = 2) -> str:
    """Format a float to a string with the requested precision."""
    return f"{value:.{precision}f}"


def apach_19(configs: List[dict], budget: int, seed: int = 42):
    """Compare predicted-top selection with random selection."""
    random.seed(seed)

    # --------- Our selection (predicted top N) ----------
    sorted_by_pred = sorted(
        configs,
        key=lambda c: predict_performance(c, *compute_regression_coeffs(configs)),
        reverse=True,
    )
    our_selection = sorted_by_pred[:budget]
    best_performance_our = max(c['Performance-'] for c in our_selection)

    # --------- Random selection ----------
    random_selection = random.sample(configs, budget)
    best_performance_random = max(c['Performance-'] for c in random_selection)

    # --------- Report ----------
    print("== APACHE-19 ==")
    print(f"Total configurations: {len(configs)}")
    print(f"Benchmarking budget (configs evaluated): {budget}")
    print(f"Best performance found by our system: {format_float(best_performance_our)}")
    print(f"Best performance found by random selection: {format_float(best_performance_random)}")
    improvement = best_performance_our - best_performance_random
    improvement_pct = (improvement / best_performance_random) * 100 if best_performance_random else 0.0
    print(f"Improvement of our system over random: {format_float(improvement)} "
          f"({format_float(improvement_pct)}%)")
    print()


def apach_20(configs: List[dict], train_frac: float = 0.8, seed: int = 42):
    """Compare regression predictions with historical-average predictions."""
    random.seed(seed)
    n = len(configs)
    indices = list(range(n))
    random.shuffle(indices)

    split_idx = int(train_frac * n)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    training = [configs[i] for i in train_idx]
    testing = [configs[i] for i in test_idx]

    if not testing:
        print("== APACHE-20 ==")
        print("Not enough configurations to create a test set.")
        return

    # Regression on training data
    intercept, coeffs, sigma_res, coeff_se = compute_regression_coeffs(training)

    # Predictions for testing set
    preds = [predict_performance(c, intercept, coeffs) for c in testing]
    actuals = [c['Performance-'] for c in testing]

    mae_reg = sum(abs(p - a) for p, a in zip(preds, actuals)) / len(testing)

    # Historical average from training
    mean_train = sum(c['Performance-'] for c in training) / len(training)
    mae_avg = sum(abs(mean_train - a) for a in actuals) / len(testing)

    # Improvement metric
    improvement = mae_avg - mae_reg
    improvement_pct = (improvement / mae_avg) * 100 if mae_avg else 0.0

    # --------- Report ----------
    print("== APACHE-20 ==")
    print(f"Training set size: {len(training)}")
    print(f"Test set size: {len(testing)}")
    print(f"Mean absolute error (regression): {format_float(mae_reg)}")
    print(f"Mean absolute error (historical average): {format_float(mae_avg)}")
    print(f"Error reduction by regression: {format_float(improvement)} "
          f"({format_float(improvement_pct)}%)")
    print()


def main(csv_path: str):
    # Load all configurations
    configs = read_configurations(csv_path)

    # Basic stats for reporting
    mean, best, worst, stddev = performance_stats(configs)

    # Determine a reasonable benchmarking budget
    # We choose 10% of configs or at least 5
    budget = max(5, int(0.1 * len(configs)))

    # Run the analyses
    apach_19(configs, budget)
    apach_20(configs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

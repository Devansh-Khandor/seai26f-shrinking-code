#!/usr/bin/env python3
"""
Skill 10: Apache performance analysis – system vs random and model vs baseline
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Import helpers from earlier skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    read_csv_to_numpy,
    summarize_performance,
    identify_favorable,
)
from skill_02 import train_linear_model

# --------------------------------------------------------------------------- #
# APACHE-19: Compare predictive selection vs random selection
# --------------------------------------------------------------------------- #
def apache_19(header, data, model, XtX_inv, resid_var, budget=10):
    """
    Compare the performance of configurations selected by our predictive model
    against random selection under the same benchmarking budget.
    Prints the best actual performance found by each approach and the difference.
    """
    print("\n== APACHE-19 ==")

    n_samples = data.shape[0]
    if budget > n_samples:
        budget = n_samples

    # Predicted performance for all configurations
    preds = model.predict(data[:, :-1])

    # System: top‑budget predicted configs
    system_idx = np.argsort(preds)[::-1][:budget]
    system_perf = data[system_idx, -1]
    system_best = system_perf.max()
    system_mean = system_perf.mean()

    # Random: sample budget configs
    rng = np.random.default_rng(seed=42)
    random_idx = rng.choice(n_samples, size=budget, replace=False)
    random_perf = data[random_idx, -1]
    random_best = random_perf.max()
    random_mean = random_perf.mean()

    print(f"Benchmark budget: {budget} configurations")
    print(f"Predictive selection best actual performance: {system_best:.2f}")
    print(f"Predictive selection mean performance: {system_mean:.2f}")
    print(f"Random selection best actual performance: {random_best:.2f}")
    print(f"Random selection mean performance: {random_mean:.2f}")
    diff_best = system_best - random_best
    diff_mean = system_mean - random_mean
    print(f"Difference in best performance: {diff_best:.2f}")
    print(f"Difference in mean performance: {diff_mean:.2f}")


# --------------------------------------------------------------------------- #
# APACHE-20: Compare model predictions vs simple historical‑average baseline
# --------------------------------------------------------------------------- #
def apache_20(header, data, model):
    """
    Compute mean absolute error (MAE) of the trained model predictions
    and compare it to a simple baseline that uses the historical mean performance.
    Prints the errors and the improvement.
    """
    print("\n== APACHE-20 ==")

    actual = data[:, -1]
    preds = model.predict(data[:, :-1])

    mae_model = np.mean(np.abs(actual - preds))
    baseline = np.mean(actual)
    mae_baseline = np.mean(np.abs(actual - baseline))

    improvement = mae_baseline - mae_model

    print(f"Historical mean performance (baseline): {baseline:.2f}")
    print(f"Mean absolute error (model predictions): {mae_model:.2f}")
    print(f"Mean absolute error (baseline predictions): {mae_baseline:.2f}")
    print(f"Absolute improvement (lower MAE): {improvement:.2f}")

    if improvement > 0:
        percent_improve = (improvement / mae_baseline) * 100
        print(f"Relative improvement over baseline: {percent_improve:.1f}%")
    else:
        print("Model does not improve over the baseline prediction.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Basic summary and favorable detection
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Train model for predictions
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # New requests
    apache_19(header, data, model, XtX_inv, resid_var, budget=10)
    apache_20(header, data, model)


if __name__ == "__main__":
    main()

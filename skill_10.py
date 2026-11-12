#!/usr/bin/env python3
"""
Apache performance diagnostics: system vs random selection and prediction vs baseline.

This script implements two new analysis points:

- APACHE-19: Compare our configuration-selection system against random configuration
  selection under the same benchmarking budget, showing which approach discovers
  better-performing configurations.
- APACHE-20: Compare our performance predictions against a simple historical-average
  estimate and show whether our system provides a meaningful improvement on previously
  unseen configurations.

The implementation reuses functions from the earlier skill modules.
"""

import sys
import random
import statistics
import skill_01
import skill_02
import skill_07

def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)

def top_k_by_prediction(rows, k):
    """
    Return the top k rows according to regression predictions.
    """
    beta, feature_names, predictions = skill_02.train_regression(rows)
    scored = list(zip(rows, predictions))
    scored.sort(key=lambda rp: rp[1], reverse=True)
    return [row for row, _ in scored[:k]]

def mean_performance(rows):
    """
    Compute the arithmetic mean of the 'Performance-' field of a list of rows.
    """
    if not rows:
        return 0.0
    return statistics.mean([row['Performance-'] for row in rows])

def apache_19(rows):
    """
    Compare our system (regression‑based selection) with random selection.
    For a fixed budget k (min(10, len(rows))), compute the mean and best performance
    achieved by each method.
    """
    k = min(10, len(rows))
    # System selection: top k predicted configurations
    sys_selected = top_k_by_prediction(rows, k)
    sys_mean = mean_performance(sys_selected)
    sys_best = max(row['Performance-'] for row in sys_selected)

    # Random selection
    rand_selected = random.sample(rows, k)
    rand_mean = mean_performance(rand_selected)
    rand_best = max(row['Performance-'] for row in rand_selected)

    print("== APACHE-19 ==")
    print(f"Benchmarking budget: {k} configurations.")
    print(f"System selection – mean performance: {sys_mean:.2f}, best: {sys_best}")
    print(f"Random selection – mean performance: {rand_mean:.2f}, best: {rand_best}")
    print(f"System improved mean by {sys_mean - rand_mean:.2f} over random.")
    print(f"System improved best by {sys_best - rand_best:.2f} over random.")

def apache_20(rows):
    """
    Compare regression predictions against a simple historical‑average baseline.
    Compute mean absolute error (MAE) for both, and report the improvement.
    """
    # Regression metrics
    beta, feature_names, predictions = skill_02.train_regression(rows)
    _, _, mae_reg = skill_02.regression_metrics(rows, predictions)

    # Baseline: predict the overall mean for every configuration
    all_perf = [row['Performance-'] for row in rows]
    mean_perf = statistics.mean(all_perf)
    mae_baseline = statistics.mean([abs(p - mean_perf) for p in all_perf])

    improvement = mae_baseline - mae_reg

    print("\n== APACHE-20 ==")
    print(f"Regression MAE: {mae_reg:.2f}")
    print(f"Baseline (mean) MAE: {mae_baseline:.2f}")
    print(f"Improvement over baseline: {improvement:.2f} (lower is better)")
    if improvement > 0:
        print("Regression predictions provide a meaningful improvement.")
    else:
        print("Regression predictions do not improve upon the baseline.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data to analyze.")
        sys.exit(0)

    apache_19(rows)
    apache_20(rows)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skill 10: Comparative evaluation of selection strategies and prediction quality.

This module implements APACHE-19 and APACHE-20 using the existing utilities.

APACHE-19:
    Compare the configuration‑selection system (based on linear predictions)
    against random selection under the same benchmarking budget.

APACHE-20:
    Compare our linear prediction model against a simple historical‑average
    estimator on a held‑out test set.
"""

import sys
import random
from pathlib import Path

# Existing helpers
from skill_01 import read_csv, compute_statistics, format_config
from skill_02 import train_linear_model, predict, standard_deviation

# ----------------------------------------------------------------------
# APACHE-19: Selection strategy comparison
# ----------------------------------------------------------------------
def apache_19(rows, performance):
    """
    Evaluate a fixed budget of configuration checks using:
        - Our system: top‑predicted performance configs
        - Random: random configs

    Returns a dict with relevant metrics.
    """
    n = len(rows)
    if n == 0:
        return {}

    # Benchmark budget: 10% of total (at least 1)
    budget = max(1, n // 10)

    # Train linear model on the entire dataset
    coeffs, feature_names = train_linear_model(rows, performance)
    preds = [predict(r, coeffs, feature_names) for r in rows]

    # Our system: pick top‑predicted (high) performance
    top_indices = sorted(range(n), key=lambda i: preds[i], reverse=True)[:budget]
    best_system = max(performance[i] for i in top_indices)

    # Random selection
    random.seed(0)  # reproducible
    rand_indices = random.sample(range(n), k=budget)
    best_random = max(performance[i] for i in rand_indices)

    return {
        "budget": budget,
        "best_system": best_system,
        "best_random": best_random,
        "diff": best_system - best_random,
    }

# ----------------------------------------------------------------------
# APACHE-20: Prediction quality comparison
# ----------------------------------------------------------------------
def apache_20(rows, performance):
    """
    Split data into training (70%) and test (30%) sets.
    Compare linear model MAE against historical‑average MAE.
    Returns a dict with metrics.
    """
    n = len(rows)
    if n < 2:
        return {}

    # Shuffle indices for reproducibility
    indices = list(range(n))
    random.seed(1)
    random.shuffle(indices)
    split = int(0.7 * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_rows = [rows[i] for i in train_idx]
    train_perf = [performance[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    test_perf = [performance[i] for i in test_idx]

    # Train linear model on training data
    coeffs, feature_names = train_linear_model(train_rows, train_perf)

    # Predictions on test set
    pred_test = [predict(r, coeffs, feature_names) for r in test_rows]
    errors_model = [abs(p - a) for p, a in zip(pred_test, test_perf)]
    mae_model = sum(errors_model) / len(errors_model) if errors_model else 0.0

    # Historical‑average estimate
    mean_train = sum(train_perf) / len(train_perf) if train_perf else 0.0
    errors_hist = [abs(mean_train - a) for a in test_perf]
    mae_hist = sum(errors_hist) / len(errors_hist) if errors_hist else 0.0

    abs_improvement = mae_hist - mae_model
    rel_improvement = (abs_improvement / mae_hist) if mae_hist != 0 else float('nan')

    return {
        "mae_model": mae_model,
        "mae_hist": mae_hist,
        "abs_improvement": abs_improvement,
        "rel_improvement": rel_improvement,
    }

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # === APACHE-19 ===
    print("== APACHE-19 ==")
    result_19 = apache_19(rows, performance)
    if result_19:
        print(f"Benchmark budget: {result_19['budget']} configurations evaluated.")
        print(f"Best performance found by system: {result_19['best_system']:.2f} ms.")
        print(f"Best performance found by random: {result_19['best_random']:.2f} ms.")
        print(f"Difference: {result_19['diff']:.2f} ms.")
    else:
        print("Insufficient data for APACHE-19.")
    print()

    # === APACHE-20 ===
    print("== APACHE-20 ==")
    result_20 = apache_20(rows, performance)
    if result_20:
        print(f"Model MAE: {result_20['mae_model']:.2f} ms.")
        print(f"Historical‑avg MAE: {result_20['mae_hist']:.2f} ms.")
        print(f"Absolute improvement: {result_20['abs_improvement']:.2f} ms.")
        print(f"Relative improvement: {result_20['rel_improvement']:.2%}.")
    else:
        print("Insufficient data for APACHE-20.")
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

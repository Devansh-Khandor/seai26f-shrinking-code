#!/usr/bin/env python3
"""
skill_10.py – Comparative analysis of configuration selection and prediction accuracy.

APACHE-19: Compare a model‑guided configuration‑selection strategy against random
selection under the same benchmarking budget, and report which approach finds
the best performing configuration.

APACHE-20: Compare the regression‑based performance predictions with a naive
historical‑average estimate on a hold‑out test set, and quantify the improvement.
"""

import sys
import random
import numpy as np
import skill_01 as s1          # load_csv, compute_summary
import skill_02 as s2          # feature_matrix, target_vector, train_regressor, predict_performance, residual_std
import skill_07 as s7          # apache_14 for model‑based recommendation

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def best_performance(rows, indices):
    """Return the lowest Performance- value among the rows at the given indices."""
    return min(int(rows[i]['Performance-']) for i in indices)

def split_dataset(rows, train_ratio=0.7, seed=42):
    """Randomly split rows into train and test sets, returning two lists."""
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_train = int(len(rows) * train_ratio)
    return rows[:n_train], rows[n_train:]

# ----------------------------------------------------------------------
# APACHE-19: Selection strategy comparison
# ----------------------------------------------------------------------
def apache_19(rows, headers, budget=5):
    """Compare model‑guided selection with random selection."""
    n = len(rows)
    budget = min(budget, n)

    # Model‑guided selection using skill_07's recommendation
    model_recs = s7.apache_14(rows, headers, budget=budget)
    model_indices = [rec['config_num'] - 1 for rec in model_recs]  # 0‑based

    # Random selection
    all_indices = list(range(n))
    random.seed(42)  # deterministic for reproducibility
    random.shuffle(all_indices)
    random_indices = all_indices[:budget]

    # Evaluate best performance achieved by each approach
    best_model_perf = best_performance(rows, model_indices)
    best_random_perf = best_performance(rows, random_indices)

    # Prepare results
    result = {
        'budget': budget,
        'best_model_perf': best_model_perf,
        'best_random_perf': best_random_perf,
        'model_indices': model_indices,
        'random_indices': random_indices
    }
    return result

# ----------------------------------------------------------------------
# APACHE-20: Prediction vs baseline comparison
# ----------------------------------------------------------------------
def apache_20(rows, headers):
    """Compare regression predictions with a historical‑average baseline."""
    # Split into train / test
    train_rows, test_rows = split_dataset(rows, train_ratio=0.7, seed=42)

    # Feature matrices and targets
    X_train, _ = s2.feature_matrix(train_rows, headers)
    y_train = s2.target_vector(train_rows, headers)
    X_test, _  = s2.feature_matrix(test_rows, headers)
    y_test = s2.target_vector(test_rows, headers)

    # Train model
    model = s2.train_regressor(X_train, y_train)
    preds = s2.predict_performance(model, X_test)

    # Baseline: historical mean of training performance
    baseline = np.mean(y_train)
    baseline_preds = np.full_like(preds, fill_value=baseline)

    # Errors
    mse_model = np.mean((y_test - preds) ** 2)
    mse_baseline = np.mean((y_test - baseline_preds) ** 2)

    # Improvement percentage
    improvement = (mse_baseline - mse_model) / mse_baseline * 100 if mse_baseline != 0 else float('inf')

    result = {
        'mse_model': mse_model,
        'mse_baseline': mse_baseline,
        'improvement_percent': improvement,
        'n_train': len(train_rows),
        'n_test': len(test_rows)
    }
    return result

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = s1.load_csv(csv_path)

    # APACHE-19
    print("== APACHE-19 ==")
    res19 = apache_19(rows, headers, budget=5)
    print(f"  Budget (configurations evaluated): {res19['budget']}")
    print(f"  Best performance with model‑guided selection: {res19['best_model_perf']}")
    print(f"  Best performance with random selection: {res19['best_random_perf']}")
    better = "model‑guided" if res19['best_model_perf'] <= res19['best_random_perf'] else "random"
    print(f"  Model‑guided selection found a better configuration: {better}")

    # APACHE-20
    print("\n== APACHE-20 ==")
    res20 = apache_20(rows, headers)
    print(f"  Training set size: {res20['n_train']}, Test set size: {res20['n_test']}")
    print(f"  Mean squared error of regression predictions: {res20['mse_model']:.2f}")
    print(f"  Mean squared error of historical‑average baseline: {res20['mse_baseline']:.2f}")
    print(f"  Improvement over baseline: {res20['improvement_percent']:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_10.py <csv_path>")
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_07.py – Active discovery of low‑performance (good) Apache configurations
and recommendation of an optimal evaluation order under a limited benchmarking budget.

The module implements two new APACHE analyses:

APACHE-13 – Discover a configuration with a low performance‑metric value while evaluating
              as few candidate configurations as reasonably possible.

APACHE-14 – When a limited benchmarking budget is available, recommend the
              configurations that should be evaluated first to discover promising
              server settings early.

The implementation reuses the utilities from the earlier skills (skill_01 and
skill_02) for loading data, feature extraction, regression modelling,
prediction, and residual analysis.
"""

import sys
import numpy as np
import skill_01 as s1
import skill_02 as s2

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def load_csv(path):
    """Delegate to skill_01 for consistency."""
    return s1.load_csv(path)

def feature_matrix(rows, headers):
    """Return numpy array of features for each row (excluding the target)."""
    return s2.feature_matrix(rows, headers)[0], s2.feature_matrix(rows, headers)[1]

def target_vector(rows, headers):
    """Return numpy array of target values (Performance-)."""
    return s2.target_vector(rows, headers)

def train_model(X, y):
    """Train and return a linear regression model."""
    return s2.train_regressor(X, y)

def predict_performance(model, X):
    """Return predicted performance values."""
    return s2.predict_performance(model, X)

def residual_std(model, X, y):
    """Return standard deviation of residuals."""
    return s2.residual_std(model, X, y)

# ----------------------------------------------------------------------
# APACHE-13: Efficient low‑performance discovery
# ----------------------------------------------------------------------
def apache_13(rows, headers):
    """Identify the configuration with the lowest actual performance
    while evaluating as few candidates as possible, using the regression
    model predictions to order the evaluation.
    """
    # Build data for modelling
    X, feature_cols = feature_matrix(rows, headers)
    y = target_vector(rows, headers)

    # Train a linear regression model
    model = train_model(X, y)

    # Predict all configurations
    preds = predict_performance(model, X)

    # Sort indices by predicted performance (ascending – lower is better)
    sorted_indices = np.argsort(preds)

    # Determine the true minimum performance value
    min_perf = int(np.min(y))
    min_perf_idx = np.where(y == min_perf)[0][0]  # first occurrence

    # Iterate over sorted predictions until we hit the true minimum
    evaluated = 0
    for idx in sorted_indices:
        evaluated += 1
        if idx == min_perf_idx:
            # Found the best configuration
            actual_perf = int(y[idx])
            predicted_perf = preds[idx]
            config_num = idx + 1  # 1‑based numbering for display
            return {
                'config_num': config_num,
                'actual_perf': actual_perf,
                'predicted_perf': predicted_perf,
                'evaluated': evaluated
            }

    # Fallback – should never reach here
    return {
        'config_num': None,
        'actual_perf': None,
        'predicted_perf': None,
        'evaluated': evaluated
    }

# ----------------------------------------------------------------------
# APACHE-14: Recommendation of first configurations to benchmark
# ----------------------------------------------------------------------
def apache_14(rows, headers, budget=5):
    """Recommend the first few configurations to evaluate under a limited
    benchmarking budget.  The recommendation is based on the lowest
    predicted performance values (i.e., the most promising settings).
    """
    X, feature_cols = feature_matrix(rows, headers)
    y = target_vector(rows, headers)
    model = train_model(X, y)
    preds = predict_performance(model, X)

    # Sort by predicted performance ascending
    sorted_indices = np.argsort(preds)

    # Select top budgets (or fewer if dataset is smaller)
    n_select = min(budget, len(rows))
    selected = sorted_indices[:n_select]

    result = []
    for i, idx in enumerate(selected, start=1):
        result.append({
            'rank': i,
            'config_num': idx + 1,
            'predicted_perf': float(preds[idx]),
            'actual_perf': int(y[idx])
        })
    return result

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)

    # ------------------------------------------------------------------
    # APACHE-13
    # ------------------------------------------------------------------
    print("== APACHE-13 ==")
    res13 = apache_13(rows, headers)
    if res13['config_num'] is not None:
        print(f"  Discovered configuration #{res13['config_num']} with lowest actual performance.")
        print(f"  Actual performance: {res13['actual_perf']}")
        print(f"  Predicted performance: {res13['predicted_perf']:.2f}")
        print(f"  Configurations evaluated before finding the optimum: {res13['evaluated']}")
    else:
        print("  No configuration found (dataset may be empty).")

    # ------------------------------------------------------------------
    # APACHE-14
    # ------------------------------------------------------------------
    print("\n== APACHE-14 ==")
    recommendations = apache_14(rows, headers, budget=5)
    if not recommendations:
        print("  No configurations available to recommend.")
    else:
        print(f"  Recommend evaluating the following {len(recommendations)} configurations first:")
        for rec in recommendations:
            print(f"    Rank {rec['rank']}: Configuration #{rec['config_num']} – "
                  f"Predicted: {rec['predicted_perf']:.2f}, "
                  f"Actual: {rec['actual_perf']}")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_07.py <csv_path>")
    main(sys.argv[1])

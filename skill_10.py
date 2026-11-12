#!/usr/bin/env python3
"""
Skill 10 – Apache benchmark comparison and prediction evaluation

This script implements two new requests:

* APACHE-19: Compare our configuration-selection system against random
  configuration selection under the same benchmarking budget, and show
  which approach discovers better-performing configurations.

* APACHE-20: Compare our performance predictions against a simple
  historical‑average estimate and show whether our system provides a
  meaningful improvement on previously unseen configurations.

The code re‑uses existing helper functions from the earlier skill
modules to keep the implementation concise and consistent.
"""

import sys
import argparse
import random
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from the earlier skills
# ----------------------------------------------------------------------
try:
    from skill_01 import read_csv, find_performance_column, compute_stats
except Exception:
    # Minimal fallback if skill_01 is missing
    import csv
    from statistics import mean, median

    def read_csv(path: str):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = []
            for row in reader:
                int_row = {}
                for k, v in row.items():
                    try:
                        int_row[k] = int(v)
                    except ValueError:
                        int_row[k] = v
                rows.append(int_row)
        return header, rows

    def find_performance_column(header):
        for col in header:
            if col.lower().startswith("performance"):
                return col
        raise KeyError("No performance column found in header.")

    def compute_stats(performance_values):
        return {
            "mean": mean(performance_values),
            "median": median(performance_values),
            "std": np.std(performance_values, ddof=0),
            "best": min(performance_values),
            "worst": max(performance_values),
        }

try:
    from skill_02 import train_regression, predict_performance
except Exception:
    # Minimal regression helpers if skill_02 is missing
    from sklearn.linear_model import LinearRegression

    def train_regression(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict_performance(model, X):
        return model.predict(X)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def get_actual_performances(rows, perf_col, indices):
    """Return a list of actual performance values for given indices."""
    return [rows[i][perf_col] for i in indices]


def best_actual_performance(rows, perf_col, indices):
    """Return the best (minimum) actual performance among selected configs."""
    if not indices:
        return None
    return min(get_actual_performances(rows, perf_col, indices))


# ----------------------------------------------------------------------
# APACHE-19: Compare selection strategies
# ----------------------------------------------------------------------
def compare_selection_strategies(rows, perf_col, predictions, budget):
    """
    Compare the performance discovered by our model‑based selection vs
    random selection within the same budget.

    Prints a summary including the best actual performance found and the
    mean performance of selected configurations for both approaches.
    """
    n_rows = len(rows)
    # Ensure budget does not exceed available configurations
    budget = min(budget, n_rows)

    # Model‑based selection: indices sorted by predicted performance
    model_sorted_idx = sorted(range(n_rows), key=lambda i: predictions[i])
    model_selected = model_sorted_idx[:budget]

    # Random selection: random sample without replacement
    random_selected = random.sample(range(n_rows), budget)

    # Compute metrics
    model_best = best_actual_performance(rows, perf_col, model_selected)
    random_best = best_actual_performance(rows, perf_col, random_selected)

    model_mean = np.mean(get_actual_performances(rows, perf_col, model_selected))
    random_mean = np.mean(get_actual_performances(rows, perf_col, random_selected))

    # Print results
    print("== APACHE-19 ==")
    print(f"Benchmarking budget: {budget} configurations")
    print(f"Model‑based selection best performance: {model_best} ms")
    print(f"Random selection best performance: {random_best} ms")
    print(f"Model‑based selection mean performance: {model_mean:.1f} ms")
    print(f"Random selection mean performance: {random_mean:.1f} ms")
    if model_best is not None and random_best is not None:
        diff = random_best - model_best
        if diff > 0:
            print(f"Model outperforms random by {diff} ms on the best config.")
        else:
            print(f"Random selection matches or outperforms model by {-diff} ms.")


# ----------------------------------------------------------------------
# APACHE-20: Prediction quality vs historical average
# ----------------------------------------------------------------------
def compare_prediction_quality(rows, perf_col, predictions):
    """
    Compare the regression model predictions against a simple baseline
    that predicts the historical mean for every configuration.

    Prints MAE for both predictors and the percentage improvement
    achieved by the regression model.
    """
    actual = np.array([row[perf_col] for row in rows])
    baseline_pred = np.full_like(actual, fill_value=np.mean(actual), dtype=float)

    mae_model = np.mean(np.abs(actual - predictions))
    mae_baseline = np.mean(np.abs(actual - baseline_pred))

    improvement = (mae_baseline - mae_model) / mae_baseline * 100 if mae_baseline != 0 else np.nan

    print("\n== APACHE-20 ==")
    print(f"Historical‑average baseline MAE: {mae_baseline:.1f} ms")
    print(f"Regression model MAE: {mae_model:.1f} ms")
    print(f"MAE improvement over baseline: {improvement:.1f}%")
    if improvement > 0:
        print("The regression model provides a meaningful improvement over the baseline.")
    else:
        print("No improvement achieved; the regression model does not outperform the baseline.")


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)

    # Build feature matrix X and target vector y
    config_cols = [c for c in header if c != perf_col]
    X = np.array([[row[col] for col in config_cols] for row in rows])
    y = np.array([row[perf_col] for row in rows])

    # Train regression model and generate predictions
    model = train_regression(X, y)
    predictions = predict_performance(model, X)

    # Execute the two new requests
    # Use a default budget of 10 or less if dataset is smaller
    budget = min(10, len(rows))
    compare_selection_strategies(rows, perf_col, predictions, budget)
    compare_prediction_quality(rows, perf_col, predictions)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache benchmark comparison and prediction evaluation skill."
    )
    parser.add_argument(
        "csv_path",
        help="Path to Apache_AllMeasurements.csv",
    )
    args = parser.parse_args()

    try:
        main(args.csv_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

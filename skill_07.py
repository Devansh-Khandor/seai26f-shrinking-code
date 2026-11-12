#!/usr/bin/env python3
"""
Skill 07 – Apache targeted benchmarking

This script implements two new requests:

* APACHE-13: Help me discover a configuration with a low performance-metric
  value while evaluating as few candidate configurations as reasonably
  possible.
* APACHE-14: When I have a limited benchmarking budget, recommend which
  configurations should be evaluated first so that we can discover
  promising server settings early.

The implementation re‑uses utilities from skill_01 and skill_02 to keep
the code DRY and maintainable.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import helpers from the earlier skills
# ----------------------------------------------------------------------
try:
    from skill_01 import read_csv, find_performance_column
except Exception:
    # Minimal fallback if skill_01 is missing
    import csv

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

try:
    from skill_02 import train_regression, predict_performance
except Exception:
    # Fallback simple linear regression implementation
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
def _build_feature_matrix(rows, config_cols):
    """Return a numpy array of shape (n_rows, n_features)."""
    return np.array([[row[col] for col in config_cols] for row in rows])


def _sorted_indices_by_prediction(preds, ascending=True):
    """Return indices sorted by predicted performance."""
    return sorted(
        range(len(preds)), key=lambda i: preds[i], reverse=not ascending
    )


def _format_config(row, perf_col):
    """Return a string representation of a configuration, excluding the performance column."""
    cfg = {k: v for k, v in row.items() if k != perf_col}
    return str(cfg)


# ----------------------------------------------------------------------
# APACHE-13: Discover a configuration with low metric
# ----------------------------------------------------------------------
def discover_best_prediction(rows, perf_col, predictions, top_n=3):
    """
    Identify the configuration with the lowest predicted performance.
    Show its actual performance as well as the next best predictions
    to give a sense of the prediction spread.
    """
    sorted_idx = _sorted_indices_by_prediction(predictions, ascending=True)
    best_idx = sorted_idx[0]
    best_row = rows[best_idx]
    best_pred = predictions[best_idx]
    best_actual = best_row[perf_col]

    print("== APACHE-13 ==")
    print(f"Recommended configuration (index={best_idx}):")
    print(f"  Settings: {_format_config(best_row, perf_col)}")
    print(f"  Predicted performance: {best_pred:.1f} ms")
    print(f"  Actual performance: {best_actual} ms")
    print()

    # Show the next best predictions for context
    if len(sorted_idx) > 1:
        print(f"Top {min(top_n, len(sorted_idx))} predicted configurations:")
        for rank, idx in enumerate(sorted_idx[:top_n], start=1):
            row = rows[idx]
            pred = predictions[idx]
            actual = row[perf_col]
            print(
                f"  Rank {rank}: idx={idx}, predicted={pred:.1f} ms, actual={actual} ms, settings={_format_config(row, perf_col)}"
            )
    else:
        print("Only one configuration available.")


# ----------------------------------------------------------------------
# APACHE-14: Recommend configurations to evaluate first given a budget
# ----------------------------------------------------------------------
def recommend_configurations(rows, perf_col, predictions, budget=5):
    """
    Provide a ranked list of configurations to evaluate first, sorted
    by predicted performance (lowest first). The list length is limited
    by the provided budget.
    """
    sorted_idx = _sorted_indices_by_prediction(predictions, ascending=True)
    top_n = min(budget, len(sorted_idx))

    print("== APACHE-14 ==")
    print(f"Top {top_n} configurations to evaluate first (budget = {budget}):")
    for rank, idx in enumerate(sorted_idx[:top_n], start=1):
        row = rows[idx]
        pred = predictions[idx]
        actual = row[perf_col]
        print(
            f"  Rank {rank}: idx={idx}, predicted={pred:.1f} ms, actual={actual} ms, settings={_format_config(row, perf_col)}"
        )
    if top_n < len(sorted_idx):
        print(f"  ...and {len(sorted_idx)-top_n} more configurations not shown.")


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)
    config_cols = [col for col in header if col != perf_col]

    # Build feature matrix and target vector
    X = _build_feature_matrix(rows, config_cols)
    y = np.array([row[perf_col] for row in rows])

    # Train regression model and get predictions
    model = train_regression(X, y)
    predictions = predict_performance(model, X)

    # Execute the two new requests
    discover_best_prediction(rows, perf_col, predictions, top_n=3)
    print()  # blank line between requests
    recommend_configurations(rows, perf_col, predictions, budget=5)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache targeted benchmarking skill."
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

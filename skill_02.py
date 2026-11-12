#!/usr/bin/env python3
"""
Skill 02 – Apache predictive performance analysis

This script extends the existing Apache analysis by providing:

* APACHE-03: A simple regression model that estimates the expected
  performance value of a configuration based on its settings.
* APACHE-04: For historically observed configurations that differ by
  exactly one setting, an estimate of the performance difference
  along with a measure of uncertainty.

The code re‑uses helper functions from skill_01.py where possible.
"""

import sys
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ----------------------------------------------------------------------
# Import utilities from skill_01
# ----------------------------------------------------------------------
# The following imports assume skill_01.py is in the same directory.
# Importing the specific functions we need.
try:
    from skill_01 import (
        read_csv,
        find_performance_column,
        column_modes,
        distinguish_settings,
    )
except Exception:
    # If skill_01 is not present, re‑define minimal helpers
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

    def column_modes(rows, cols):
        from collections import Counter

        modes = {}
        for col in cols:
            counter = Counter(row[col] for row in rows)
            mode_value = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            modes[col] = mode_value
        return modes

    def distinguish_settings(overall_modes, favorable_modes):
        diff = {}
        for col, fav_mode in favorable_modes.items():
            overall_mode = overall_modes[col]
            if fav_mode != overall_mode:
                diff[col] = (fav_mode, overall_mode)
        return diff

# ----------------------------------------------------------------------
# Helper functions for prediction
# ----------------------------------------------------------------------
def train_regression(X: np.ndarray, y: np.ndarray):
    """Fit a simple linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_performance(model, X: np.ndarray):
    """Predict performance values for given feature matrix."""
    return model.predict(X)


def pairwise_differences(rows, config_cols, perf_col, predictions):
    """
    Find all pairs of rows that differ by exactly one configuration setting.

    Returns a list of dicts:
        {
            'idx1': int,
            'idx2': int,
            'actual_diff': float,
            'pred_diff': float,
            'residual': float
        }
    """
    n = len(rows)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diff_count = sum(
                1
                for col in config_cols
                if rows[i][col] != rows[j][col]
            )
            if diff_count == 1:
                actual_diff = abs(rows[i][perf_col] - rows[j][perf_col])
                pred_diff = abs(predictions[i] - predictions[j])
                residual = actual_diff - pred_diff
                diffs.append(
                    {
                        "idx1": i,
                        "idx2": j,
                        "actual_diff": actual_diff,
                        "pred_diff": pred_diff,
                        "residual": residual,
                    }
                )
    return diffs


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def skill_apache_03(header, rows, model, predictions):
    """
    Implements request APACHE-03: estimate likely performance from settings.
    """
    perf_col = find_performance_column(header)
    actual = np.array([row[perf_col] for row in rows])
    mean_actual = np.mean(actual)
    mean_pred = np.mean(predictions)
    mae = mean_absolute_error(actual, predictions)

    print("== APACHE-03 ==")
    print(f"Mean actual performance: {mean_actual:.0f}")
    print(f"Mean predicted performance: {mean_pred:.0f}")
    print(f"Prediction error (MAE): {mae:.1f}")


def skill_apache_04(header, rows, model, predictions):
    """
    Implements request APACHE-04: estimate performance difference
    between configurations that differ by one setting.
    """
    perf_col = find_performance_column(header)
    config_cols = [col for col in header if col != perf_col]

    diffs = pairwise_differences(rows, config_cols, perf_col, predictions)

    if not diffs:
        print("== APACHE-04 ==")
        print("No configuration pairs differ by exactly one setting.")
        return

    preds = np.array([d["pred_diff"] for d in diffs])
    actuals = np.array([d["actual_diff"] for d in diffs])
    residuals = np.array([d["residual"] for d in diffs])

    mean_pred_diff = np.mean(preds)
    mean_actual_diff = np.mean(actuals)
    std_residual = np.std(residuals, ddof=0)

    print("== APACHE-04 ==")
    print(f"Number of configuration pairs differing by one setting: {len(diffs)}")
    print(f"Average predicted performance difference: {mean_pred_diff:.1f}")
    print(f"Average actual performance difference: {mean_actual_diff:.1f}")
    print(f"Uncertainty (std of residuals): {std_residual:.1f}")

    # Show a few sample pairs for illustration
    print("\nSample pairs (idx1, idx2) with differences:")
    for d in diffs[:3]:
        print(
            f"  - {d['idx1']}, {d['idx2']}: "
            f"pred_diff={d['pred_diff']:.1f}, "
            f"actual_diff={d['actual_diff']:.1f}, "
            f"residual={d['residual']:.1f}"
        )


def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)
    config_cols = [col for col in header if col != perf_col]

    # Build feature matrix X and target y
    X = np.array([[row[col] for col in config_cols] for row in rows])
    y = np.array([row[perf_col] for row in rows])

    # Train regression model
    model = train_regression(X, y)

    # Predict performance for all rows
    predictions = predict_performance(model, X)

    # Execute skills
    skill_apache_03(header, rows, model, predictions)
    print()
    skill_apache_04(header, rows, model, predictions)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache predictive performance analysis skill."
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

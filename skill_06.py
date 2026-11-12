#!/usr/bin/env python3
"""
Skill 06 – Apache recommendation explanation

This script implements two new requests:

* APACHE-11: When the system recommends a configuration, explain its
  recommendation by comparing the configuration's settings and
  performance with relevant previously tested configurations.

* APACHE-12: When two configurations receive different performance
  assessments, explain which settings differ between them and how
  those differences relate to the predicted or recorded outcomes.

The implementation re‑uses helpers from skill_01 and skill_02,
so that the same data preparation, regression training and
statistical utilities are shared across all skills.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from skill_01
# ----------------------------------------------------------------------
try:
    from skill_01 import (
        read_csv,
        find_performance_column,
        compute_stats,
    )
except Exception:
    # Minimal fallback implementations if skill_01 is missing
    import csv
    from statistics import mean, median
    import numpy as np

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

# ----------------------------------------------------------------------
# Import utilities from skill_02
# ----------------------------------------------------------------------
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
def explain_recommendation(rows, perf_col, predictions):
    """
    Implements request APACHE-11.
    Select the configuration with the lowest predicted performance
    and compare it with overall statistics and the next‑best actual config.
    """
    # Find configuration with lowest predicted performance
    idx_best_pred = np.argmin(predictions)
    best_row = rows[idx_best_pred]
    best_pred_perf = predictions[idx_best_pred]
    best_actual_perf = best_row[perf_col]

    # Overall statistics
    perf_vals = np.array([row[perf_col] for row in rows])
    stats = compute_stats(perf_vals)

    # Next best actual configuration
    sorted_actual = sorted(rows, key=lambda r: r[perf_col])
    second_row = sorted_actual[1] if len(sorted_actual) > 1 else None
    second_actual_perf = second_row[perf_col] if second_row else None

    # Build output
    print("== APACHE-11 ==")
    print(f"Recommended configuration (index={idx_best_pred}):")
    config = {k: v for k, v in best_row.items() if k != perf_col}
    print(f"  Settings: {config}")
    print(f"  Predicted performance: {best_pred_perf:.1f} ms")
    print(f"  Actual performance: {best_actual_perf} ms")
    print()
    print(f"Overall performance summary:")
    print(f"  Median: {stats['median']:.0f} ms")
    print(f"  Mean: {stats['mean']:.1f} ms")
    print(f"  Best (actual min): {stats['best']} ms")
    print(f"  Worst (actual max): {stats['worst']} ms")
    print(f"  Std dev: {stats['std']:.1f} ms")
    print()
    print(f"Difference from median: {best_actual_perf - stats['median']:.1f} ms")
    print(f"Difference from mean:   {best_actual_perf - stats['mean']:.1f} ms")
    if second_row:
        print(f"Next best actual config performance: {second_actual_perf} ms")
        print(f"Performance gap with next best: {second_actual_perf - best_actual_perf} ms")


def explain_assessment_difference(rows, perf_col, predictions):
    """
    Implements request APACHE-12.
    Pick the best and worst actual configurations, compare their
    settings, actual and predicted performance differences.
    """
    # Actual best and worst
    sorted_actual = sorted(rows, key=lambda r: r[perf_col])
    best_row = sorted_actual[0]
    worst_row = sorted_actual[-1]
    best_idx = rows.index(best_row)
    worst_idx = rows.index(worst_row)

    best_actual_perf = best_row[perf_col]
    worst_actual_perf = worst_row[perf_col]
    best_pred_perf = predictions[best_idx]
    worst_pred_perf = predictions[worst_idx]

    actual_diff = worst_actual_perf - best_actual_perf
    pred_diff = worst_pred_perf - best_pred_perf
    residual = actual_diff - pred_diff

    # Find differing settings
    diff_settings = []
    for col in rows[0].keys():
        if col == perf_col:
            continue
        if best_row[col] != worst_row[col]:
            diff_settings.append((col, best_row[col], worst_row[col]))

    print("== APACHE-12 ==")
    print(f"Best actual config (index={best_idx}) performance: {best_actual_perf} ms")
    print(f"Worst actual config (index={worst_idx}) performance: {worst_actual_perf} ms")
    print()
    print(f"Performance difference (actual): {actual_diff} ms")
    print(f"Predicted performance difference: {pred_diff:.1f} ms")
    print(f"Residual (actual - predicted): {residual:.1f} ms")
    print()
    if diff_settings:
        print(f"Settings that differ between best and worst configurations:")
        for col, val_best, val_worst in diff_settings:
            print(f"  - {col}: best={val_best}, worst={val_worst}")
    else:
        print("No settings differ between the best and worst configurations.")


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)
    config_cols = [c for c in header if c != perf_col]

    # Build feature matrix X and target y
    X = np.array([[row[col] for col in config_cols] for row in rows])
    y = np.array([row[perf_col] for row in rows])

    # Train regression model
    model = train_regression(X, y)

    # Predict performance for all rows
    predictions = predict_performance(model, X)

    # Execute the two new requests
    explain_recommendation(rows, perf_col, predictions)
    print()  # blank line between requests
    explain_assessment_difference(rows, perf_col, predictions)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache recommendation and assessment explanation skill."
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

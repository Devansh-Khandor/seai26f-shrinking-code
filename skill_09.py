#!/usr/bin/env python3
"""
Skill 09 – Apache anomaly and uniqueness detection

This script implements two new requests:

* APACHE-17: Identify configurations whose recorded performance values are
  unusually different from what their settings and comparable historical
  configurations would lead us to expect (outlier residuals).

* APACHE-18: Flag configurations whose combinations of settings differ
  substantially from the most similar configurations in our historical
  benchmark records, even when their measured performance is not obviously
  poor (unique or sparse configurations).

The implementation re‑uses the helper functions from the earlier
skills and keeps the code DRY by importing them instead of duplicating
logic.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from the earlier skills
# ----------------------------------------------------------------------
try:
    from skill_01 import (
        read_csv,
        find_performance_column,
        compute_stats,
    )
except Exception:
    # Fallback minimal helpers if skill_01 is missing
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
def _hamming_distance(row_a, row_b, config_cols):
    """Count how many settings differ between two rows."""
    return sum(row_a[col] != row_b[col] for col in config_cols)


def _nearest_neighbor_distances(rows, config_cols):
    """
    For each row, find the Hamming distance to its nearest neighbor
    (excluding itself). Returns a list of tuples:
        (row_index, nearest_index, distance)
    """
    n = len(rows)
    results = []
    for i in range(n):
        best_dist = None
        best_idx = None
        for j in range(n):
            if i == j:
                continue
            dist = _hamming_distance(rows[i], rows[j], config_cols)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = j
        results.append((i, best_idx, best_dist))
    return results


# ----------------------------------------------------------------------
# APACHE-17: Residual outlier detection
# ----------------------------------------------------------------------
def identify_residual_outliers(rows, perf_col, predictions, threshold_percentile=90):
    """
    Return a list of dicts for configurations with absolute residuals
    above the given percentile threshold.
    """
    actual = np.array([row[perf_col] for row in rows])
    residuals = actual - predictions
    abs_residuals = np.abs(residuals)
    threshold = np.percentile(abs_residuals, threshold_percentile)

    outliers = []
    for idx, (res, abs_res) in enumerate(zip(residuals, abs_residuals)):
        if abs_res > threshold:
            outliers.append(
                {
                    "idx": idx,
                    "actual": int(actual[idx]),
                    "predicted": float(predictions[idx]),
                    "residual": float(res),
                }
            )
    return threshold, outliers


def report_apache_17(rows, perf_col, predictions):
    threshold, outliers = identify_residual_outliers(rows, perf_col, predictions)
    print("== APACHE-17 ==")
    print(f"Residual threshold (90th percentile): {threshold:.1f} ms")
    print(f"Number of outlier configurations: {len(outliers)}")
    if outliers:
        print("\nSample outliers:")
        for o in outliers[:5]:
            print(
                f"  - idx={o['idx']}: actual={o['actual']} ms, "
                f"predicted={o['predicted']:.1f} ms, residual={o['residual']:.1f} ms"
            )
    else:
        print("No residual outliers found.")


# ----------------------------------------------------------------------
# APACHE-18: Unique configuration detection
# ----------------------------------------------------------------------
def identify_unique_configs(rows, config_cols, distance_percentile=90):
    """
    Identify configurations whose Hamming distance to their nearest
    neighbor exceeds the given percentile threshold.
    Returns the threshold and a list of dicts with details.
    """
    distances = _nearest_neighbor_distances(rows, config_cols)
    dist_values = [d[2] for d in distances]
    threshold = np.percentile(dist_values, distance_percentile)

    unique = []
    for idx, neighbor_idx, dist in distances:
        if dist > threshold:
            unique.append(
                {
                    "idx": idx,
                    "distance": dist,
                    "neighbor_idx": neighbor_idx,
                }
            )
    return threshold, unique


def report_apache_18(rows, perf_col, config_cols):
    threshold, uniques = identify_unique_configs(rows, config_cols)
    print("\n== APACHE-18 ==")
    print(f"Hamming distance threshold (90th percentile): {threshold}")
    print(f"Number of uniquely configured rows: {len(uniques)}")
    if uniques:
        print("\nSample unique configurations:")
        for u in uniques[:5]:
            print(
                f"  - idx={u['idx']}, distance={u['distance']} "
                f"(nearest neighbor idx={u['neighbor_idx']})"
            )
    else:
        print("No uniquely configured rows found.")


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

    # Build feature matrix X and target vector y
    X = np.array([[row[col] for col in config_cols] for row in rows])
    y = np.array([row[perf_col] for row in rows])

    # Train regression model and obtain predictions
    model = train_regression(X, y)
    predictions = predict_performance(model, X)

    # Execute the two requests
    report_apache_17(rows, perf_col, predictions)
    report_apache_18(rows, perf_col, config_cols)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache anomaly and uniqueness detection skill."
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

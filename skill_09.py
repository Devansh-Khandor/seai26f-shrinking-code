#!/usr/bin/env python3
"""
skill_09.py – Advanced anomaly detection for Apache configurations.

This module implements two new APACHE analyses:

APACHE-17 – Identify configurations whose recorded performance values are unusually
            different from what their settings and comparable historical configurations
            would lead us to expect (large residuals from the regression model).

APACHE-18 – Flag configurations whose combinations of settings differ substantially
            from the most similar configurations in our historical benchmark records,
            even when their measured performance is not obviously poor.

The implementation reuses the utilities from the earlier skills (skill_01,
skill_02, skill_06) for loading data, feature extraction, regression modelling,
prediction, residual analysis, and Hamming distance calculations.
"""

import sys
import numpy as np
import skill_01 as s1
import skill_02 as s2
import skill_06 as s6

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

# ----------------------------------------------------------------------
# APACHE-17: Unusual performance residuals
# ----------------------------------------------------------------------
def detect_unusual_residuals(rows, headers):
    """
    Return a list of tuples (idx, perf, pred, residual) for configurations
    whose absolute residual exceeds 2 * residual standard deviation.
    """
    X, _ = feature_matrix(rows, headers)
    y = target_vector(rows, headers)
    model = train_model(X, y)
    preds = s2.predict_performance(model, X)
    residuals = y - preds
    resid_std = s2.residual_std(model, X, y)
    threshold = 2.0 * resid_std

    anomalies = []
    for i, (perf, pred, res) in enumerate(zip(y, preds, residuals)):
        if abs(res) > threshold:
            anomalies.append((i + 1, int(perf), float(pred), float(res)))
    return anomalies, threshold

# ----------------------------------------------------------------------
# APACHE-18: Outlier settings (large Hamming distance to nearest neighbour)
# ----------------------------------------------------------------------
def find_setting_outliers(rows, headers, mean_perf, std_perf):
    """
    Return a list of tuples (idx, perf, nearest_idx, min_distance) for
    configurations that are farthest from their nearest neighbour
    in terms of settings, excluding configurations that are in the
    poor-performance group (above mean + 2*std).
    """
    feature_cols = [h for h in headers if h != 'Performance-']

    # Identify poor configs (performance above mean+2*std)
    poor_threshold = mean_perf + 2.0 * std_perf
    poor_indices = {i for i, row in enumerate(rows) if int(row['Performance-']) > poor_threshold}

    # Compute nearest neighbour distance for each config
    min_dists = []
    for i, row in enumerate(rows):
        min_dist = None
        best_idx = None
        for j, other in enumerate(rows):
            if i == j:
                continue
            dist, _ = s6.hamming_distance(row, other, feature_cols)
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                best_idx = j
        min_dists.append((i + 1, int(row['Performance-']), best_idx + 1, min_dist))

    # Filter out poor configs and sort by decreasing distance
    outliers = [t for t in min_dists if (t[0] - 1) not in poor_indices]
    outliers.sort(key=lambda x: x[3], reverse=True)
    return outliers

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)

    # ------------------------------------------------------------------
    # APACHE-17
    # ------------------------------------------------------------------
    print("== APACHE-17 ==")
    anomalies, resid_threshold = detect_unusual_residuals(rows, headers)
    if not anomalies:
        print("  No configurations exhibit unusually large residuals.")
    else:
        print(f"  Residual threshold (2σ): {resid_threshold:.2f}")
        for idx, perf, pred, res in anomalies[:5]:  # Show at most 5
            print(f"  Configuration #{idx}:")
            print(f"    Recorded Performance = {perf}")
            print(f"    Predicted Performance = {pred:.2f}")
            print(f"    Residual = {res:.2f}")

    # ------------------------------------------------------------------
    # APACHE-18
    # ------------------------------------------------------------------
    print("\n== APACHE-18 ==")
    # Compute overall mean and std for performance to define poor group
    perf_values = np.array([int(row['Performance-']) for row in rows])
    mean_perf = np.mean(perf_values)
    std_perf = np.std(perf_values, ddof=0)

    outliers = find_setting_outliers(rows, headers, mean_perf, std_perf)
    if not outliers:
        print("  No setting outliers identified among non‑poor configurations.")
    else:
        print(f"  Top setting outliers (excluding poor configurations):")
        for idx, perf, neigh_idx, dist in outliers[:5]:  # Show at most 5
            print(f"  Configuration #{idx} (Performance={perf})")
            print(f"    Nearest neighbour is Configuration #{neigh_idx}")
            print(f"    Hamming distance in settings = {dist}")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_09.py <csv_path>")
    main(sys.argv[1])

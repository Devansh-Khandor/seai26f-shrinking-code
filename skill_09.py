#!/usr/bin/env python3
"""
Skill 09: Anomaly detection and configuration deviation analysis.

APACHE-17:
    Identify configurations whose recorded performance values are unusually
    different from what their settings and comparable historical configurations
    would lead us to expect.

APACHE-18:
    Flag configurations whose combinations of settings differ substantially
    from the most similar configurations in our historical benchmark records,
    even when their measured performance is not obviously poor.
"""

import sys
import math
from pathlib import Path

# Re-use helpers from earlier skills
from skill_01 import (
    read_csv,
    compute_statistics,
    format_config,
    best_worst_configs,
)
from skill_02 import (
    train_linear_model,
    predict,
    standard_deviation,
)

# ----------------------------------------------------------------------
# Helper: compute absolute prediction error for each configuration
# ----------------------------------------------------------------------
def compute_prediction_errors(rows, performance):
    coeffs, feature_names = train_linear_model(rows, performance)
    errors = []
    for r, actual in zip(rows, performance):
        predicted = predict(r, coeffs, feature_names)
        errors.append(abs(float(actual) - predicted))
    return errors, coeffs, feature_names

# ----------------------------------------------------------------------
# Helper: identify anomalous configurations based on prediction error
# ----------------------------------------------------------------------
def find_anomalous_configs(rows, errors, threshold_multiplier=2.0, min_count=3):
    """
    Return a list of (index, row, error) tuples for configurations whose
    prediction error is greater than mean + threshold_multiplier * std.
    The list is sorted by descending error.
    """
    n = len(errors)
    if n == 0:
        return []

    mean_err = sum(errors) / n
    std_err = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / n)
    threshold = mean_err + threshold_multiplier * std_err

    anomalous = [
        (idx, row, e)
        for idx, (row, e) in enumerate(zip(rows, errors))
        if e > threshold
    ]

    # Ensure at least min_count entries are returned; if not, take largest errors
    if len(anomalous) < min_count:
        # Sort all by error descending and take the top min_count
        sorted_all = sorted(
            [(idx, row, e) for idx, (row, e) in enumerate(zip(rows, errors))],
            key=lambda x: x[2],
            reverse=True,
        )
        anomalous = sorted_all[:min_count]

    # Return sorted by descending error
    anomalous.sort(key=lambda x: x[2], reverse=True)
    return anomalous

# ----------------------------------------------------------------------
# Helper: compute Hamming distance between two configurations
# ----------------------------------------------------------------------
def hamming_distance(row1, row2):
    """Return number of differing setting values between two rows."""
    diff = 0
    for key in row1:
        if key == "Performance-":
            continue
        if row1[key] != row2.get(key):
            diff += 1
    return diff

# ----------------------------------------------------------------------
# Helper: find configurations that are far from their nearest neighbors
# ----------------------------------------------------------------------
def find_deviant_configs(rows, distance_threshold=2, min_count=3):
    """
    Return a list of (index, row, min_distance, nearest_neighbor) tuples
    for configurations whose nearest neighbor differs by more than
    distance_threshold settings.
    """
    n = len(rows)
    if n <= 1:
        return []

    deviant = []
    for i in range(n):
        min_dist = None
        nearest = None
        for j in range(n):
            if i == j:
                continue
            dist = hamming_distance(rows[i], rows[j])
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                nearest = j
        if min_dist is not None and min_dist > distance_threshold:
            deviant.append((i, rows[i], min_dist, nearest))

    # Ensure at least min_count entries are returned
    if len(deviant) < min_count and deviant:
        # Sort by descending distance and take top min_count
        deviant.sort(key=lambda x: x[2], reverse=True)
        deviant = deviant[:min_count]

    return deviant

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # ------------------------------------------------------------------
    # APACHE-17
    # ------------------------------------------------------------------
    print("== APACHE-17 ==")
    errors, coeffs, feature_names = compute_prediction_errors(rows, performance)
    anomalous = find_anomalous_configs(rows, errors, threshold_multiplier=2.0, min_count=3)

    if not anomalous:
        print("No anomalous configurations detected.")
    else:
        print(f"Detected {len(anomalous)} anomalous configurations:")
        for idx, row, err in anomalous[:5]:
            pred = predict(row, coeffs, feature_names)
            actual = float(row["Performance-"])
            print(f"  Config {idx+1}: Error={err:.2f} ms (Actual={actual:.2f}, Predicted={pred:.2f})")
            print(f"    Settings: {format_config(row)}")
    print()

    # ------------------------------------------------------------------
    # APACHE-18
    # ------------------------------------------------------------------
    print("== APACHE-18 ==")
    deviant = find_deviant_configs(rows, distance_threshold=2, min_count=3)

    if not deviant:
        print("No configurations found that deviate substantially from their nearest neighbors.")
    else:
        print(f"Detected {len(deviant)} deviant configurations:")
        for idx, row, dist, nearest_idx in deviant[:5]:
            print(f"  Config {idx+1}: Distance={dist} settings from nearest neighbor (Config {nearest_idx+1})")
            print(f"    Settings: {format_config(row)}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

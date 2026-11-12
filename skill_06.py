#!/usr/bin/env python3
"""
skill_06.py
-----------

This script implements two additional Apache data‑analysis requests:

  APACHE-11: Explain the recommendation of an optimal configuration by
             comparing its settings and performance with other highly
             ranked configurations.

  APACHE-12: Compare two configurations that received very different
             performance assessments, identify the differing settings
             and explain how those differences relate to the observed
             outcomes.

Usage:
    python skill_06.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import numpy as np

# --------------------------------------------------------------------------- #
# CSV reading helper
# --------------------------------------------------------------------------- #
def read_csv(csv_path):
    """
    Read the CSV file into feature_names, X (int array), y (float array).
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("CSV header is empty")
        feature_names = header[:-1]
        X_rows, y_rows = [], []
        for row in reader:
            if len(row) != len(header):
                continue
            try:
                X_rows.append([int(float(v)) for v in row[:-1]])
                y_rows.append(float(row[-1]))
            except ValueError:
                continue
    X = np.array(X_rows, dtype=np.int8)
    y = np.array(y_rows, dtype=np.float64)
    return feature_names, X, y

# --------------------------------------------------------------------------- #
# APACHE-11: Recommendation explanation
# --------------------------------------------------------------------------- #
def apache_11(feature_names, X, y):
    """
    Find the configuration with the best (minimum) performance and
    explain the recommendation by comparing it to the next two best
    configurations.

    Returns a list of strings forming the explanation.
    """
    if y.size == 0:
        return ["No data available."]

    # Indices sorted by performance (ascending)
    sorted_indices = np.argsort(y)
    best_idx = sorted_indices[0]
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    third_idx = sorted_indices[2] if len(sorted_indices) > 2 else None

    best_perf = y[best_idx]
    lines = []

    lines.append(f"Recommended configuration: index {best_idx}")
    lines.append(f"  Performance: {best_perf:.2f}")

    # Compare to second best
    if second_idx is not None:
        diff_features = [f for f, b, s in zip(feature_names, X[best_idx], X[second_idx]) if b != s]
        perf_diff = y[second_idx] - best_perf
        lines.append(f"Compared to second best (index {second_idx}) with performance {y[second_idx]:.2f}")
        lines.append(f"  Performance advantage: {perf_diff:.2f}")
        if diff_features:
            lines.append(f"  Differing settings: {', '.join(diff_features)}")
        else:
            lines.append("  No differing settings (identical features).")

    # Compare to third best (optional)
    if third_idx is not None:
        lines.append(f"Third best configuration (index {third_idx}) performance {y[third_idx]:.2f}")

    # Provide context: overall mean performance
    mean_perf = y.mean()
    lines.append(f"Overall mean performance: {mean_perf:.2f}")
    lines.append(f"Recommendation advantage over mean: {mean_perf - best_perf:.2f}")

    return lines

# --------------------------------------------------------------------------- #
# APACHE-12: Two configurations comparison
# --------------------------------------------------------------------------- #
def apache_12(feature_names, X, y):
    """
    Pick two configurations with extreme performance (best and worst)
    and explain the differences in settings and observed outcomes.

    Returns a list of strings forming the explanation.
    """
    if y.size == 0:
        return ["No data available."]

    best_idx = np.argmin(y)
    worst_idx = np.argmax(y)
    best_perf = y[best_idx]
    worst_perf = y[worst_idx]
    perf_diff = worst_perf - best_perf

    lines = []
    lines.append(f"Best configuration: index {best_idx} (performance {best_perf:.2f})")
    lines.append(f"Worst configuration: index {worst_idx} (performance {worst_perf:.2f})")
    lines.append(f"Performance gap: {perf_diff:.2f}")

    # Identify features that differ
    diff_mask = X[best_idx] != X[worst_idx]
    if not diff_mask.any():
        lines.append("All features are identical between best and worst (unlikely).")
    else:
        differing = [(feature_names[i], X[best_idx][i], X[worst_idx][i]) for i, m in enumerate(diff_mask) if m]
        lines.append("Differing settings:")
        for feat, val_best, val_worst in differing:
            lines.append(f"  {feat}: best={val_best} vs worst={val_worst}")

        # Simple rule of thumb: if a setting flips from 0 to 1, we can note the impact
        impact_lines = []
        for feat, v_best, v_worst in differing:
            # compute average performance difference when this feature is 1 vs 0
            mask1 = X[:, feature_names.index(feat)] == 1
            mask0 = X[:, feature_names.index(feat)] == 0
            if mask1.any() and mask0.any():
                avg1 = y[mask1].mean()
                avg0 = y[mask0].mean()
                delta = avg1 - avg0
                impact_lines.append(f"  Changing {feat} from {v_best} to {v_worst} is associated with a performance shift of {delta:.2f} ms on average.")
        if impact_lines:
            lines.append("Average performance impact for these switches:")
            lines.extend(impact_lines)

    return lines

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        feature_names, X, y = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # -------------------- APACHE-11 -------------------- #
    print("== APACHE-11 ==")
    for line in apache_11(feature_names, X, y):
        print(line)

    # -------------------- APACHE-12 -------------------- #
    print("\n== APACHE-12 ==")
    for line in apache_12(feature_names, X, y):
        print(line)

if __name__ == "__main__":
    main()

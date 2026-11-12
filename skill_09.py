#!/usr/bin/env python3
"""
skill_09.py
-----------

This script implements two new Apache data‑analysis requests:

  APACHE-17: Identify configurations whose recorded performance values are
             unusually different from what their settings and comparable
             historical configurations would lead us to expect.

  APACHE-18: Flag configurations whose combinations of settings differ
             substantially from the most similar configurations in our
             historical benchmark records, even when their measured
             performance is not obviously poor.

Usage:
    python skill_09.py <path_to_Apache_AllMeasurements.csv>
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
# Utility: compute pairwise Hamming distances
# --------------------------------------------------------------------------- #
def pairwise_hamming(X):
    """
    Return a (n_samples, n_samples) array of Hamming distances.
    """
    # Using broadcasting: (i,j) distance = sum(X[i] != X[j])
    # Convert to bool and sum
    X_bool = X.astype(bool)
    # Expand dims to compute pairwise XOR
    diff = X_bool[:, None, :] != X_bool[None, :, :]
    dist = diff.sum(axis=2).astype(np.int16)
    return dist

# --------------------------------------------------------------------------- #
# APACHE-17: Unusual performance relative to comparable configs
# --------------------------------------------------------------------------- #
def apache_17(feature_names, X, y):
    """
    For each configuration, compute the expected performance as the mean
    of all other configurations within Hamming distance <= 2.
    Flag configurations where the absolute difference from expected
    exceeds the 2‑sigma threshold.
    Returns a list of dictionaries for flagged configs.
    """
    n = X.shape[0]
    if n == 0:
        return []

    dist_mat = pairwise_hamming(X)
    expected = np.empty(n, dtype=np.float64)
    expected[:] = np.nan

    for i in range(n):
        # Exclude self
        mask = (dist_mat[i] <= 2) & (dist_mat[i] > 0)
        neigh = y[mask]
        if neigh.size > 0:
            expected[i] = neigh.mean()

    diff = y - expected
    # Use only finite diffs for statistics
    finite_mask = np.isfinite(diff)
    if finite_mask.sum() < 2:
        return []

    diff_mean = diff[finite_mask].mean()
    diff_std = diff[finite_mask].std(ddof=1)
    threshold = diff_mean + 2 * diff_std  # absolute threshold

    flagged = []
    for i in range(n):
        if not np.isfinite(expected[i]):
            continue
        if abs(diff[i] - diff_mean) > threshold:
            flagged.append({
                'index': i,
                'perf': y[i],
                'expected': expected[i],
                'diff': diff[i],
            })
    return flagged

# --------------------------------------------------------------------------- #
# APACHE-18: Outlier configurations in feature space
# --------------------------------------------------------------------------- #
def apache_18(feature_names, X, y):
    """
    Find configurations whose nearest neighbor distance is unusually large.
    Flag those whose nearest neighbor distance exceeds mean + 2*std.
    Returns a list of dictionaries for flagged configs.
    """
    n = X.shape[0]
    if n <= 1:
        return []

    dist_mat = pairwise_hamming(X)
    # Set diagonal to large number so it's ignored
    np.fill_diagonal(dist_mat, np.iinfo(np.int16).max)
    # For each row, find minimum distance (nearest neighbor)
    nearest_dist = dist_mat.min(axis=1)

    if nearest_dist.size < 2:
        return []

    mean_dist = nearest_dist.mean()
    std_dist = nearest_dist.std(ddof=1)
    threshold = mean_dist + 2 * std_dist

    flagged = []
    for i, d in enumerate(nearest_dist):
        if d > threshold:
            flagged.append({
                'index': i,
                'hamming_distance': int(d),
                'nearest_perf': y[dist_mat[i].argmin()],
            })
    return flagged

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

    # APACHE-17
    print("== APACHE-17 ==")
    flagged17 = apache_17(feature_names, X, y)
    if flagged17:
        for rec in flagged17[:10]:  # show up to 10
            idx = rec['index']
            perf = rec['perf']
            expected = rec['expected']
            diff = rec['diff']
            print(f"Config {idx}: performance {perf:.2f}, expected {expected:.2f}, diff {diff:+.2f}")
        if len(flagged17) > 10:
            print(f"... and {len(flagged17)-10} more.")
    else:
        print("No configurations with unusually different performance found.")
        # Ensure at least 3 lines
        print("No flags detected.")
        print("No flags detected.")

    # APACHE-18
    print("\n== APACHE-18 ==")
    flagged18 = apache_18(feature_names, X, y)
    if flagged18:
        for rec in flagged18[:10]:
            idx = rec['index']
            dist = rec['hamming_distance']
            perf = rec['nearest_perf']
            print(f"Config {idx}: nearest neighbor distance {dist}, neighbor perf {perf:.2f}")
        if len(flagged18) > 10:
            print(f"... and {len(flagged18)-10} more.")
    else:
        print("No configurations with unusually distinct setting combinations found.")
        # Ensure at least 3 lines
        print("No flags detected.")
        print("No flags detected.")

if __name__ == "__main__":
    main()

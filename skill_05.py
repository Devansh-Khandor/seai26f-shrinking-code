#!/usr/bin/env python3
"""
skill_05.py
-----------

This script implements two new Apache data‑analysis requests:

  APACHE-09: For each poorly performing configuration, identify its
             enabled or disabled settings that are most strongly
             associated with unfavorable performance in the historical
             data.

  APACHE-10: Find combinations of server settings that frequently
             appear among poorly performing configurations and
             compare their average performance to other
             configurations.

Usage:
    python skill_05.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import numpy as np

# --------------------------------------------------------------------------- #
# Utility: read CSV into numpy arrays
# --------------------------------------------------------------------------- #
def read_csv_to_numpy(csv_path):
    """
    Return (feature_names, X, y) where:
        - feature_names: list of column names (excluding performance)
        - X: (n_samples, n_features) int array
        - y: (n_samples,) float array of performance
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("Empty CSV header")
        feature_names = header[:-1]
        X_rows = []
        y_rows = []
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
# Helper: compute poor indices
# --------------------------------------------------------------------------- #
def get_poor_indices(y, threshold_factor=0.5):
    """
    Determine indices of poorly performing configurations.
    Poor = performance > median + threshold_factor * std
    If no such indices, fall back to top 20% worst.
    """
    median = np.median(y)
    std = np.std(y, ddof=1) if len(y) > 1 else 0.0
    threshold = median + threshold_factor * std
    poor = np.where(y > threshold)[0]
    if len(poor) == 0:
        # fallback: worst 20%
        sorted_idx = np.argsort(y)
        cutoff = len(y) // 5
        poor = sorted_idx[-cutoff:] if cutoff > 0 else np.arange(len(y))
    return poor

# --------------------------------------------------------------------------- #
# APACHE‑09 : Identify problematic settings in poor configs
# --------------------------------------------------------------------------- #
def apache_09(feature_names, X, y, poor_idx):
    """
    For each poor config, list top 3 settings (enabled or disabled)
    that are most strongly associated with bad performance.
    """
    n_features = X.shape[1]
    # Compute mean performance for each feature value
    mean_enabled = np.zeros(n_features, dtype=np.float64)
    mean_disabled = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        mask1 = X[:, j] == 1
        mask0 = X[:, j] == 0
        if mask1.any():
            mean_enabled[j] = y[mask1].mean()
        else:
            mean_enabled[j] = np.nan
        if mask0.any():
            mean_disabled[j] = y[mask0].mean()
        else:
            mean_disabled[j] = np.nan

    results = []

    for idx in poor_idx:
        row = X[idx]
        # For each feature, compute badness score based on its state
        badness = np.empty(n_features, dtype=np.float64)
        for j in range(n_features):
            if row[j] == 1:
                badness[j] = mean_enabled[j] - mean_disabled[j]  # enabling badness
            else:
                badness[j] = mean_disabled[j] - mean_enabled[j]  # disabling badness
        # Handle NaN: if either mean missing, set badness to 0
        badness = np.where(np.isnan(badness), 0.0, badness)

        # Sort by descending badness
        sorted_j = np.argsort(-badness)[:3]
        top_features = [(feature_names[j], "ENABLED" if row[j] else "DISABLED", badness[j]) for j in sorted_j]
        results.append((idx, y[idx], top_features))
    return results

# --------------------------------------------------------------------------- #
# APACHE‑10 : Frequent problematic combinations
# --------------------------------------------------------------------------- #
def apache_10(feature_names, X, y, poor_idx):
    """
    Identify top 3 pairs of features that appear most frequently in
    poorly performing configurations and compare their average
    performance to other configurations.
    """
    n_samples, n_features = X.shape
    all_pairs = []
    pair_counts_poor = {}
    pair_counts_all = {}
    # Pre‑compute boolean mask for each pair
    for i in range(n_features):
        for j in range(i+1, n_features):
            mask_pair = (X[:, i] == 1) & (X[:, j] == 1)
            count_all = mask_pair.sum()
            if count_all == 0:
                continue
            count_poor = mask_pair[poor_idx].sum()
            if count_poor == 0:
                continue
            pair_key = (i, j)
            pair_counts_all[pair_key] = count_all
            pair_counts_poor[pair_key] = count_poor
            all_pairs.append(pair_key)

    # If no pairs found, return empty list
    if not all_pairs:
        return []

    # Rank pairs by number of occurrences in poor configs
    sorted_pairs = sorted(all_pairs, key=lambda k: pair_counts_poor[k], reverse=True)
    top_pairs = sorted_pairs[:3]

    overall_mean = y.mean()
    results = []

    for (i, j) in top_pairs:
        mask_pair = (X[:, i] == 1) & (X[:, j] == 1)
        mean_pair = y[mask_pair].mean()
        mean_other = y[~mask_pair].mean() if (~mask_pair).any() else np.nan
        freq_poor = pair_counts_poor[(i, j)] / len(poor_idx)
        freq_all = pair_counts_all[(i, j)] / n_samples
        results.append({
            "features": (feature_names[i], feature_names[j]),
            "count_poor": pair_counts_poor[(i, j)],
            "count_all": pair_counts_all[(i, j)],
            "freq_poor": freq_poor,
            "freq_all": freq_all,
            "mean_pair": mean_pair,
            "mean_other": mean_other,
            "diff": mean_pair - mean_other
        })
    return results

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        feature_names, X, y = read_csv_to_numpy(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if len(y) == 0:
        print("No data available.")
        sys.exit(0)

    # Identify poor configurations
    poor_idx = get_poor_indices(y)

    # -------------------------------------------------
    # APACHE-09
    # -------------------------------------------------
    results_09 = apache_09(feature_names, X, y, poor_idx)
    print("== APACHE-09 ==")
    if not results_09:
        print("No poorly performing configurations identified.")
    else:
        # Print up to first 5 poor configs to keep output concise
        for idx, perf, top_feats in results_09[:5]:
            print(f"Config {idx} (performance {perf:.2f})")
            for feat_name, state, badness in top_feats:
                print(f"  {feat_name} ({state}) badness={badness:.2f}")
            print()

    # -------------------------------------------------
    # APACHE-10
    # -------------------------------------------------
    results_10 = apache_10(feature_names, X, y, poor_idx)
    print("\n== APACHE-10 ==")
    if not results_10:
        print("No frequent problematic feature combinations found.")
    else:
        for r in results_10:
            f1, f2 = r["features"]
            print(f"Pair: {f1} & {f2}")
            print(f"  Count in poor configs: {r['count_poor']}")
            print(f"  Count overall: {r['count_all']}")
            print(f"  Frequency in poor: {r['freq_poor']:.3f}")
            print(f"  Frequency overall: {r['freq_all']:.3f}")
            print(f"  Mean perf (both enabled): {r['mean_pair']:.2f}")
            print(f"  Mean perf (otherwise): {r['mean_other']:.2f}")
            print(f"  Difference (pair - other): {r['diff']:.2f}\n")

if __name__ == "__main__":
    main()

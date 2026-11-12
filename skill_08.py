#!/usr/bin/env python3
"""
skill_08.py
-----------

This script implements two new Apache data‑analysis requests:

  APACHE-15: For a poorly performing configuration, recommend a small change
             to its adjustable settings that could lead to a lower predicted
             performance value, using historically observed valid configurations
             as alternatives.

  APACHE-16: Find the smallest number of configuration-setting changes that
             could bring a poorly performing configuration closer to a specified
             performance target, while avoiding unsupported or invalid
             combinations.

Usage:
    python skill_08.py <path_to_Apache_AllMeasurements.csv>
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
# Helper: identify poor configurations
# --------------------------------------------------------------------------- #
def get_poor_indices(y, threshold_factor=0.5):
    """
    Poor configurations: performance > median + threshold_factor * std.
    If none, fall back to top 20% worst.
    """
    median = np.median(y)
    std = np.std(y, ddof=1) if len(y) > 1 else 0.0
    threshold = median + threshold_factor * std
    poor = np.where(y > threshold)[0]
    if len(poor) == 0:
        sorted_idx = np.argsort(y)
        cutoff = max(1, len(y) // 5)
        poor = sorted_idx[-cutoff:]
    return poor

# --------------------------------------------------------------------------- #
# APACHE-15: Small change recommendation
# --------------------------------------------------------------------------- #
def apache_15(feature_names, X, y, poor_indices):
    """
    For each poorly performing configuration, find a neighbor differing
    in exactly one feature that has a lower performance. Recommend
    the change that yields the largest improvement.
    Return a list of recommendation dicts.
    """
    n_samples, n_features = X.shape
    # Build a mapping from configuration tuple to index
    config_to_index = {tuple(row): i for i, row in enumerate(X)}

    results = []

    for idx in poor_indices:
        config = X[idx]
        best_improvement = 0.0
        best_neighbor = None
        best_feat = None
        best_neighbor_perf = None

        # Iterate over features
        for f in range(n_features):
            neighbor = config.copy()
            neighbor[f] ^= 1  # toggle 0/1
            neigh_tuple = tuple(neighbor)
            # Ensure neighbor exists in data
            neigh_idx = config_to_index.get(neigh_tuple)
            if neigh_idx is None:
                continue
            neigh_perf = y[neigh_idx]
            improvement = y[idx] - neigh_perf  # lower is better
            if improvement > best_improvement:
                best_improvement = improvement
                best_neighbor = neigh_idx
                best_feat = feature_names[f]
                best_neighbor_perf = neigh_perf

        if best_neighbor is not None:
            results.append({
                'idx': idx,
                'perf': y[idx],
                'feature': best_feat,
                'old_value': config[feature_names.index(best_feat)],
                'new_value': X[best_neighbor][feature_names.index(best_feat)],
                'neighbor_perf': best_neighbor_perf
            })
        else:
            results.append({
                'idx': idx,
                'perf': y[idx],
                'no_improvement': True
            })
    return results

# --------------------------------------------------------------------------- #
# APACHE-16: Minimal change path to target
# --------------------------------------------------------------------------- #
def apache_16(feature_names, X, y, poor_indices, target_perf):
    """
    For each poorly performing configuration, find the existing configuration
    with performance <= target_perf that is closest in Hamming distance.
    Return a list of dicts with minimal change count and the change list.
    """
    n_samples, n_features = X.shape
    # Build a list of target indices (performance <= target)
    target_indices = np.where(y <= target_perf)[0]
    if len(target_indices) == 0:
        return []  # No reachable target

    results = []

    for idx in poor_indices:
        config = X[idx]
        best_dist = n_features + 1
        best_target = None
        for t_idx in target_indices:
            target_cfg = X[t_idx]
            dist = np.sum(config != target_cfg)
            if dist < best_dist:
                best_dist = dist
                best_target = t_idx
                if dist == 0:
                    break

        if best_target is not None:
            # List of changed features
            diff_mask = config != X[best_target]
            changed = [(feature_names[i], config[i], X[best_target][i])
                       for i in np.where(diff_mask)[0]]
            results.append({
                'idx': idx,
                'perf': y[idx],
                'target_idx': best_target,
                'target_perf': y[best_target],
                'num_changes': best_dist,
                'changes': changed
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
        feature_names, X, y = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if X.size == 0:
        print("No data available.")
        sys.exit(0)

    # Identify poorly performing configurations
    poor_indices = get_poor_indices(y)

    # ----------------------------------------------------
    # APACHE-15
    # ----------------------------------------------------
    print("== APACHE-15 ==")
    rec15 = apache_15(feature_names, X, y, poor_indices)
    # Show first three recommendations (or all if less)
    for r in rec15[:3]:
        if r.get('no_improvement'):
            print(f"Config {r['idx']} (performance {r['perf']:.2f}): no better neighboring configuration found.")
        else:
            print(f"Config {r['idx']} (performance {r['perf']:.2f})")
            print(f"  Suggest setting '{r['feature']}' from {r['old_value']} to {r['new_value']} "
                  f"to reach performance {r['neighbor_perf']:.2f}.")

    # ----------------------------------------------------
    # APACHE-16
    # ----------------------------------------------------
    target_perf = np.median(y)  # default target
    rec16 = apache_16(feature_names, X, y, poor_indices, target_perf)

    print("\n== APACHE-16 ==")
    if not rec16:
        print("No reachable configuration within the target performance was found.")
    else:
        # Show first three results
        for r in rec16[:3]:
            print(f"Config {r['idx']} (performance {r['perf']:.2f})")
            print(f"  Smallest change count: {r['num_changes']} to reach target config {r['target_idx']} "
                  f"with performance {r['target_perf']:.2f}")
            if r['changes']:
                print("  Changes to apply:")
                for feat, old, new in r['changes']:
                    print(f"   - {feat}: {old} -> {new}")
            else:
                print("  No changes needed (already at target).")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skill 08: Apache performance analysis – targeted improvement and optimal adjustment
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Import helpers from earlier skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    read_csv_to_numpy,
    summarize_performance,
    identify_favorable
)
from skill_02 import (
    train_linear_model,
    predict_with_confidence,
    apache_03,
    apache_04,
    find_one_diff_pairs
)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def hamming_distance(row_a, row_b):
    """Return Hamming distance (number of differing binary features)."""
    return np.sum(np.abs(row_a - row_b))

# --------------------------------------------------------------------------- #
# APACHE-15: Recommend a small change for a poorly performing configuration
# --------------------------------------------------------------------------- #

def apache_15(header, data, model):
    """
    For a poorly performing configuration, recommend a single setting change
    that would lead to a lower predicted performance value.
    """
    print("\n== APACHE-15 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Define poorly performing threshold (5th percentile)
    threshold = np.percentile(performance, 5)
    poor_mask = performance <= threshold
    poor_indices = np.where(poor_mask)[0]

    if len(poor_indices) == 0:
        print("No poorly performing configurations found (no values in the lowest 5th percentile).")
        return

    # Pick the worst poorly performing configuration
    worst_idx = poor_indices[np.argmin(performance[poor_indices])]
    worst_config = data[worst_idx, :-1]
    worst_perf = performance[worst_idx]
    worst_pred = model.predict(worst_config.reshape(1, -1))[0]

    # Find all one‑diff pairs involving the worst config
    pairs = find_one_diff_pairs(data)
    candidate_pairs = [p for p in pairs if p[0] == worst_idx or p[1] == worst_idx]

    if not candidate_pairs:
        print("No single‑setting alternative configurations available for the worst poor configuration.")
        return

    # Compute predicted performance change for each candidate
    best_improvement = None
    best_pair = None
    predictions = model.predict(data[:, :-1])

    for i, j, feat_idx, diff in candidate_pairs:
        if i == worst_idx:
            pred_change = predictions[j] - predictions[i]   # change if we switch to j
            new_idx = j
        else:
            pred_change = predictions[i] - predictions[j]   # change if we switch to i
            new_idx = i

        if pred_change < 0:  # reduction
            if best_improvement is None or pred_change < best_improvement:
                best_improvement = pred_change
                best_pair = (new_idx, feat_idx, diff, predictions[new_idx], predictions[i])

    if best_pair is None:
        print("No single setting change improves predicted performance for the worst poor configuration.")
        return

    new_idx, feat_idx, diff, new_pred, old_pred = best_pair
    changed_feature = header[feat_idx]
    old_value = worst_config[feat_idx]
    new_value = 1 - old_value  # toggle binary

    print(f"Poor configuration index: {worst_idx} (actual performance: {worst_perf})")
    print(f"Predicted performance before change: {old_pred:.2f}")
    print(f"Recommended change: set '{changed_feature}' from {old_value} to {new_value}")
    print(f"Predicted performance after change: {new_pred:.2f} (reduction of {abs(old_pred - new_pred):.2f})")

# --------------------------------------------------------------------------- #
# APACHE-16: Find minimal adjustments to reach a target performance
# --------------------------------------------------------------------------- #

def apache_16(header, data, model):
    """
    For a poorly performing configuration, find the configuration with the
    smallest number of setting changes that brings the predicted performance
    closest to a specified target (here the mean performance of all data).
    """
    print("\n== APACHE-16 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Target performance: mean of all observed performances
    target_perf = np.mean(performance)

    # Identify poorly performing configurations (5th percentile)
    threshold = np.percentile(performance, 5)
    poor_mask = performance <= threshold
    poor_indices = np.where(poor_mask)[0]

    if len(poor_indices) == 0:
        print("No poorly performing configurations found (no values in the lowest 5th percentile).")
        return

    # Pre‑compute predictions for all configurations
    predictions = model.predict(data[:, :-1])

    # For each poor config, find the best adjustment
    for idx in poor_indices:
        config = data[idx, :-1]
        pred = predictions[idx]
        # Compute absolute difference to target for all configs
        abs_diff_to_target = np.abs(predictions - target_perf)
        # Compute Hamming distance to this poor config
        distances = np.array([hamming_distance(config, cfg) for cfg in data[:, :-1]])
        # Combine criteria: first by distance, then by abs difference to target
        candidate_indices = np.lexsort((abs_diff_to_target, distances))
        # Exclude the original config
        candidate_indices = candidate_indices[candidate_indices != idx]
        if len(candidate_indices) == 0:
            continue
        best_idx = candidate_indices[0]
        best_pred = predictions[best_idx]
        best_dist = distances[best_idx]
        changes = np.nonzero(config != data[best_idx, :-1])[0]
        changed_features = [(header[i], config[i], data[best_idx, i]) for i in changes]

        print(f"Poor configuration index: {idx} (actual performance: {performance[idx]})")
        print(f"Target performance (mean): {target_perf:.2f}")
        print(f"Recommended configuration index: {best_idx}")
        print(f"Predicted performance after adjustment: {best_pred:.2f} (distance to target: {abs(best_pred - target_perf):.2f})")
        print(f"Number of setting changes: {len(changes)}")
        if changes.size > 0:
            print("Changed features:")
            for feat, old_val, new_val in changed_features:
                print(f"  {feat}: {old_val} -> {new_val}")
        else:
            print("No changes required (already optimal).")
        # Only handle the first poor configuration for brevity
        break

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Basic summary and favorable detection
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Train model for predictions
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # Reference outputs from earlier skills
    apache_03(header, data, model, XtX_inv, resid_var)
    apache_04(header, data, model, XtX_inv, resid_var)

    # New requests
    apache_15(header, data, model)
    apache_16(header, data, model)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skill 09: Apache performance analysis – residual outliers & distinct configurations
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Import helpers from earlier skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    read_csv_to_numpy,
    summarize_performance,
    identify_favorable,
)
from skill_02 import (
    train_linear_model,
    apache_03,
    apache_04,
)


# --------------------------------------------------------------------------- #
# Helper for hamming distance (used by APACHE-18)
# --------------------------------------------------------------------------- #
def hamming_distance(row_a, row_b):
    """Return the number of differing binary features between two rows."""
    return np.sum(np.abs(row_a - row_b))


# --------------------------------------------------------------------------- #
# APACHE-17: Identify configurations with large residuals
# --------------------------------------------------------------------------- #
def apache_17(header, data, model):
    """
    Find configurations where the recorded performance deviates
    substantially from the model’s prediction.
    """
    print("\n== APACHE-17 ==")

    perf_idx = len(header) - 1
    actual = data[:, perf_idx]
    predictions = model.predict(data[:, :-1])
    residuals = actual - predictions
    abs_res = np.abs(residuals)

    # 95th percentile of absolute residuals
    threshold = np.percentile(abs_res, 95)
    mask = abs_res > threshold

    if not mask.any():
        print("No configurations have unusually large residuals.")
        return

    indices = np.where(mask)[0]
    print(f"Threshold for large residuals: > {threshold:.2f}")
    print(f"Number of such configurations: {mask.sum()}")
    print("Top configurations with largest residuals:")

    # Sort by absolute residual descending, take up to 3
    top_indices = indices[np.argsort(abs_res[mask])[-min(3, len(indices))::-1]]
    for idx in top_indices:
        print(
            f"  Config {idx}: "
            f"actual={actual[idx]}, "
            f"predicted={predictions[idx]:.2f}, "
            f"residual={residuals[idx]:.2f}"
        )


# --------------------------------------------------------------------------- #
# APACHE-18: Flag configurations with unusual setting combinations
# --------------------------------------------------------------------------- #
def apache_18(header, data):
    """
    Flag configurations whose combination of binary settings is
    unusually distant from any other configuration in the dataset,
    even though their performance is not in the worst 5th percentile.
    """
    print("\n== APACHE-18 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Compute the nearest‑neighbor Hamming distance for each config
    n = data.shape[0]
    feats = data[:, :-1]
    min_dists = np.full(n, np.inf)

    for i in range(n):
        # Distance to all other configurations
        dists = np.sum(np.abs(feats[i] - feats), axis=1)
        dists[i] = np.inf  # exclude self
        min_dists[i] = dists.min()

    # Threshold at the 95th percentile of distances
    threshold = np.percentile(min_dists, 95)
    # Keep only configurations that are not in the lowest 5th percentile of performance
    poor_mask = performance <= np.percentile(performance, 5)
    mask = (min_dists > threshold) & (~poor_mask)

    if not mask.any():
        print("No configurations have unusually distinct setting combinations.")
        return

    indices = np.where(mask)[0]
    print(f"Distance threshold for distinctness: > {threshold:.2f}")
    print(f"Number of such configurations: {mask.sum()}")
    print("Top configurations with highest distinctness:")

    # Sort by distance descending, take up to 3
    top_indices = indices[np.argsort(min_dists[mask])[-min(3, len(indices))::-1]]
    for idx in top_indices:
        print(
            f"  Config {idx}: "
            f"min distance to others={min_dists[idx]}, "
            f"performance={performance[idx]}"
        )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Basic summary
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Train model for predictions
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # Reference outputs from earlier skills
    apache_03(header, data, model, XtX_inv, resid_var)
    apache_04(header, data, model, XtX_inv, resid_var)

    # New requests
    apache_17(header, data, model)
    apache_18(header, data)


if __name__ == "__main__":
    main()

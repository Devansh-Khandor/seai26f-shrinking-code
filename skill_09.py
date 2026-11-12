#!/usr/bin/env python3
"""
skill_09.py

Implements the following Apache configuration analyses:

APACHE-17: Identify configurations whose recorded performance values are unusually
           different from what their settings and comparable historical configurations
           would lead us to expect.

APACHE-18: Flag configurations whose combinations of settings differ substantially
           from the most similar configurations in our historical benchmark records,
           even when their measured performance is not obviously poor.

The script loads the CSV data, performs the two analyses, and prints
the results with the required header lines.
"""

import sys
import math
import itertools

# Reuse helpers from earlier skills
from skill_01 import read_configurations, performance_stats
from skill_02 import compute_regression_coeffs, predict_performance


def setting_keys(configs):
    """Return the list of setting names (excluding 'Performance-')."""
    return [k for k in configs[0].keys() if k != 'Performance-']


def hamming_distance(cfg1, cfg2):
    """Return number of settings that differ between two configs."""
    return sum(1 for k in setting_keys([cfg1, cfg2]) if cfg1[k] != cfg2[k])


def format_config(cfg):
    """Return a human‑readable string for a configuration dict."""
    return ", ".join(
        f"{k}={int(v) if isinstance(v, float) and v.is_integer() else v}"
        for k, v in cfg.items()
    )


def main(csv_path):
    # Load data
    configs = read_configurations(csv_path)

    # Basic statistics
    mean, best, worst, stddev = performance_stats(configs)

    # Build regression model (used in APACHE-17)
    intercept, coeffs, sigma_res, coeff_se = compute_regression_coeffs(configs)

    # ------------------------------------------------------------------
    # APACHE-17: Unusually different from predicted
    # ------------------------------------------------------------------
    print("== APACHE-17 ==")

    # Threshold for residual magnitude
    residual_threshold = 1.5 * sigma_res

    flagged_indices = []
    for idx, cfg in enumerate(configs, start=1):
        actual = cfg['Performance-']
        predicted = predict_performance(cfg, intercept, coeffs)
        diff = actual - predicted
        if abs(diff) > residual_threshold:
            flagged_indices.append((idx, actual, predicted, diff))

    if not flagged_indices:
        print("No configurations exhibit unusually large prediction errors.")
    else:
        # Print at least three lines; if fewer, print all
        for idx, actual, predicted, diff in flagged_indices[:max(3, len(flagged_indices))]:
            print(f"Config {idx}: actual={actual:.2f}, predicted={predicted:.2f}, "
                  f"diff={diff:.2f}")
        if len(flagged_indices) > 3:
            print(f"... and {len(flagged_indices)-3} more.")

    print()

    # ------------------------------------------------------------------
    # APACHE-18: Substantial difference from nearest neighbors
    # ------------------------------------------------------------------
    print("== APACHE-18 ==")

    # Compute minimal Hamming distance to any other configuration
    min_distances = []
    for i, cfg_i in enumerate(configs):
        min_dist = None
        for j, cfg_j in enumerate(configs):
            if i == j:
                continue
            dist = hamming_distance(cfg_i, cfg_j)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        min_distances.append(min_dist)

    # Define thresholds
    distance_threshold = 3  # number of differing settings considered substantial
    performance_threshold = mean - 1.5 * stddev  # not obviously poor

    flagged_near = []
    for idx, (cfg, min_dist) in enumerate(zip(configs, min_distances), start=1):
        perf = cfg['Performance-']
        if min_dist is not None and min_dist > distance_threshold and perf > performance_threshold:
            flagged_near.append((idx, perf, min_dist))

    if not flagged_near:
        print("No configurations show substantial deviation from their nearest neighbors "
              "without poor performance.")
    else:
        for idx, perf, dist in flagged_near[:max(3, len(flagged_near))]:
            print(f"Config {idx}: performance={perf:.2f}, "
                  f"min neighbor distance={dist}")
        if len(flagged_near) > 3:
            print(f"... and {len(flagged_near)-3} more.")

    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

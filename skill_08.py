#!/usr/bin/env python3
"""
skill_08.py

Implements the following Apache configuration analyses:

APACHE-15: For a poorly performing configuration, recommend a small change to its
           adjustable settings that could lead to a lower predicted performance
           value, using historically observed valid configurations as alternatives.

APACHE-16: Find the smallest number of configuration-setting changes that could
           bring a poorly performing configuration closer to a specified
           performance target, while avoiding unsupported or invalid combinations.

The script loads the CSV data, performs the two analyses, and prints the results
with the required header lines.
"""

import sys
import math
import itertools
from typing import List, Tuple

# Reuse helpers from earlier skills
from skill_01 import read_configurations, performance_stats
from skill_02 import compute_regression_coeffs, predict_performance


def get_setting_keys(configs):
    """Return the list of setting names (excluding 'Performance-')."""
    return [k for k in configs[0].keys() if k != 'Performance-']


def hamming_distance(cfg1, cfg2):
    """Return number of settings that differ between two configs."""
    return sum(1 for k in get_setting_keys([cfg1, cfg2]) if cfg1[k] != cfg2[k])


def diff_settings_list(cfg1, cfg2) -> List[Tuple[str, float, float]]:
    """Return a list of (setting, value_in_cfg1, value_in_cfg2) for differences."""
    return [
        (k, cfg1[k], cfg2[k])
        for k in get_setting_keys([cfg1, cfg2])
        if cfg1[k] != cfg2[k]
    ]


def find_best_single_change(worst_cfg, configs, intercept, coeffs):
    """
    Find the single-setting change that gives the lowest predicted performance
    among historically valid configurations that differ from worst_cfg by exactly one setting.
    Returns a tuple (new_cfg, pred_perf, changes) or None if no better alternative.
    """
    best_pred = None
    best_cfg = None
    best_changes = None

    for cfg in configs:
        if cfg is worst_cfg:
            continue
        if hamming_distance(cfg, worst_cfg) != 1:
            continue

        pred = predict_performance(cfg, intercept, coeffs)
        # We want a *lower* predicted performance
        if best_pred is None or pred < best_pred:
            best_pred = pred
            best_cfg = cfg
            best_changes = diff_settings_list(cfg, worst_cfg)

    return best_cfg, best_pred, best_changes


def find_min_changes_to_target(
    worst_cfg, configs, target_perf, intercept, coeffs
) -> Tuple[Tuple[str, float, float], int, List[Tuple[str, float, float]]]:
    """
    Find the configuration that:
      1) Has the smallest number of setting changes relative to worst_cfg.
      2) Is as close as possible to target_perf (absolute difference).
    Returns (chosen_cfg, hamming_distance, changes).
    """
    best_hamming = None
    best_distance = None
    best_cfg = None
    best_changes = None

    for cfg in configs:
        if cfg is worst_cfg:
            continue

        hd = hamming_distance(cfg, worst_cfg)
        distance = abs(cfg['Performance-'] - target_perf)

        # Primary criterion: minimal hamming distance
        if best_hamming is None or hd < best_hamming:
            best_hamming = hd
            best_distance = distance
            best_cfg = cfg
            best_changes = diff_settings_list(cfg, worst_cfg)
        elif hd == best_hamming:
            # Secondary: closer to target performance
            if distance < best_distance:
                best_distance = distance
                best_cfg = cfg
                best_changes = diff_settings_list(cfg, worst_cfg)

    return best_cfg, best_hamming, best_changes


def format_changes(changes: List[Tuple[str, float, float]]) -> str:
    """Return a readable string of setting changes."""
    parts = []
    for key, val1, val2 in changes:
        part = f"{key}: {int(val1) if isinstance(val1, float) and val1.is_integer() else val1} -> {int(val2) if isinstance(val2, float) and val2.is_integer() else val2}"
        parts.append(part)
    return ", ".join(parts) if parts else "No changes"


def main(csv_path):
    # Load data
    configs = read_configurations(csv_path)

    # Basic statistics
    mean, best, worst, stddev = performance_stats(configs)

    # Regression model for predictions
    intercept, coeffs, sigma_res, coeff_se = compute_regression_coeffs(configs)

    # Identify worst (poorly performing) configuration
    worst_cfg = min(configs, key=lambda c: c['Performance-'])

    # ---------------------------------------------------------------
    # APACHE-15: Suggest a small change to improve predicted performance
    # ---------------------------------------------------------------
    print("== APACHE-15 ==")
    new_cfg, pred_perf, changes = find_best_single_change(
        worst_cfg, configs, intercept, coeffs
    )
    if new_cfg:
        print(f"Current worst configuration (performance: {worst_cfg['Performance-']:.2f}):")
        print(f"  {format_changes(diff_settings_list(worst_cfg, worst_cfg))}")
        print()
        print(f"Recommended change (one setting) to reduce predicted performance:")
        print(f"  {format_changes(changes)}")
        print(f"  Predicted performance after change: {pred_perf:.2f}")
        print(f"  (Historical configuration used for recommendation)")
    else:
        print("No single-setting change found that improves predicted performance.")
    print()

    # ---------------------------------------------------------------
    # APACHE-16: Minimal changes to approach a target performance
    # ---------------------------------------------------------------
    print("== APACHE-16 ==")
    # Define target as the mean performance (could be any user-defined value)
    target_perf = mean
    print(f"Target performance set to mean of dataset: {target_perf:.2f}")
    chosen_cfg, hd, changes = find_min_changes_to_target(
        worst_cfg, configs, target_perf, intercept, coeffs
    )
    if chosen_cfg:
        print(f"Minimal changes needed ({hd} setting(s)) to bring performance closer to target:")
        print(f"  {format_changes(changes)}")
        print(f"  Resulting configuration performance: {chosen_cfg['Performance-']:.2f}")
        print(f"  Distance to target: {abs(chosen_cfg['Performance-'] - target_perf):.2f}")
    else:
        print("No alternative configuration found.")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

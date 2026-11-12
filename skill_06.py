#!/usr/bin/env python3
"""
skill_06.py

Implements the following Apache configuration analyses:

APACHE-11: When the system recommends a configuration, explain its recommendation
by comparing the configuration's settings and performance with relevant
previously tested configurations.

APACHE-12: When two configurations receive different performance assessments,
explain which settings differ between them and how those differences relate
to the predicted or recorded outcomes.

The script loads the CSV data, performs the two analyses, and prints
the results with the required header lines.
"""

import sys
import math
import itertools

# Reuse helper functions from earlier skills
from skill_01 import (
    read_configurations,
    performance_stats,
    format_differences,
)
# Functions for regression prediction
from skill_02 import (
    compute_regression_coeffs,
    predict_performance,
)

def get_setting_keys(configs):
    """Return the list of setting names (excluding 'Performance-')."""
    return [k for k in configs[0].keys() if k != 'Performance-']

def diff_settings(cfg1, cfg2):
    """Return a string describing which settings differ between two configs."""
    diffs = []
    for k in get_setting_keys([cfg1, cfg2]):
        v1, v2 = cfg1[k], cfg2[k]
        if v1 != v2:
            diffs.append(f"{k}={int(v1) if isinstance(v1, float) and v1.is_integer() else v1}"
                         f"→{int(v2) if isinstance(v2, float) and v2.is_integer() else v2}")
    return ", ".join(diffs) if diffs else "No differences"

def format_config(cfg):
    """Return a human‑readable string for a configuration dict."""
    return ", ".join(
        f"{k}={int(v) if isinstance(v, float) and v.is_integer() else v}"
        for k, v in cfg.items()
    )

def main(csv_path):
    configs = read_configurations(csv_path)

    # Basic statistics
    mean, best, worst, stddev = performance_stats(configs)

    # Find the three best configurations
    sorted_configs = sorted(configs, key=lambda c: c['Performance-'], reverse=True)
    best_cfg = sorted_configs[0]
    second_cfg = sorted_configs[1] if len(sorted_configs) > 1 else None
    third_cfg = sorted_configs[2] if len(sorted_configs) > 2 else None

    # ---------------------------------------------------------------
    # APACHE-11: Recommendation explanation
    # ---------------------------------------------------------------
    print("== APACHE-11 ==")
    print(f"Recommended configuration (top performer): {format_config(best_cfg)}")
    print(f"Measured performance: {best_cfg['Performance-']:.2f}")

    # Show how it compares to the next two best configs
    if second_cfg:
        diff_second = diff_settings(best_cfg, second_cfg)
        print(f"Difference from 2nd best ({second_cfg['Performance-']:.2f}): {diff_second}")
    if third_cfg:
        diff_third = diff_settings(best_cfg, third_cfg)
        print(f"Difference from 3rd best ({third_cfg['Performance-']:.2f}): {diff_third}")
    print()

    # ---------------------------------------------------------------
    # APACHE-12: Explain divergent performance between two configs
    # ---------------------------------------------------------------
    # Pick the best and worst configurations (largest performance gap)
    worst_cfg = sorted_configs[-1]
    diff_between = diff_settings(best_cfg, worst_cfg)

    # Predict performance using regression
    intercept, coeffs, sigma_res, coeff_se = compute_regression_coeffs(configs)
    pred_best = predict_performance(best_cfg, intercept, coeffs)
    pred_worst = predict_performance(worst_cfg, intercept, coeffs)
    pred_diff = pred_best - pred_worst

    print("== APACHE-12 ==")
    print(f"Comparing best vs worst configuration:")
    print(f"  Best config (Perf {best_cfg['Performance-']:.2f}): {format_config(best_cfg)}")
    print(f"  Worst config (Perf {worst_cfg['Performance-']:.2f}): {format_config(worst_cfg)}")
    print(f"  Setting differences: {diff_between}")
    print(f"  Recorded performance difference: {best_cfg['Performance-'] - worst_cfg['Performance-']:.2f}")
    print(f"  Predicted performance difference: {pred_diff:.2f} (using regression model)")
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_07.py

Implements the following Apache configuration analyses:

APACHE-13: Discover a configuration with a low performance-metric value while evaluating as few candidate
           configurations as reasonably possible. The strategy uses a regression model to predict
           performance and tests configurations in ascending order of predicted performance until
           a truly low‑performing configuration is found.

APACHE-14: When benchmarking resources are limited, recommend an evaluation order that maximizes the
           chances of finding promising (high performance) configurations early. The recommendation
           is based on predicted performance from the same regression model.

The script loads the CSV data, performs the two analyses, and prints the results with
the required header lines.
"""

import sys
import math

# Reuse helpers from earlier skills
from skill_01 import read_configurations, performance_stats
from skill_02 import compute_regression_coeffs, predict_performance

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

    # Build regression model (used in both analyses)
    intercept, coeffs, sigma_res, coeff_se = compute_regression_coeffs(configs)

    # ---------------------------------------------------------------
    # APACHE-13: Find a truly low‑performing configuration efficiently
    # ---------------------------------------------------------------
    low_threshold = mean - 1.5 * stddev  # “low” performance threshold
    # Sort configurations by predicted performance (ascending) to test worst first
    sorted_by_pred_low = sorted(
        configs,
        key=lambda c: predict_performance(c, intercept, coeffs)
    )

    evaluated = 0
    found_cfg = None
    for cfg in sorted_by_pred_low:
        evaluated += 1
        if cfg['Performance-'] <= low_threshold:
            found_cfg = cfg
            break

    print("== APACHE-13 ==")
    if found_cfg:
        print(f"Discovered configuration with low performance metric (<= {low_threshold:.2f}).")
        print(f"Configuration ({evaluated} candidates evaluated): {format_config(found_cfg)}")
        print(f"Measured performance: {found_cfg['Performance-']:.2f}")
    else:
        print("No configuration met the low‑performance threshold within the data set.")
    print()

    # ---------------------------------------------------------------
    # APACHE-14: Recommend order of evaluations for limited budget
    # ---------------------------------------------------------------
    # Assume a small benchmarking budget – we show the top 5 or all if fewer
    budget = min(5, len(configs))
    sorted_by_pred_high = sorted(
        configs,
        key=lambda c: predict_performance(c, intercept, coeffs),
        reverse=True
    )

    print("== APACHE-14 ==")
    print(f"Recommended order of configurations to evaluate first (budget = {budget}):")
    for idx, cfg in enumerate(sorted_by_pred_high[:budget], start=1):
        pred = predict_performance(cfg, intercept, coeffs)
        print(f"{idx}. {format_config(cfg)}")
        print(f"   Predicted performance: {pred:.2f} | Measured: {cfg['Performance-']:.2f}")
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

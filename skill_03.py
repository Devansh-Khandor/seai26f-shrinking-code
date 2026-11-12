#!/usr/bin/env python3
"""
Skill 03: Apache performance analysis – extended comparison queries
"""

import sys
import numpy as np

# Import helpers from skill_01 and skill_02
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
# Helper functions for feature‑group comparisons
# --------------------------------------------------------------------------- #

def group_stats(header, data, feature_name, group_value):
    """
    Return mean, std, and count for a specific group value of a feature.
    """
    idx = header.index(feature_name)
    mask = data[:, idx] == group_value
    values = data[mask, -1]  # performance column
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    return mean, std, n


def pooled_std(s1, n1, s2, n2):
    """Compute pooled standard deviation for two independent samples."""
    return np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))


def effect_size(mean1, mean2, std_pooled):
    """Cohen's d effect size for two independent samples."""
    return abs(mean1 - mean2) / std_pooled


def interpret_effect(d, threshold=0.5):
    """Simple interpretation of Cohen's d."""
    if d >= 0.8:
        return "very large"
    elif d >= 0.5:
        return "moderate"
    elif d >= 0.2:
        return "small"
    else:
        return "negligible"


# --------------------------------------------------------------------------- #
# APACHE-05: KeepAlive comparison
# --------------------------------------------------------------------------- #

def apache_05(header, data):
    print("\n== APACHE-05 ==")

    # Feature name and expected binary values
    feature = "keepAlive"
    values = [0, 1]

    # Compute stats for each group
    stats = {}
    for v in values:
        stats[v] = group_stats(header, data, feature, v)

    # Unpack
    mean0, std0, n0 = stats[0]
    mean1, std1, n1 = stats[1]

    # Compute effect size
    std_pooled = pooled_std(std0, n0, std1, n1)
    d = effect_size(mean1, mean0, std_pooled)
    interp = interpret_effect(d)

    # Print results
    print(f"KeepAlive=0: n={n0}, mean={mean0:.2f}, std={std0:.2f}")
    print(f"KeepAlive=1: n={n1}, mean={mean1:.2f}, std={std1:.2f}")
    print(f"Mean difference (1 – 0): {mean1 - mean0:.2f}")
    print(f"Pooled std: {std_pooled:.2f}")
    print(f"Cohen's d: {d:.2f} ({interp} effect)")
    print("Interpretation: ", end='')
    if d >= 0.5:
        print("The difference is moderate to large; enabling KeepAlive "
              "appears to impact performance enough to consider in decisions.")
    else:
        print("The difference is small or negligible; more evidence is needed "
              "before making a decision based on KeepAlive.")


# --------------------------------------------------------------------------- #
# APACHE-06: Access logging comparison
# --------------------------------------------------------------------------- #

def apache_06(header, data):
    print("\n== APACHE-06 ==")

    feature = "accessLog"
    values = [0, 1]

    stats = {}
    for v in values:
        stats[v] = group_stats(header, data, feature, v)

    mean0, std0, n0 = stats[0]
    mean1, std1, n1 = stats[1]

    std_pooled = pooled_std(std0, n0, std1, n1)
    d = effect_size(mean1, mean0, std_pooled)
    interp = interpret_effect(d)

    print(f"accessLog=0: n={n0}, mean={mean0:.2f}, std={std0:.2f}")
    print(f"accessLog=1: n={n1}, mean={mean1:.2f}, std={std1:.2f}")
    print(f"Mean difference (1 – 0): {mean1 - mean0:.2f}")
    print(f"Pooled std: {std_pooled:.2f}")
    print(f"Cohen's d: {d:.2f} ({interp} effect)")
    print("Interpretation: ", end='')
    if d >= 0.5:
        print("Access logging has a moderate to large effect on performance; "
              "consider disabling it if performance is critical.")
    else:
        print("The effect of access logging is small or unclear; "
              "current evidence is inconclusive.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Summary from earlier skills
    summarize_performance(header, data)
    identify_favorable(header, data)

    # Model training
    model, XtX_inv, resid_var = train_linear_model(data, header)

    # APACHE-03 and APACHE-04 for reference
    apache_03(header, data, model, XtX_inv, resid_var)
    apache_04(header, data, model, XtX_inv, resid_var)

    # New requests
    apache_05(header, data)
    apache_06(header, data)


if __name__ == "__main__":
    main()

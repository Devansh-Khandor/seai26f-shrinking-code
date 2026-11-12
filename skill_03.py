#!/usr/bin/env python3
"""
Skill 03: Cylinder & Model-Year comparison for the auto93 dataset.

This script implements CAR-05 and CAR-06:
- CAR-05: Compare fuel efficiency (MPG) across different cylinder counts.
- CAR-06: Compare overall performance (MPG) between two model-year groups.

Usage:
    python skill_03.py <path_to_auto93.csv>
"""

import sys
import math
import numpy as np

# Re-use utilities from earlier skills
from skill_02 import (
    load_vehicles,
    compute_stats,
    Vehicle,
)

# --------------------------------------------------------------------------- #
# Helper statistical utilities
# --------------------------------------------------------------------------- #

def mean_and_std(values):
    """Return mean and standard deviation of a list of numbers."""
    if not values:
        return None, None
    arr = np.array(values)
    return np.mean(arr), np.std(arr, ddof=1)  # sample std

def ci_diff(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    """
    Compute difference of means and its confidence interval.
    Returns (diff, lower_ci, upper_ci)
    """
    diff = mean1 - mean2
    se = math.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
    z = 1.96  # approx for 95% CI
    lower = diff - z * se
    upper = diff + z * se
    return diff, lower, upper

# --------------------------------------------------------------------------- #
# CAR-05: Fuel efficiency by cylinder count
# --------------------------------------------------------------------------- #

def compare_mpg_by_cylinders(vehicles):
    """
    Compare mean MPG for each cylinder count pairwise.
    Returns a list of result strings.
    """
    # Group mpg by cylinder count, ignoring vehicles with missing mpg
    cyl_groups = {}
    for v in vehicles:
        if v.Mpg is None:
            continue
        cyl_groups.setdefault(v.Clndrs, []).append(v.Mpg)

    if len(cyl_groups) < 2:
        return ["Not enough cylinder groups to compare."]

    # Pre-compute stats for each group
    group_stats = {}
    for cyl, mpgs in cyl_groups.items():
        mean, std = mean_and_std(mpbs)
        group_stats[cyl] = {"mean": mean, "std": std, "n": len(mpbs)}

    results = [f"Analyzing {len(cyl_groups)} cylinder groups: {sorted(cyl_groups.keys())}"]
    cyl_list = sorted(cyl_groups.keys())
    for i in range(len(cyl_list)):
        for j in range(i + 1, len(cyl_list)):
            c1, c2 = cyl_list[i], cyl_list[j]
            s1, s2 = group_stats[c1], group_stats[c2]
            diff, lower, upper = ci_diff(
                s1["mean"], s1["std"], s1["n"],
                s2["mean"], s2["std"], s2["n"]
            )
            sign = "significant" if (lower > 0 or upper < 0) else "not significant"
            results.append(
                f"Cylinders {c1} vs {c2}: mean MPG {s1['mean']:.2f} vs {s2['mean']:.2f}, "
                f"diff {diff:.2f} (95% CI {lower:.2f} to {upper:.2f}) => {sign}"
            )
    return results

# --------------------------------------------------------------------------- #
# CAR-06: Model-year group comparison
# --------------------------------------------------------------------------- #

def compare_groups_by_model_year(vehicles):
    """
    Split vehicles into two groups based on model year median
    and compare mean MPG. Return list of result strings.
    """
    # Ensure we have mpg values
    mpg_vehicles = [v for v in vehicles if v.Mpg is not None]
    if len(mpg_vehicles) < 2:
        return ["Not enough vehicles with known MPG to compare."]

    # Determine median model year
    model_years = [v.Model for v in mpg_vehicles]
    median_year = np.median(model_years)

    group_a = [v.Mpg for v in mpg_vehicles if v.Model < median_year]
    group_b = [v.Mpg for v in mpg_vehicles if v.Model >= median_year]

    # If either group empty, use alternative split (e.g., quartiles)
    if not group_a or not group_b:
        # fallback: split at first quartile
        q1 = np.percentile(model_years, 25)
        group_a = [v.Mpg for v in mpg_vehicles if v.Model <= q1]
        group_b = [v.Mpg for v in mpg_vehicles if v.Model > q1]

    if not group_a or not group_b:
        return ["Unable to form two distinct model-year groups for comparison."]

    # Compute stats
    mean_a, std_a = mean_and_std(group_a)
    mean_b, std_b = mean_and_std(group_b)
    n_a, n_b = len(group_a), len(group_b)

    diff, lower, upper = ci_diff(mean_a, std_a, n_a, mean_b, std_b, n_b)

    sign = "significant" if (lower > 0 or upper < 0) else "not significant"
    return [
        f"Model-year split at median {median_year:.1f}.",
        f"Group A (year < {median_year:.0f}): n={n_a}, mean MPG={mean_a:.2f}",
        f"Group B (year ≥ {median_year:.0f}): n={n_b}, mean MPG={mean_b:.2f}",
        f"Mean MPG difference: {diff:.2f} (95% CI {lower:.2f} to {upper:.2f}) => {sign}",
    ]

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-05
    print("== CAR-05 ==")
    for line in compare_mpg_by_cylinders(vehicles):
        print(line)

    # CAR-06
    print("\n== CAR-06 ==")
    for line in compare_groups_by_model_year(vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_03.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
Skill 05: In-depth vehicle performance analysis – CAR-09 and CAR-10.

This script implements:
  CAR-09 – Identify the most strongly correlated attributes for a vehicle
           with unusually poor fuel efficiency.
  CAR-10 – Find the common characteristics of vehicles with unfavorable
           overall performance and compare them to better‑performing vehicles.

Usage:
    python skill_05.py <path_to_auto93.csv>
"""

import sys
import numpy as np

# Re‑use utilities from earlier skills
from skill_02 import (
    load_vehicles,
    compute_stats,
    best_combination,
    Vehicle,
)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _numeric_features(v):
    """Return a list of numeric feature values for a Vehicle."""
    return [
        v.Clndrs,
        v.Volume,
        v.HpX,
        v.Model,
        v.origin,
        v.Lbs,
        v.Acc,
    ]

def _feature_names():
    """Return the names of numeric features in the order of _numeric_features."""
    return ["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs", "Acc"]

def _correlations_with_mpg(vehicles):
    """
    Compute Pearson correlation coefficients between each numeric feature
    and mpg across all vehicles that have a known mpg value.
    """
    # Filter vehicles with known mpg
    data = [
        (_numeric_features(v) + [v.Mpg]) for v in vehicles if v.Mpg is not None
    ]
    if not data:
        return {}

    arr = np.array(data)
    # Transpose so that each row is a feature
    corr_matrix = np.corrcoef(arr.T)
    # corr_matrix[feature_index, -1] gives correlation with mpg
    feature_correlations = {}
    for idx, name in enumerate(_feature_names()):
        feature_correlations[name] = corr_matrix[idx, -1]
    return feature_correlations

def _lowest_mpg_vehicle(vehicles):
    """Return the Vehicle instance with the lowest non‑None mpg."""
    lowest = None
    for v in vehicles:
        if v.Mpg is None:
            continue
        if lowest is None or v.Mpg < lowest.Mpg:
            lowest = v
    return lowest

def _split_by_performance(vehicles):
    """
    Split vehicles into two groups based on composite performance score.
    Returns (unfavorable_group, favorable_group, feature_means).
    """
    # Compute composite scores
    scores = best_combination(vehicles, top_n=len(vehicles))
    # scores: list of (score, idx, Vehicle)
    score_values = [s for s, _, _ in scores]
    median_score = np.median(score_values)

    unfavorable = []
    favorable = []
    for score, idx, v in scores:
        if score <= median_score:
            unfavorable.append((idx, v))
        else:
            favorable.append((idx, v))

    def mean_of_group(group):
        features = np.array([_numeric_features(v) for _, v in group])
        return features.mean(axis=0) if features.size else np.array([])

    unfav_means = mean_of_group(unfavorable)
    fav_means = mean_of_group(favorable)
    return unfavorable, favorable, unfav_means, fav_means

# --------------------------------------------------------------------------- #
# CAR-09 implementation
# --------------------------------------------------------------------------- #

def analyze_poor_fuel_efficiency(vehicles):
    """
    Identify the attributes most strongly negatively correlated with MPG
    for the vehicle with the lowest fuel efficiency.
    Returns a list of result strings.
    """
    low_vehicle = _lowest_mpg_vehicle(vehicles)
    if low_vehicle is None:
        return ["No vehicle with known MPG to analyze."]

    correlations = _correlations_with_mpg(vehicles)

    # Sort attributes by magnitude of negative correlation
    sorted_attrs = sorted(
        correlations.items(),
        key=lambda kv: kv[1]  # correlation value
    )
    # Take top 3 most negative correlations
    top_neg = sorted_attrs[:3]

    lines = []
    idx = vehicles.index(low_vehicle)
    lines.append(f"Vehicle {idx + 1} has the lowest MPG: {low_vehicle.Mpg:.2f}")
    lines.append("Most strongly negatively correlated attributes:")
    for name, corr in top_neg:
        lines.append(f"  {name:>6} : correlation = {corr:.3f}")
    return lines

# --------------------------------------------------------------------------- #
# CAR-10 implementation
# --------------------------------------------------------------------------- #

def compare_unfavorable_vs_favorable(vehicles):
    """
    Compare common characteristics of vehicles with unfavorable overall
    performance to those with favorable performance.
    Returns a list of result strings.
    """
    unfav, fav, unfav_means, fav_means = _split_by_performance(vehicles)
    if not unfav or not fav:
        return ["Insufficient data to split performance groups."]

    feature_names = _feature_names()
    diffs = []
    for i, name in enumerate(feature_names):
        diff = fav_means[i] - unfav_means[i]
        diffs.append((name, unfav_means[i], fav_means[i], diff))

    # Sort by absolute difference descending
    diffs.sort(key=lambda t: abs(t[3]), reverse=True)
    top_diffs = diffs[:3]

    lines = []
    lines.append(f"Unfavorable group size: {len(unfav)}")
    lines.append(f"Favorable group size: {len(fav)}")
    lines.append("Top 3 attribute differences (favorable - unfavorable):")
    for name, unf_val, fav_val, diff in top_diffs:
        lines.append(
            f"  {name:>6} : unfav={unf_val:.2f}, fav={fav_val:.2f}, diff={diff:.2f}"
        )
    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-09
    print("== CAR-09 ==")
    for line in analyze_poor_fuel_efficiency(vehicles):
        print(line)

    # CAR-10
    print("\n== CAR-10 ==")
    for line in compare_unfavorable_vs_favorable(vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_05.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv_path>

The script prints:
    * CAR-09: Identify characteristics most strongly associated with unusually poor fuel efficiency.
    * CAR-10: Common characteristics of vehicles with unfavorable overall performance and how they differ from better performers.
"""

import sys
import math
from pathlib import Path

# Import helpers from previous skills
from skill_01 import read_csv, compute_composite_scores, format_number, mean

# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #

def pearson_corr(values_x, values_y):
    """Compute Pearson correlation coefficient between two numeric lists."""
    n = len(values_x)
    if n == 0:
        return float('nan')
    mean_x = mean(values_x)
    mean_y = mean(values_y)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in values_x))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in values_y))
    if den_x == 0 or den_y == 0:
        return float('nan')
    return num / (den_x * den_y)

# --------------------------------------------------------------------------- #
# CAR-09: Poor fuel efficiency correlation analysis
# --------------------------------------------------------------------------- #

def print_car_09_analysis(data):
    # List of features to analyze (numeric)
    feature_keys = [
        'Clndrs',
        'Volume',
        'HpX',
        'Model',
        'origin',
        'Lbs-',
        'Acc+',
    ]

    mpgs = [v['Mpg+'] for v in data if v['Mpg+'] > 0]

    # Compute correlation of each feature with mpg
    correlations = {}
    for key in feature_keys:
        values = [v[key] for v in data]
        corr = pearson_corr(values, mpgs)
        correlations[key] = corr

    # Sort by most negative correlation (i.e., strongest association with low MPG)
    sorted_corr = sorted(
        ((k, v) for k, v in correlations.items() if not math.isnan(v)),
        key=lambda item: item[1]
    )

    print("\n== CAR-09 ==")
    print("Characteristics most strongly associated with unusually poor fuel efficiency (low MPG):")
    if not sorted_corr:
        print("  No sufficient data to compute correlations.")
        return

    # Show top 3 features
    for i, (feature, corr) in enumerate(sorted_corr[:3], start=1):
        print(
            f"  {i}. Feature '{feature}' correlation with MPG: {format_number(corr)}"
        )

# --------------------------------------------------------------------------- #
# CAR-10: Common characteristics of unfavorable performers
# --------------------------------------------------------------------------- #

def print_car_10_characteristics(data):
    # Compute composite scores
    scores = compute_composite_scores(data)
    if not scores:
        print("\n== CAR-10 ==")
        print("  No vehicle records to analyze.")
        return

    # Determine threshold for unfavorable vehicles (worst 25% scores)
    scores.sort(key=lambda x: x[0])  # lower is better
    n = len(scores)
    top_25_pct = math.ceil(0.25 * n)
    unfavorable_indices = range(n - top_25_pct, n)
    unfavorable_scores = [scores[i][1] for i in unfavorable_indices]

    # Features to compare
    feature_keys = [
        'Clndrs',
        'Volume',
        'HpX',
        'Model',
        'origin',
        'Lbs-',
        'Acc+',
        'Mpg+',
    ]

    # Compute overall averages
    overall_avgs = {k: mean([v[k] for v in data]) for k in feature_keys}

    # Compute averages for unfavorable subset
    unfav_avgs = {k: mean([v[k] for v in unfavorable_scores]) for k in feature_keys}

    print("\n== CAR-10 ==")
    print("Common characteristics of vehicles with unfavorable overall performance:")
    if not unfavorable_scores:
        print("  No vehicles identified as unfavorable.")
        return

    # Show difference for each feature (unfavorable - overall)
    # We display at least three differences
    for k in feature_keys:
        diff = unfav_avgs[k] - overall_avgs[k]
        # Only show if diff is significant or if we want at least 3 lines
        print(
            f"  Feature '{k}': overall mean = {format_number(overall_avgs[k])}, "
            f"unfavorable mean = {format_number(unfav_avgs[k])}, "
            f"diff = {format_number(diff)}"
        )

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    # CAR-09
    print_car_09_analysis(data)

    # CAR-10
    print_car_10_characteristics(data)

if __name__ == "__main__":
    main()

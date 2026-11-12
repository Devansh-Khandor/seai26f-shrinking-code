#!/usr/bin/env python3
"""
Skill 05: Unfavorable Performance Analysis (CAR-09 & CAR-10)

Usage:
    python skill_05.py <path_to_auto93.csv>
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #

def load_csv(path: str) -> np.ndarray:
    """
    Load the CSV file into a structured NumPy array with all columns as floats.
    Missing values are represented by NaN.
    """
    try:
        data = np.genfromtxt(
            path,
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Convert numeric columns to float
    numeric_cols = [
        'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
        'Lbs-', 'Acc+', 'Mpg+'
    ]
    for col in numeric_cols:
        if col in data.dtype.names:
            try:
                data[col] = data[col].astype(float)
            except Exception:
                pass
    return data

# --------------------------------------------------------------------------- #
# Composite score calculation
# --------------------------------------------------------------------------- #

def composite_score(data: np.ndarray) -> np.ndarray:
    """
    Return an array of composite scores for each vehicle.
    Lower weight, higher mpg, lower acceleration => higher score.
    """
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    def norm_minmax(arr):
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

    weight_score = 1.0 - norm_minmax(weight)  # lower is better
    mpg_score    = norm_minmax(mpg)           # higher is better
    acc_score    = 1.0 - norm_minmax(acc)    # lower is better

    return weight_score + mpg_score + acc_score

# --------------------------------------------------------------------------- #
# CAR-09: Characteristics associated with poor fuel efficiency
# --------------------------------------------------------------------------- #

def car_09(data: np.ndarray):
    print("== CAR-09 ==")

    mpg = data['Mpg+'].astype(float)

    # Bottom 25% of MPG
    threshold = np.nanpercentile(mpg, 25)
    poor_mask = mpg < threshold

    # Features to evaluate
    feature_names = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+']
    correlations = []

    for fname in feature_names:
        feat = data[fname].astype(float)[poor_mask]
        mpg_poor = mpg[poor_mask]

        # Remove pairs with NaN
        valid_mask = ~np.isnan(feat) & ~np.isnan(mpg_poor)
        if valid_mask.sum() < 2:
            corr = np.nan
        else:
            corr = np.corrcoef(feat[valid_mask], mpg_poor[valid_mask])[0, 1]

        correlations.append((fname, corr))

    # Sort by absolute correlation descending
    correlations.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else -1, reverse=True)

    # Print top 3
    for fname, corr in correlations[:3]:
        print(f"{fname:>7} : correlation with MPG = {corr:+.3f}")

    print()

# --------------------------------------------------------------------------- #
# CAR-10: Common characteristics of unfavorable overall performance
# --------------------------------------------------------------------------- #

def car_10(data: np.ndarray):
    print("== CAR-10 ==")

    comp = composite_score(data)

    # Bottom 20% composite score
    threshold = np.nanpercentile(comp, 20)
    unfav_mask = comp < threshold

    # Compute overall means
    overall_means = {}
    for col in data.dtype.names:
        vals = data[col].astype(float)
        overall_means[col] = np.nanmean(vals)

    # Compute means for unfavorable group
    diff_stats = []
    for col in data.dtype.names:
        vals = data[col].astype(float)[unfav_mask]
        mean_val = np.nanmean(vals)
        diff = mean_val - overall_means[col]
        diff_stats.append((col, mean_val, diff))

    # Sort by absolute difference descending
    diff_stats.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else -1, reverse=True)

    # Print top 3
    for col, mean_val, diff in diff_stats[:3]:
        direction = "higher" if diff > 0 else "lower"
        print(f"{col:>7} : mean = {mean_val:+.2f} ({direction} than overall)")

    print()

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    data = load_csv(csv_path)
    car_09(data)
    car_10(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)

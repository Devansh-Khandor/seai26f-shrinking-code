#!/usr/bin/env python3
"""
Skill 01: Inventory Summary & Composite Vehicle Ranking

Usage:
    python skill_01.py <path_to_auto93.csv>
"""

import sys
import numpy as np

def main(csv_path):
    # Load the CSV into a structured NumPy array
    try:
        data = np.genfromtxt(
            csv_path,
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Check that required columns exist
    required_cols = ('Lbs-', 'Mpg+', 'Acc+')
    if not all(col in data.dtype.names for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        sys.exit(1)

    # Extract numeric columns
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    # Summary statistics
    w_mean, w_min, w_max = weight.mean(), weight.min(), weight.max()
    mpg_mean, mpg_min, mpg_max = mpg.mean(), mpg.min(), mpg.max()
    acc_mean, acc_min, acc_max = acc.mean(), acc.min(), acc.max()

    # Print CAR-01 summary
    print("== CAR-01 ==")
    print(f"Weight: mean={w_mean:.2f} lbs, min={w_min:.2f} lbs, max={w_max:.2f} lbs")
    print(f"Fuel efficiency: mean={mpg_mean:.2f} mpg, min={mpg_min:.2f} mpg, max={mpg_max:.2f} mpg")
    print(f"Acceleration: mean={acc_mean:.2f} s, min={acc_min:.2f} s, max={acc_max:.2f} s")
    print()

    # Composite score calculation
    # Normalise each feature to [0,1]
    def norm_minmax(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    # Weight: lower is better => 1 - normalized
    weight_score = 1.0 - norm_minmax(weight)
    # MPG: higher is better => normalized directly
    mpg_score = norm_minmax(mpg)
    # Acceleration: lower is better => 1 - normalized
    acc_score = 1.0 - norm_minmax(acc)

    composite = weight_score + mpg_score + acc_score

    # Add composite to structured array
    dtype_with_score = data.dtype.descr + [('CompositeScore', '<f8')]
    structured_with_score = np.empty(data.shape, dtype=dtype_with_score)
    for name in data.dtype.names:
        structured_with_score[name] = data[name]
    structured_with_score['CompositeScore'] = composite

    # Sort by composite score descending
    sorted_indices = np.argsort(-structured_with_score['CompositeScore'])
    top_n = 5
    top_indices = sorted_indices[:top_n]

    # Print CAR-02 ranking
    print("== CAR-02 ==")
    print(f"Top {top_n} vehicles with best composite score (low weight, high mpg, low acceleration):")
    header = ["Idx", "Clndrs", "Volume", "HpX", "Model", "origin", "Lbs-", "Acc+", "Mpg+", "Score"]
    print("{:>4} {:>7} {:>7} {:>4} {:>5} {:>7} {:>6} {:>5} {:>5} {:>6}".format(*header))
    for rank, idx in enumerate(top_indices, start=1):
        row = structured_with_score[idx]
        print("{:>4} {:>7} {:>7} {:>4} {:>5} {:>7} {:>6.1f} {:>5.1f} {:>5.1f} {:>6.3f}".format(
            rank,
            row['Clndrs'],
            row['Volume'],
            row['HpX'],
            row['Model'],
            row['origin'],
            row['Lbs-'],
            row['Acc+'],
            row['Mpg+'],
            row['CompositeScore']
        ))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path_to_auto93.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

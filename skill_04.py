#!/usr/bin/env python3
"""
Skill 04: Dynamic Inventory Assessment & Summary (CAR-07 & CAR-08)

Usage:
    python skill_04.py <path_to_auto93.csv>
"""

import sys
import time
import os
import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def load_csv(path: str):
    """Load the CSV file into a structured NumPy array with all columns as floats."""
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

    # Cast numeric columns to float (missing values become nan)
    numeric_cols = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                    'Lbs-', 'Acc+', 'Mpg+']
    for col in numeric_cols:
        if col in data.dtype.names:
            try:
                data[col] = data[col].astype(float)
            except Exception:
                # If conversion fails, keep as original (likely string)
                pass
    return data


def composite_score(data):
    """Return an array of composite scores for each vehicle."""
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    def norm_minmax(arr):
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

    weight_score = 1.0 - norm_minmax(weight)
    mpg_score    = norm_minmax(mpg)
    acc_score    = 1.0 - norm_minmax(acc)

    return weight_score + mpg_score + acc_score


def analyze(csv_path):
    """Run CAR-07 and CAR-08 analysis and print the results."""
    data = load_csv(csv_path)

    # Composite scores for all vehicles
    comp = composite_score(data)

    # ----------------------------------------------------------------------- #
    # CAR-07 : Immediate desirability assessment
    # ----------------------------------------------------------------------- #
    threshold = np.percentile(comp, 80)  # top 20% are considered desirable
    print("== CAR-07 ==")
    for idx, score in enumerate(comp):
        status = "DESIRABLE" if score >= threshold else "NOT DESIRABLE"
        print(f"Vehicle {idx:3d}: score = {score:.3f} -> {status}")
    print()

    # ----------------------------------------------------------------------- #
    # CAR-08 : Updated inventory summary & shortlist
    # ----------------------------------------------------------------------- #
    print("== CAR-08 ==")

    # Summary statistics
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    def stats(arr, name):
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return (np.nan, np.nan, np.nan)
        return (arr.mean(), arr.min(), arr.max())

    w_mean, w_min, w_max = stats(weight, "Weight")
    m_mean, m_min, m_max = stats(mpg, "MPG")
    a_mean, a_min, a_max = stats(acc, "Acceleration")

    print(f"Inventory Summary:")
    print(f"  Weight: mean={w_mean:.2f} lbs, min={w_min:.2f} lbs, max={w_max:.2f} lbs")
    print(f"  MPG:    mean={m_mean:.2f} mpg, min={m_min:.2f} mpg, max={m_max:.2f} mpg")
    print(f"  Acc:    mean={a_mean:.2f} s,   min={a_min:.2f} s,   max={a_max:.2f} s")
    print()

    # Top 5 vehicles by composite score
    top_n = 5
    top_indices = np.argsort(-comp)[:top_n]
    header = ["Rank", "Idx", "Clndrs", "Volume", "HpX", "Model", "origin",
              "Lbs-", "Acc+", "Mpg+", "Score"]
    print(f"Top {top_n} vehicles (by composite score):")
    print("{:>4} {:>4} {:>7} {:>7} {:>4} {:>5} {:>7} {:>6.1f} {:>5.1f} {:>5.1f} {:>6.3f}".format(*header))
    for rank, idx in enumerate(top_indices, start=1):
        row = data[idx]
        print("{:>4} {:>4} {:>7} {:>7} {:>4} {:>5} {:>7} {:>6.1f} {:>5.1f} {:>5.1f} {:>6.3f}".format(
            rank,
            idx,
            int(row['Clndrs']) if not np.isnan(row['Clndrs']) else np.nan,
            int(row['Volume']) if not np.isnan(row['Volume']) else np.nan,
            int(row['HpX']) if not np.isnan(row['HpX']) else np.nan,
            int(row['Model']) if not np.isnan(row['Model']) else np.nan,
            int(row['origin']) if not np.isnan(row['origin']) else np.nan,
            row['Lbs-'],
            row['Acc+'],
            row['Mpg+'],
            comp[idx]
        ))
    print()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main(csv_path):
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    last_mtime = None
    # Initial run
    analyze(csv_path)
    last_mtime = os.path.getmtime(csv_path)

    # Poll for changes every 5 seconds
    try:
        while True:
            time.sleep(5)
            try:
                current_mtime = os.path.getmtime(csv_path)
            except OSError:
                # File disappeared
                print(f"File disappeared: {csv_path}")
                sys.exit(1)
            if current_mtime != last_mtime:
                print("\n=== Detected changes, re-running analysis ===\n")
                analyze(csv_path)
                last_mtime = current_mtime
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <path_to_auto93.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

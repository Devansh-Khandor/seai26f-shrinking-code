#!/usr/bin/env python3
"""
Skill 01: Apache performance analysis
"""

import sys
import csv
import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv_to_numpy(path):
    """
    Read a CSV file into a tuple (header, data_array) where data_array is a
    NumPy array of shape (n_samples, n_features) with dtype=int.
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(int, row)) for row in reader]
    data = np.array(rows, dtype=int)
    return header, data


def summarize_performance(header, data):
    """
    Print a summary of performance: mean (typical), max (best), min (worst).
    Assumes the performance column is the last column.
    """
    perf_idx = len(header) - 1
    performance = data[:, perf_idx]
    typical = np.mean(performance)
    best = np.max(performance)
    worst = np.min(performance)

    print("== APACHE-01 ==")
    print(f"Typical performance (mean): {typical:.2f}")
    print(f"Best performance (max): {best}")
    print(f"Worst performance (min): {worst}")


def identify_favorable(header, data, percentile=95):
    """
    Identify configurations with unusually favorable performance values.
    Print which settings (features) distinguish them from the rest.

    A configuration is considered favorable if its performance value is above
    the given percentile. If fewer than 3 records qualify, the top 3 are used.
    """
    perf_idx = len(header) - 1
    performance = data[:, perf_idx]
    # Determine threshold
    threshold = np.percentile(performance, percentile)
    fav_mask = performance >= threshold

    # Fallback to at least 3 records
    if fav_mask.sum() < 3:
        fav_indices = np.argsort(performance)[-3:]
        fav_mask = np.zeros_like(performance, dtype=bool)
        fav_mask[fav_indices] = True

    favorable = data[fav_mask]
    rest = data[~fav_mask]

    # Compute mean of each feature for favorable and rest
    fav_means = np.mean(favorable, axis=0)
    rest_means = np.mean(rest, axis=0)

    # Identify features with significant difference (absolute > 0.5 or >10% of rest mean)
    diff = fav_means - rest_means
    significant = []
    for i, (d, r) in enumerate(zip(diff, rest_means)):
        if abs(d) > 0.5 or (abs(d) > 0.1 * abs(r)):
            significant.append((header[i], d))

    print("\n== APACHE-02 ==")
    print(f"Threshold for favorable: performance >= {threshold:.2f}")
    print(f"Number of favorable configurations: {fav_mask.sum()}")
    if significant:
        print("Features distinguishing favorable configs:")
        for name, d in significant:
            sign = '+' if d > 0 else '-'
            print(f"  {name}: {sign}{abs(d):.2f}")
    else:
        print("No features show significant difference.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)
    summarize_performance(header, data)
    identify_favorable(header, data)


if __name__ == "__main__":
    main()

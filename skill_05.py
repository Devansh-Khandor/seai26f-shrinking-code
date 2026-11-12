#!/usr/bin/env python3
"""
Skill 05: Apache performance analysis – blame & frequent poor combos
"""

import sys
import numpy as np
from collections import Counter

# Import helpers from earlier skills
from skill_01 import read_csv_to_numpy, summarize_performance, identify_favorable


# --------------------------------------------------------------------------- #
# APACHE-09: Identify settings most strongly associated with poor performance
# --------------------------------------------------------------------------- #
def apache_09(header, data):
    """
    For each poorly performing configuration (performance in the lowest 5th percentile),
    list the configuration's enabled/disabled settings that are most strongly
    associated with unfavorable performance based on historical averages.
    """
    print("\n== APACHE-09 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Define poor-performance threshold (5th percentile)
    threshold = np.percentile(performance, 5)
    poor_mask = performance <= threshold
    poor_indices = np.where(poor_mask)[0]

    if len(poor_indices) == 0:
        print("No poor configurations found (no values in the lowest 5th percentile).")
        return

    print(f"Poor-performance threshold: performance <= {threshold:.2f} (5th percentile)")
    print(f"Number of poor configurations: {len(poor_indices)}\n")

    # Compute mean feature values for all data and for poor data
    mean_all = np.mean(data[:, :-1], axis=0)
    mean_poor = np.mean(data[poor_mask, :-1], axis=0)

    # Feature difference between poor and overall
    diff = mean_poor - mean_all
    # Sort features by absolute difference (most associated)
    top_feature_indices = np.argsort(np.abs(diff))[::-1][:3]

    # Show details for first few poor configurations
    for idx in poor_indices[:min(3, len(poor_indices))]:
        print(f"Config {idx} (performance: {performance[idx]})")
        print("  Features most strongly associated with poor performance:")
        for fidx in top_feature_indices:
            feature = header[fidx]
            diff_val = diff[fidx]
            sign = '+' if diff_val > 0 else '-'
            print(f"    {feature}: {sign}{abs(diff_val):.2f}")
        print()


# --------------------------------------------------------------------------- #
# APACHE-10: Frequent poor-performance combinations
# --------------------------------------------------------------------------- #
def apache_10(header, data, top_n=3):
    """
    Identify combinations of server settings that frequently appear among
    poorly performing configurations and compare their mean performance with
    the overall mean.
    """
    print("\n== APACHE-10 ==")

    perf_idx = len(header) - 1
    performance = data[:, perf_idx]

    # Define poor-performance threshold (5th percentile)
    threshold = np.percentile(performance, 5)
    poor_mask = performance <= threshold
    if not np.any(poor_mask):
        print("No poor configurations found (no values in the lowest 5th percentile).")
        return

    poor_rows = data[poor_mask, :-1]
    all_rows = data[:, :-1]

    # Convert rows to tuples for counting
    poor_tuples = [tuple(row) for row in poor_rows]
    all_tuples = [tuple(row) for row in all_rows]

    poor_counts = Counter(poor_tuples)
    all_counts = Counter(all_tuples)

    # Get the most frequent combinations among poor configurations
    combos = sorted(poor_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not combos:
        print("No frequent combinations among poor configurations.")
        return

    overall_mean_perf = np.mean(performance)

    for combo, poor_cnt in combos:
        combo_list = list(combo)
        # Mask all rows matching this combo
        combo_mask = np.array([tuple(row) == combo for row in all_rows])
        combo_perf = performance[combo_mask]
        combo_mean_perf = np.mean(combo_perf) if len(combo_perf) > 0 else float('nan')
        ratio = poor_cnt / all_counts[combo]

        print(f"\nCombination {combo_list}")
        print(f"  Occurs {poor_cnt} times among poor configs "
              f"(ratio {ratio:.2f} of all configs)")
        print(f"  Mean performance for this combo: {combo_mean_perf:.2f}")
        print(f"  Overall mean performance: {overall_mean_perf:.2f}")
        print(f"  Difference: {combo_mean_perf - overall_mean_perf:.2f}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, data = read_csv_to_numpy(csv_path)

    # Basic summary
    summarize_performance(header, data)
    identify_favorable(header, data)

    # New requests
    apache_09(header, data)
    apache_10(header, data, top_n=3)


if __name__ == "__main__":
    main()

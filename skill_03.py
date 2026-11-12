#!/usr/bin/env python3
"""
skill_03.py

Entry point for CAR-05 and CAR-06 skills.

Usage:
    python skill_03.py <path/to/auto93.csv>
"""

import argparse
import sys
from typing import Dict, Tuple, List

import numpy as np

# Reuse utilities from existing modules
# We import the read_csv_as_array and compute_summary_stats functions
# from the skill_02 module as they handle NaNs gracefully.
try:
    from skill_02 import read_csv_as_array, compute_summary_stats
except Exception:
    # Fallback: provide minimal implementations if skill_02 is missing
    def read_csv_as_array(path: str) -> np.ndarray:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                processed_row = []
                for cell in row:
                    if cell.strip() == "":
                        processed_row.append(np.nan)
                    else:
                        try:
                            processed_row.append(float(cell))
                        except ValueError:
                            processed_row.append(np.nan)
                data.append(processed_row)
        return np.array(data, dtype=float)

    def compute_summary_stats(arr: np.ndarray, col_index: int):
        col = arr[:, col_index]
        mean = float(np.nanmean(col))
        min_val = float(np.nanmin(col))
        max_val = float(np.nanmax(col))
        return mean, min_val, max_val


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def group_by_column(arr: np.ndarray, col_index: int) -> Dict[float, np.ndarray]:
    """
    Groups rows of arr by unique values in the specified column.
    Returns a dict mapping unique value -> indices (array of int).
    """
    unique_vals = np.unique(arr[:, col_index])
    groups = {}
    for val in unique_vals:
        groups[val] = np.where(arr[:, col_index] == val)[0]
    return groups


def compute_group_stats(arr: np.ndarray, indices: np.ndarray, col_index: int) -> Tuple[float, float, int]:
    """
    Computes mean, standard deviation, and count of values at col_index for the given indices.
    Ignores NaN values.
    """
    values = arr[indices, col_index]
    values = values[~np.isnan(values)]
    mean = float(np.mean(values)) if values.size > 0 else float('nan')
    std = float(np.std(values, ddof=1)) if values.size > 1 else float('nan')
    count = int(values.size)
    return mean, std, count


# --------------------------------------------------------------------------- #
# CAR-05: Compare fuel efficiency across cylinder counts
# --------------------------------------------------------------------------- #

def print_car05(arr: np.ndarray) -> None:
    """
    Prints comparison of fuel efficiency (mpg) by cylinder count.
    Provides means, standard deviations, and pairwise differences.
    """
    print("== CAR-05 ==")
    cyl_groups = group_by_column(arr, 0)  # cylinder count in column 0
    mpg_stats: Dict[float, Tuple[float, float, int]] = {}

    # Compute stats for each cylinder group
    for cyl, indices in sorted(cyl_groups.items()):
        mean, std, count = compute_group_stats(arr, indices, 7)  # mpg in column 7
        mpg_stats[cyl] = (mean, std, count)
        print(f"Cylinders {cyl}: mean mpg = {mean:.2f}, std = {std:.2f}, n = {count}")

    # Pairwise comparisons
    cyl_values = sorted(mpg_stats.keys())
    print("\nPairwise mean mpg differences:")
    for i in range(len(cyl_values)):
        for j in range(i + 1, len(cyl_values)):
            c1, c2 = cyl_values[i], cyl_values[j]
            m1, s1, n1 = mpg_stats[c1]
            m2, s2, n2 = mpg_stats[c2]
            diff = abs(m1 - m2)
            # Pooled standard deviation for difference estimate
            pooled_std = np.sqrt((s1 ** 2 / n1) + (s2 ** 2 / n2)) if n1 > 1 and n2 > 1 else np.nan
            # Decision rule: if diff > 2 * pooled_std, consider reliable
            reliable = pooled_std != np.nan and diff > 2 * pooled_std
            status = "RELIABLE" if reliable else "NOT RELIABLE"
            print(f"  {c1} vs {c2}: diff = {diff:.2f} mpg, pooled SE = {pooled_std:.2f} => {status}")

    print("\nInterpretation:")
    print("If the difference between cylinder groups is more than twice the pooled standard error,")
    print("the observed difference is statistically reliable and can inform inventory decisions.")


# --------------------------------------------------------------------------- #
# CAR-06: Compare two model-year ranges
# --------------------------------------------------------------------------- #

def print_car06(arr: np.ndarray) -> None:
    """
    Compares overall performance (fuel efficiency and weight) between two model-year ranges.
    Determines if differences are meaningful or evidence is weak.
    """
    print("== CAR-06 ==")
    years = arr[:, 3]  # model year in column 3
    unique_years = np.unique(years[~np.isnan(years)])

    if unique_years.size < 2:
        print("Insufficient variation in model years to form two groups.")
        return

    # Split into two groups: lower half years vs upper half years
    median_year = np.median(unique_years)
    group1_idx = np.where(years <= median_year)[0]
    group2_idx = np.where(years > median_year)[0]

    def group_metrics(indices: np.ndarray, name: str) -> Dict[str, float]:
        mpg_mean, mpg_std, mpg_n = compute_group_stats(arr, indices, 7)
        weight_mean, weight_std, weight_n = compute_group_stats(arr, indices, 5)
        return {
            "mpg_mean": mpg_mean,
            "mpg_std": mpg_std,
            "mpg_n": mpg_n,
            "weight_mean": weight_mean,
            "weight_std": weight_std,
            "weight_n": weight_n,
        }

    g1 = group_metrics(group1_idx, "Group 1 (Older)")
    g2 = group_metrics(group2_idx, "Group 2 (Newer)")

    print(f"Median split year: {median_year:.1f}")
    print("\nGroup statistics:")
    print(f"  Group 1 (<= {median_year:.0f}):")
    print(f"    MPG mean = {g1['mpg_mean']:.2f} (n={g1['mpg_n']}), std={g1['mpg_std']:.2f}")
    print(f"    Weight mean = {g1['weight_mean']:.1f} lbs (n={g1['weight_n']}), std={g1['weight_std']:.1f}")
    print(f"  Group 2 (> {median_year:.0f}):")
    print(f"    MPG mean = {g2['mpg_mean']:.2f} (n={g2['mpg_n']}), std={g2['mpg_std']:.2f}")
    print(f"    Weight mean = {g2['weight_mean']:.1f} lbs (n={g2['weight_n']}), std={g2['weight_std']:.1f}")

    # Compute differences and standard errors
    mpg_diff = g2['mpg_mean'] - g1['mpg_mean']
    mpg_se = np.sqrt((g1['mpg_std'] ** 2 / g1['mpg_n']) + (g2['mpg_std'] ** 2 / g2['mpg_n']))
    weight_diff = g2['weight_mean'] - g1['weight_mean']
    weight_se = np.sqrt((g1['weight_std'] ** 2 / g1['weight_n']) + (g2['weight_std'] ** 2 / g2['weight_n']))

    print("\nDifferences between groups (Newer - Older):")
    print(f"  MPG difference = {mpg_diff:.2f} mpg (SE = {mpg_se:.2f})")
    print(f"  Weight difference = {weight_diff:.1f} lbs (SE = {weight_se:.1f})")

    # Decision rule: difference > 2 * SE indicates meaningful difference
    mpg_reliable = mpg_se != 0 and mpg_diff > 2 * mpg_se
    weight_reliable = weight_se != 0 and weight_diff > 2 * weight_se

    print("\nInterpretation:")
    if mpg_reliable:
        print("  Fuel efficiency differs meaningfully between the two year ranges.")
    else:
        print("  Fuel efficiency difference is weak; evidence insufficient.")

    if weight_reliable:
        print("  Vehicle weight differs meaningfully between the two year ranges.")
    else:
        print("  Vehicle weight difference is weak; evidence insufficient.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-05 & CAR-06 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # CAR-05
    print_car05(data)

    # CAR-06
    print_car06(data)


if __name__ == "__main__":
    main()

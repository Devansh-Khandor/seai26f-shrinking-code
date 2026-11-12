#!/usr/bin/env python3
"""
skill_03.py

This script extends the existing NASA data analysis code to answer two
additional requests:

NASA-05: Compare projects with different programmer capability ratings
         (column 'apex') and determine if their development-effort
         differences are statistically significant.

NASA-06: Compare projects with different complexity ratings
         (column 'cplx') and determine if their defect outcomes are
         statistically significant.
"""

import sys
from typing import Dict, List, Tuple

import numpy as np

# Import helper functions from the earlier skills
from skill_01 import read_csv, extract_numeric_columns, compute_stats


def compute_group_stats(values: np.ndarray, groups: np.ndarray) -> Dict[str, Tuple[float, float, int]]:
    """
    For each unique group value, compute mean, standard deviation, and
    count of the provided numeric values that correspond to that group.

    Parameters
    ----------
    values : np.ndarray
        Numeric array (floats) containing the metric of interest
        (effort or defects). Missing values are np.nan.
    groups : np.ndarray
        Array of group labels (strings) corresponding to each value.

    Returns
    -------
    Dict[str, Tuple[float, float, int]]
        Mapping from group label to (mean, std, n). Values with no
        valid entries will have mean=np.nan, std=np.nan, n=0.
    """
    stats: Dict[str, Tuple[float, float, int]] = {}
    for group in np.unique(groups):
        mask = groups == group
        vals = values[mask]
        valid = ~np.isnan(vals)

        n = np.sum(valid)
        if n > 0:
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals, ddof=0)
        else:
            mean_val = np.nan
            std_val = np.nan

        stats[group] = (mean_val, std_val, int(n))
    return stats


def pairwise_z_test(stats: Dict[str, Tuple[float, float, int]], metric_name: str) -> List[str]:
    """
    Perform pairwise z-tests between all groups based on the provided
    means and standard deviations. Assumes unequal variances and uses
    the standard error of the difference.

    Parameters
    ----------
    stats : Dict[str, Tuple[float, float, int]]
        Group statistics as returned by ``compute_group_stats``.
    metric_name : str
        Name of the metric (used in output text).

    Returns
    -------
    List[str]
        Text lines describing each pairwise comparison.
    """
    results = []
    groups = sorted(stats.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            m1, s1, n1 = stats[g1]
            m2, s2, n2 = stats[g2]

            # Skip comparisons where we don't have enough data
            if np.isnan(m1) or np.isnan(m2) or n1 < 2 or n2 < 2:
                continue

            diff = m1 - m2
            se_diff = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
            if se_diff == 0:
                z = np.nan
                significant = False
            else:
                z = diff / se_diff
                significant = abs(z) > 1.96  # 5% two-sided

            results.append(
                f"  {metric_name} difference between '{g1}' and '{g2}': "
                f"diff={diff:.2f}, z={z:.2f}, significant={significant}"
            )
    return results


def main(csv_path: str):
    rows = read_csv(csv_path)

    # ---- NASA-05: Programmer capability (apex) vs effort --------------------------------
    apex_labels = np.array([row['apex'] for row in rows], dtype=str)
    effort_vals = extract_numeric_columns(rows, ['Effort-'])

    # Filter out missing effort values
    valid_mask = ~np.isnan(effort_vals)
    apex_labels = apex_labels[valid_mask]
    effort_vals = effort_vals[valid_mask]

    apex_stats = compute_group_stats(effort_vals, apex_labels)

    print("== NASA-05 ==")
    print("Developer effort (man-months) by programmer capability rating:")
    for rating, (mean, std, n) in sorted(apex_stats.items()):
        print(f"  Rating '{rating}': mean={mean:.2f}, std={std:.2f}, n={n}")

    print("\nPairwise comparisons:")
    for line in pairwise_z_test(apex_stats, "Effort"):
        print(line)

    # ---- NASA-06: Complexity (cplx) vs defects --------------------------------------------
    cplx_labels = np.array([row['cplx'] for row in rows], dtype=str)
    defect_vals = extract_numeric_columns(rows, ['Defects-'])

    # Filter out missing defect values
    valid_mask = ~np.isnan(defect_vals)
    cplx_labels = cplx_labels[valid_mask]
    defect_vals = defect_vals[valid_mask]

    cplx_stats = compute_group_stats(defect_vals, cplx_labels)

    print("\n== NASA-06 ==")
    print("Defect counts by complexity rating:")
    for rating, (mean, std, n) in sorted(cplx_stats.items()):
        print(f"  Rating '{rating}': mean={mean:.2f}, std={std:.2f}, n={n}")

    print("\nPairwise comparisons:")
    for line in pairwise_z_test(cplx_stats, "Defects"):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

#!/usr/bin/env python3
"""
skill_05.py

This script implements NASA-09 and NASA-10.

NASA-09: For projects that required unusually high development effort,
          identify which recorded project characteristics are most
          strongly associated with those outcomes.

NASA-10: Examine projects with unusually high defect counts and identify
          the characteristics that distinguish them from projects with
          better quality outcomes.

The script re‑uses helper functions from skill_01.py and performs a
simple association analysis:

* Numeric columns – Pearson correlation with Effort- or Defects-.
* Categorical columns – absolute difference between the mean of each
  category and the overall mean of the target metric.
"""

import sys
import numpy as np
from typing import List, Tuple, Dict

# Import helper functions from the previous skills
from skill_01 import read_csv, extract_numeric_columns, compute_stats


# ----------------------------------------------------------------------
# Helper: Correlation between numeric columns and a target column
# ----------------------------------------------------------------------
def numeric_correlations(rows: List[dict], target_col: str,
                         numeric_cols: List[str]) -> List[Tuple[str, float]]:
    """
    Compute Pearson correlation between each numeric column and the
    target column. NaNs are ignored in pairwise fashion.

    Returns a list of (column_name, correlation) sorted by abs(correlation)
    descending.
    """
    target = extract_numeric_columns(rows, [target_col])
    results = []

    for col in numeric_cols:
        if col == target_col:
            continue
        x = extract_numeric_columns(rows, [col])

        # Mask of rows where both values are not NaN
        mask = ~np.isnan(x) & ~np.isnan(target)
        if np.sum(mask) < 2:
            continue

        corr = np.corrcoef(x[mask], target[mask])[0, 1]
        results.append((col, corr))

    # Sort by absolute correlation descending
    results.sort(key=lambda t: abs(t[1]), reverse=True)
    return results


# ----------------------------------------------------------------------
# Helper: Categorical characteristic with largest mean difference
# ----------------------------------------------------------------------
def categorical_diff(rows: List[dict], target_col: str,
                     categorical_cols: List[str]) -> List[Tuple[str, float]]:
    """
    For each categorical column, compute the maximum absolute difference
    between the mean target value for any category and the overall mean.
    Returns a list of (column_name, max_diff) sorted by max_diff descending.
    """
    target = extract_numeric_columns(rows, [target_col])
    overall_mean = np.nanmean(target)
    results = []

    for col in categorical_cols:
        # Gather categories and their corresponding target values
        categories = [row[col] for row in rows]
        target_vals = target
        # Map each category to its target values
        unique_cats = set(categories)
        max_diff = 0.0
        for cat in unique_cats:
            cat_mask = np.array(categories) == cat
            cat_vals = target_vals[cat_mask]
            cat_valid = cat_vals[~np.isnan(cat_vals)]
            if cat_valid.size == 0:
                continue
            cat_mean = np.mean(cat_valid)
            diff = abs(cat_mean - overall_mean)
            if diff > max_diff:
                max_diff = diff
        results.append((col, max_diff))

    # Sort by max_diff descending
    results.sort(key=lambda t: t[1], reverse=True)
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str):
    rows = read_csv(csv_path)

    # List of all column names from the first row
    all_cols = rows[0].keys()

    # Define numeric and categorical columns
    numeric_cols = [col for col in all_cols
                    if col not in {'centerX', 'prec', 'flex', 'resl', 'team',
                                   'pmat', 'rely', 'data', 'cplx', 'ruse',
                                   'docu', 'time', 'stor', 'pvol', 'acap',
                                   'pcap', 'pcon', 'apex', 'plex', 'ltex',
                                   'tool', 'site', 'sced'}]
    categorical_cols = [col for col in all_cols
                        if col not in {'idX', 'YearX', 'Kloc+', 'Effort-',
                                       'Defects-', 'MonthsX'}]

    # NASA-09: High effort characteristics
    print("== NASA-09 ==")

    # Numeric correlations with Effort-
    num_corr_effort = numeric_correlations(rows, 'Effort-', numeric_cols)
    print("\nTop numeric characteristics correlated with high effort:")
    for i, (col, corr) in enumerate(num_corr_effort[:3], 1):
        print(f"  {i}. {col} : correlation = {corr:.3f}")

    # Categorical differences with Effort-
    cat_diff_effort = categorical_diff(rows, 'Effort-', categorical_cols)
    print("\nTop categorical characteristics with largest mean effort difference:")
    for i, (col, diff) in enumerate(cat_diff_effort[:3], 1):
        print(f"  {i}. {col} : max mean difference = {diff:.2f} months")

    # NASA-10: High defect characteristics
    print("\n== NASA-10 ==")

    # Numeric correlations with Defects-
    num_corr_defect = numeric_correlations(rows, 'Defects-', numeric_cols)
    print("\nTop numeric characteristics correlated with high defect count:")
    for i, (col, corr) in enumerate(num_corr_defect[:3], 1):
        print(f"  {i}. {col} : correlation = {corr:.3f}")

    # Categorical differences with Defects-
    cat_diff_defect = categorical_diff(rows, 'Defects-', categorical_cols)
    print("\nTop categorical characteristics with largest mean defect difference:")
    for i, (col, diff) in enumerate(cat_diff_defect[:3], 1):
        print(f"  {i}. {col} : max mean difference = {diff:.2f} defects")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

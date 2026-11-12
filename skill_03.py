#!/usr/bin/env python3
"""
Skill 03: Comparative Analysis (CAR-05 & CAR-06)

Usage:
    python skill_03.py <path_to_auto93.csv>
"""

import sys
import numpy as np


def load_data(csv_path):
    """
    Load CSV into structured NumPy array and cast numeric fields.
    """
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

    # Convert numeric columns to float
    # Some columns may be read as string if missing values
    cols_float = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                  'Lbs-', 'Acc+', 'Mpg+']
    for col in cols_float:
        if col in data.dtype.names:
            try:
                data[col] = data[col].astype(float)
            except Exception:
                # If conversion fails, keep original (likely strings)
                pass
    return data


def compute_stats(values):
    """Return mean, std dev, n for array of numbers."""
    vals = values[~np.isnan(values)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, 0
    mean = vals.mean()
    std = vals.std(ddof=1)  # sample std
    return mean, std, n


def pairwise_comparison(group1, group2, name1, name2, col_name='Mpg+'):
    """Compute mean difference and 95% CI for two groups."""
    mean1, std1, n1 = group1
    mean2, std2, n2 = group2

    diff = mean1 - mean2
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    significant = (ci_low > 0) or (ci_high < 0)

    return {
        'group1': name1,
        'group2': name2,
        'mean1': mean1,
        'mean2': mean2,
        'diff': diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': significant
    }


def car_05(data):
    """CAR-05: Compare fuel efficiency across cylinder counts."""
    # Group by cylinder count
    cyl_counts = np.unique(data['Clndrs'])
    groups = {}
    for cyl in cyl_counts:
        mask = data['Clndrs'] == cyl
        mpg_values = data['Mpg+'][mask]
        groups[cyl] = compute_stats(mpg_values)

    # Pairwise comparisons
    comparisons = []
    cyl_list = sorted(groups.keys())
    for i in range(len(cyl_list)):
        for j in range(i + 1, len(cyl_list)):
            name1 = f"{cyl_list[i]} cyl"
            name2 = f"{cyl_list[j]} cyl"
            comp = pairwise_comparison(groups[cyl_list[i]],
                                       groups[cyl_list[j]],
                                       name1, name2)
            comparisons.append(comp)

    # Print results
    print("== CAR-05 ==")
    for comp in comparisons:
        print(f"{comp['group1']} vs {comp['group2']}:")
        print(f"    Mean MPG: {comp['mean1']:.2f} vs {comp['mean2']:.2f}")
        print(f"    Difference (mean1 - mean2): {comp['diff']:.2f}")
        print(f"    95% CI of difference: [{comp['ci_low']:.2f}, {comp['ci_high']:.2f}]")
        conclusion = "Significant" if comp['significant'] else "Not significant"
        print(f"    Conclusion: {conclusion}")
        print()


def car_06(data):
    """CAR-06: Compare overall performance between two model-year ranges."""
    # Determine median year to split
    years = data['Model']
    median_year = np.median(years)

    # Group 1: <= median
    mask1 = years <= median_year
    mpg1 = data['Mpg+'][mask1]
    mpg2 = data['Mpg+'][~mask1]

    mean1, std1, n1 = compute_stats(mpg1)
    mean2, std2, n2 = compute_stats(mpg2)

    diff = mean1 - mean2
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    significant = (ci_low > 0) or (ci_high < 0)

    # Print results
    print("== CAR-06 ==")
    print(f"Model-year range 1 (<= {median_year:.1f}):")
    print(f"    Number of vehicles: {n1}")
    print(f"    Mean MPG: {mean1:.2f}")
    print(f"Model-year range 2 (> {median_year:.1f}):")
    print(f"    Number of vehicles: {n2}")
    print(f"    Mean MPG: {mean2:.2f}")
    print(f"Difference in mean MPG (range1 - range2): {diff:.2f}")
    print(f"95% CI of difference: [{ci_low:.2f}, {ci_high:.2f}]")
    conclusion = "Significant difference" if significant else "Evidence too weak"
    print(f"Conclusion: {conclusion}")


def main(csv_path):
    data = load_data(csv_path)
    car_05(data)
    car_06(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <path_to_auto93.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

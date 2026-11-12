#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-05 – Compare fuel efficiency (mpg) across cylinder counts and evaluate reliability.
    CAR-06 – Compare overall performance (mpg) of two model‑year ranges and judge significance.

Usage:
    python skill_03.py <path_to_auto93.csv>
"""

import sys
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Import helper functions from previous skills
from skill_01 import read_csv, format_float


# --------------------------------------------------------------------------- #
# Statistical utilities
# --------------------------------------------------------------------------- #

def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return mean and sample standard deviation of a list of numbers."""
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'))
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return (m, math.sqrt(var))


def welch_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """Return t-statistic and two‑tailed p‑value using Welch's t‑test."""
    n1, n2 = len(sample1), len(sample2)
    m1, s1 = mean_std(sample1)
    m2, s2 = mean_std(sample2)

    se = math.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    if se == 0:
        return (float('nan'), 1.0)

    t = (m1 - m2) / se

    # Welch–Satterthwaite approximation of degrees of freedom
    df_num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
    df_den = ((s1 ** 4) / (n1 ** 2 * (n1 - 1))) + ((s2 ** 4) / (n2 ** 2 * (n2 - 1))) if n1 > 1 and n2 > 1 else 1
    df = df_num / df_den if df_den != 0 else 1

    # Normal approximation for p‑value (valid for df>30; acceptable here)
    p = 1.0 - math.erf(abs(t) / math.sqrt(2))
    return (t, p)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # CAR-05: Cylinder‑count comparison
    # ----------------------------------------------------------------------- #
    print("== CAR-05 ==")

    # Group mpg values by cylinder count
    cyl_groups: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        if r['Clndrs'] is not None and r['Mpg+'] is not None:
            cyl_groups[int(r['Clndrs'])].append(r['Mpg+'])

    # Find the three most populated groups
    top_groups = sorted(cyl_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]
    if not top_groups:
        print("No data available for cylinder‑count analysis.")
    else:
        # Print group statistics
        for cl, mpg_vals in top_groups:
            mean_mpg, std_mpg = mean_std(mpg_vals)
            print(f"Cylinders={cl} | n={len(mpg_vals)} | mean mpg={format_float(mean_mpg)} | std={format_float(std_mpg)}")

        # Pairwise t‑tests among the top 3 groups
        print("\nPairwise t‑tests:")
        for i in range(len(top_groups)):
            for j in range(i + 1, len(top_groups)):
                cl_i, vals_i = top_groups[i]
                cl_j, vals_j = top_groups[j]
                t_stat, p_val = welch_t_test(vals_i, vals_j)
                reliable = "YES" if p_val < 0.05 else "NO"
                print(f"  {cl_i} vs {cl_j}: t={format_float(t_stat)}, p={format_float(p_val)} -> Reliable={reliable}")

    # ----------------------------------------------------------------------- #
    # CAR-06: Model‑year range comparison
    # ----------------------------------------------------------------------- #
    print("\n== CAR-06 ==")

    # Group mpg values by model year
    year_groups: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        if r['Model'] is not None and r['Mpg+'] is not None:
            year_groups[int(r['Model'])].append(r['Mpg+'])

    if len(year_groups) < 2:
        print("Not enough model‑year groups to compare.")
        return

    # Select two ranges: earliest 2 years vs latest 2 years
    sorted_years = sorted(year_groups.keys())
    early_years = sorted_years[:2]
    late_years = sorted_years[-2:]

    early_mpg = [v for y in early_years for v in year_groups[y]]
    late_mpg = [v for y in late_years for v in year_groups[y]]

    # Statistics
    early_mean, early_std = mean_std(early_mpg)
    late_mean, late_std = mean_std(late_mpg)
    print(f"Early years {early_years[0]}-{early_years[1]} | n={len(early_mpg)} | mean mpg={format_float(early_mean)} | std={format_float(early_std)}")
    print(f"Late years {late_years[0]}-{late_years[1]} | n={len(late_mpg)} | mean mpg={format_float(late_mean)} | std={format_float(late_std)}")

    # t‑test
    t_stat, p_val = welch_t_test(early_mpg, late_mpg)
    reliable = "YES" if p_val < 0.05 else "NO"
    print(f"\nPerformance difference t={format_float(t_stat)}, p={format_float(p_val)} -> Reliable={reliable}")


if __name__ == "__main__":
    main()

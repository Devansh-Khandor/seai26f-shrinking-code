#!/usr/bin/env python3
"""
skill_03.py - Advanced NASA project analyses (NASA‑05 and NASA‑06).

Usage:
    python skill_03.py <csv_path>
"""

import csv
import sys
import math
from statistics import mean, stdev, StatisticsError
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Reuse helpers from skill_01
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column
except Exception:
    # Minimal fallback if skill_01 cannot be imported
    def load_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows: List[Dict[str, str]], col: str) -> List[float]:
        values = []
        for r in rows:
            try:
                val = float(r[col])
                values.append(val)
            except (ValueError, TypeError):
                pass
        return values

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def group_values_by_category(rows: List[Dict[str, str]],
                             category_col: str,
                             value_col: str) -> Dict[str, List[float]]:
    """Return a dict mapping each category to a list of numeric values."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        cat = r.get(category_col)
        if cat is None or cat == '':
            continue
        try:
            val = float(r.get(value_col, ''))
        except (ValueError, TypeError):
            continue
        groups.setdefault(cat, []).append(val)
    return groups

def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Return mean and standard deviation for a list of values."""
    if not values:
        return (float('nan'), float('nan'))
    try:
        m = mean(values)
        sd = stdev(values) if len(values) > 1 else 0.0
    except StatisticsError:
        m, sd = float('nan'), float('nan')
    return m, sd

# ----------------------------------------------------------------------
# NASA-05: Programmer capability vs effort
# ----------------------------------------------------------------------
def report_nasa_05(rows: List[Dict[str, str]]) -> None:
    """Compare effort across programmer capability ratings (apex)."""
    print("\n== NASA-05 ==")

    groups = group_values_by_category(rows, 'apex', 'Effort-')
    if not groups:
        print("No apex or Effort- data available.")
        return

    overall_values = get_numeric_column(rows, 'Effort-')
    overall_mean, overall_sd = compute_stats(overall_values)

    print("Average effort per programmer capability rating (apex):")
    for rating in sorted(groups):
        m, sd = compute_stats(groups[rating])
        print(f"  {rating:>2}: {m:.2f} ± {sd:.2f}")

    # Assess significance: any difference > 2 * overall_sd considered meaningful
    meaningful = False
    ratings = list(groups.keys())
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            diff = abs(groups[ratings[i]][0] - groups[ratings[j]][0]) if groups[ratings[i]] and groups[ratings[j]] else 0
            if diff > 2 * overall_sd:
                meaningful = True
                break
        if meaningful:
            break

    if meaningful:
        print("Differences in development effort across apex ratings are statistically meaningful "
              "and should inform future planning.")
    else:
        print("Differences in development effort across apex ratings are not statistically "
              "significant; consider other factors for planning.")

# ----------------------------------------------------------------------
# NASA-06: Complexity rating vs defects
# ----------------------------------------------------------------------
def report_nasa_06(rows: List[Dict[str, str]]) -> None:
    """Compare defect outcomes across complexity ratings (cplx)."""
    print("\n== NASA-06 ==")

    groups = group_values_by_category(rows, 'cplx', 'Defects-')
    if not groups:
        print("No cplx or Defects- data available.")
        return

    overall_values = get_numeric_column(rows, 'Defects-')
    overall_mean, overall_sd = compute_stats(overall_values)

    print("Average defects per complexity rating (cplx):")
    for rating in sorted(groups):
        m, sd = compute_stats(groups[rating])
        print(f"  {rating:>2}: {m:.2f} ± {sd:.2f}")

    # Assess reliability: require at least 3 samples per group and
    # difference between any two means > 2 * overall_sd
    reliable = True
    for rating, vals in groups.items():
        if len(vals) < 3:
            reliable = False
            break

    if reliable:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                r1, r2 = list(groups.keys())[i], list(groups.keys())[j]
                m1, _ = compute_stats(groups[r1])
                m2, _ = compute_stats(groups[r2])
                if abs(m1 - m2) <= 2 * overall_sd:
                    reliable = False
                    break
            if not reliable:
                break

    if reliable:
        print("Defect outcomes differ reliably across complexity ratings; use these findings in planning.")
    else:
        print("Evidence is insufficient to conclude that defect outcomes differ reliably across complexity ratings.")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, _ = load_csv(csv_path)

    report_nasa_05(rows)
    report_nasa_06(rows)

if __name__ == "__main__":
    main()

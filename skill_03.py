#!/usr/bin/env python3
"""
skill_03.py - NASA-05 and NASA-06 analysis.

NASA-05: Compare development effort across different programmer capability ratings.
NASA-06: Compare defect outcomes across different complexity ratings.

The script prints a summary for each request with group statistics and
indicates whether differences are statistically meaningful.
"""

import csv
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

# Import helpers from skill_01
try:
    from skill_01 import load_data
except Exception as exc:
    raise RuntimeError("Could not import load_data from skill_01.py") from exc


def group_statistics(projects: List[Dict], group_field: str, metric_field: str) -> Dict[str, Tuple[float, float, int]]:
    """
    Compute mean, standard deviation, and count for each group value.
    """
    groups: Dict[str, List[float]] = {}
    for p in projects:
        key = p.get(group_field, "")
        try:
            val = float(p[metric_field])
        except (ValueError, TypeError):
            continue
        groups.setdefault(key, []).append(val)

    stats: Dict[str, Tuple[float, float, int]] = {}
    for key, vals in groups.items():
        n = len(vals)
        mean = statistics.mean(vals) if n > 0 else 0.0
        if n > 1:
            std = statistics.stdev(vals)
        else:
            std = 0.0
        stats[key] = (mean, std, n)
    return stats


def pairwise_significance(stats: Dict[str, Tuple[float, float, int]], alpha_factor: float = 2.0) -> List[Tuple[str, str, float, float, bool]]:
    """
    Compare all pairs of groups and decide if the mean difference is significant.
    Returns list of tuples (group1, group2, diff, se_diff, significant).
    """
    results = []
    keys = list(stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            mean1, std1, n1 = stats[k1]
            mean2, std2, n2 = stats[k2]
            diff = abs(mean1 - mean2)
            # Standard error of the difference
            se = (std1 ** 2 / n1 + std2 ** 2 / n2) ** 0.5 if (n1 + n2) > 0 else 0.0
            significant = diff > alpha_factor * se
            results.append((k1, k2, diff, se, significant))
    return results


def print_nasa_05(projects: List[Dict]) -> None:
    """Generate NASA-05 output."""
    group_field = "acap"
    metric_field = "Effort-"
    stats = group_statistics(projects, group_field, metric_field)
    sig_results = pairwise_significance(stats)

    print("== NASA-05 ==")
    print(f"Grouping by programmer capability ({group_field}):")
    for group, (mean, std, n) in sorted(stats.items()):
        print(f"  {group:>4}: mean effort={mean:7.1f} ±{std:5.1f}, n={n}")

    print("\nSignificant effort differences (alpha=2σ):")
    for g1, g2, diff, se, sig in sorted(sig_results):
        status = "✓ Significant" if sig else "✗ Not significant"
        print(f"  {g1:<4} vs {g2:<4}: diff={diff:6.1f} (se={se:5.1f}) → {status}")


def print_nasa_06(projects: List[Dict]) -> None:
    """Generate NASA-06 output."""
    group_field = "cplx"
    metric_field = "Defects-"
    stats = group_statistics(projects, group_field, metric_field)
    sig_results = pairwise_significance(stats)

    print("\n== NASA-06 ==")
    print(f"Grouping by complexity rating ({group_field}):")
    for group, (mean, std, n) in sorted(stats.items()):
        print(f"  {group:>4}: mean defects={mean:7.1f} ±{std:5.1f}, n={n}")

    print("\nSignificant defect differences (alpha=2σ):")
    for g1, g2, diff, se, sig in sorted(sig_results):
        status = "✓ Significant" if sig else "✗ Not significant"
        print(f"  {g1:<4} vs {g2:<4}: diff={diff:6.1f} (se={se:5.1f}) → {status}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)
    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    print_nasa_05(projects)
    print_nasa_06(projects)


if __name__ == "__main__":
    main()

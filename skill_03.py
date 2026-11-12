#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 03

This script compares configurations with:
  - KeepAlive enabled vs disabled
  - Access logging enabled vs disabled

It reports mean performance, standard error of the mean difference,
and a simple significance assessment based on a 95% confidence interval.

Usage:
    python skill_03.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import statistics
from typing import List, Dict, Tuple


# --------------------------------------------------------------------------- #
# 1. CSV parsing
# --------------------------------------------------------------------------- #
def parse_csv(path: str) -> Tuple[List[Dict[str, int]], List[int]]:
    """
    Reads the CSV file and returns:
        configs: list of configuration dictionaries (flag -> 0/1)
        performances: list of corresponding performance values (ms)
    """
    configs: List[Dict[str, int]] = []
    performances: List[int] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace and convert to int
            cfg = {k.strip(): int(v.strip()) for k, v in row.items() if k.strip() != "Performance-"}
            perf = int(row["Performance-"].strip())
            configs.append(cfg)
            performances.append(perf)

    return configs, performances


# --------------------------------------------------------------------------- #
# 2. Group statistics
# --------------------------------------------------------------------------- #
def group_by_flag(
    configs: List[Dict[str, int]], performances: List[int], flag: str
) -> Tuple[List[int], List[int]]:
    """
    Separate performances into two groups based on the value of `flag`.
    Returns (group_0_perfs, group_1_perfs).
    """
    group_0: List[int] = []
    group_1: List[int] = []

    for cfg, perf in zip(configs, performances):
        if cfg.get(flag, 0) == 0:
            group_0.append(perf)
        else:
            group_1.append(perf)

    return group_0, group_1


def mean_and_std(perfs: List[int]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of a list of numbers.
    Returns (mean, std). For a single element, std is 0.0.
    """
    if not perfs:
        return (0.0, 0.0)
    mean_val = statistics.mean(perfs)
    std_val = statistics.stdev(perfs) if len(perfs) > 1 else 0.0
    return mean_val, std_val


def significance(
    mean0: float, mean1: float, std0: float, std1: float, n0: int, n1: int
) -> Tuple[float, str]:
    """
    Compute the mean difference (1 - 0) and its standard error.
    If the difference is more than 2 * SE, return 'Significant'; otherwise 'Not significant'.
    """
    diff = mean1 - mean0
    # Standard error of difference
    se = ((std0**2) / n0 + (std1**2) / n1) ** 0.5 if n0 > 0 and n1 > 0 else 0.0
    sign = "Significant" if abs(diff) > 2 * se and se > 0 else "Not significant"
    return diff, sign


# --------------------------------------------------------------------------- #
# 3. Main driver
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances = parse_csv(csv_path)

    # ------------------------------------------------------------------ #
    # APACHE-05: KeepAlive comparison
    # ------------------------------------------------------------------ #
    ka0, ka1 = group_by_flag(configs, performances, "keepAlive")
    n0_ka, n1_ka = len(ka0), len(ka1)
    mean0_ka, std0_ka = mean_and_std(ka0)
    mean1_ka, std1_ka = mean_and_std(ka1)
    diff_ka, sign_ka = significance(mean0_ka, mean1_ka, std0_ka, std1_ka, n0_ka, n1_ka)

    print("== APACHE-05 ==")
    print(f"KeepAlive enabled: {n1_ka} configurations, mean performance {mean1_ka:.2f} ms, std {std1_ka:.2f} ms")
    print(f"KeepAlive disabled: {n0_ka} configurations, mean performance {mean0_ka:.2f} ms, std {std0_ka:.2f} ms")
    print(f"Mean difference (enabled - disabled): {diff_ka:.2f} ms")
    print(f"Assessment: {sign_ka}")

    # ------------------------------------------------------------------ #
    # APACHE-06: Access Log comparison
    # ------------------------------------------------------------------ #
    al0, al1 = group_by_flag(configs, performances, "accessLog")
    n0_al, n1_al = len(al0), len(al1)
    mean0_al, std0_al = mean_and_std(al0)
    mean1_al, std1_al = mean_and_std(al1)
    diff_al, sign_al = significance(mean0_al, mean1_al, std0_al, std1_al, n0_al, n1_al)

    print("\n== APACHE-06 ==")
    print(f"Access Log enabled: {n1_al} configurations, mean performance {mean1_al:.2f} ms, std {std1_al:.2f} ms")
    print(f"Access Log disabled: {n0_al} configurations, mean performance {mean0_al:.2f} ms, std {std0_al:.2f} ms")
    print(f"Mean difference (enabled - disabled): {diff_al:.2f} ms")
    print(f"Assessment: {sign_al}")


if __name__ == "__main__":
    main()

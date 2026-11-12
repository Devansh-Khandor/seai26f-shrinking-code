#!/usr/bin/env python3
"""
Skill 03: Comparative analysis of key Apache features.

Usage:
    python skill_03.py <csv_path>
"""

from pathlib import Path
import sys
import math

# Import existing utilities from skill_01 and skill_02
from skill_01 import (
    read_csv,
    compute_statistics,
    format_config,
    best_worst_configs,
    top_percentile_threshold,
    find_distinguishing_settings,
)
from skill_02 import (
    standard_deviation,
)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def group_by_feature(rows, feature, value):
    """
    Return the list of rows where rows[feature] == value.
    """
    return [r for r in rows if r.get(feature) == str(value)]


def group_stats(rows, feature, value):
    """
    Return statistics (count, mean, std) for a subgroup.
    """
    subgroup = group_by_feature(rows, feature, value)
    perf_vals = [float(r["Performance-"]) for r in subgroup]
    count = len(perf_vals)
    if count == 0:
        mean = std = 0.0
    else:
        mean, _, _, _ = compute_statistics(perf_vals)
        std = standard_deviation(perf_vals)
    return {
        "count": count,
        "mean": mean,
        "std": std,
    }


def mean_diff_and_z(stat1, stat2):
    """
    Return mean difference and z-score for two groups.
    """
    diff = stat1["mean"] - stat2["mean"]
    if stat1["count"] <= 1 or stat2["count"] <= 1:
        se = float("nan")
        z = float("nan")
    else:
        se = math.sqrt((stat1["std"] ** 2) / stat1["count"] +
                       (stat2["std"] ** 2) / stat2["count"])
        z = diff / se if se != 0 else float("nan")
    return diff, z


def significance(z_score, alpha=0.05):
    """
    Return whether the difference is statistically significant.
    """
    # Two‑tailed test
    z_critical = 1.96  # approximate for alpha=0.05
    return abs(z_score) >= z_critical


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, _ = read_csv(csv_path)

    # ==== APACHE-05 ====
    print("== APACHE-05 ==")

    keep_alive_yes = group_stats(rows, "keepAlive", "1")
    keep_alive_no = group_stats(rows, "keepAlive", "0")

    diff_ka, z_ka = mean_diff_and_z(keep_alive_yes, keep_alive_no)

    print(f"Configurations with KeepAlive=1: {keep_alive_yes['count']}")
    print(f"Mean performance: {keep_alive_yes['mean']:.2f} ms")
    print(f"Configurations with KeepAlive=0: {keep_alive_no['count']}")
    print(f"Mean performance: {keep_alive_no['mean']:.2f} ms")
    print(f"Mean difference (1 – 0): {diff_ka:.2f} ms")

    if math.isnan(z_ka):
        conclusion_ka = "Insufficient data to assess significance."
    else:
        conclusion_ka = (
            "Statistically significant difference" if significance(z_ka)
            else "No statistically significant difference"
        )
    print(f"Conclusion: {conclusion_ka}")
    print()

    # ==== APACHE-06 ====
    print("== APACHE-06 ==")

    access_log_yes = group_stats(rows, "accessLog", "1")
    access_log_no = group_stats(rows, "accessLog", "0")

    diff_al, z_al = mean_diff_and_z(access_log_yes, access_log_no)

    print(f"Configurations with accessLog=1: {access_log_yes['count']}")
    print(f"Mean performance: {access_log_yes['mean']:.2f} ms")
    print(f"Configurations with accessLog=0: {access_log_no['count']}")
    print(f"Mean performance: {access_log_no['mean']:.2f} ms")
    print(f"Mean difference (1 – 0): {diff_al:.2f} ms")

    if math.isnan(z_al):
        conclusion_al = "Insufficient data to assess significance."
    else:
        conclusion_al = (
            "Statistically significant difference" if significance(z_al)
            else "No statistically significant difference"
        )
    print(f"Conclusion: {conclusion_al}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
Skill 03 – Apache configuration comparison

This script extends the existing Apache analysis by providing:

* APACHE-05: Compare configurations with KeepAlive enabled and disabled,
  and judge whether the observed performance differences are large enough
  to be actionable.
* APACHE-06: Compare configurations with access logging enabled and disabled,
  and judge whether one group consistently performs better or whether the
  evidence is inconclusive.

The implementation re‑uses helper functions from skill_01.py and
skill_02.py where appropriate.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from skill_01 and skill_02
# ----------------------------------------------------------------------
try:
    from skill_01 import read_csv, find_performance_column
except Exception:
    # Minimal fallback implementation if skill_01 is not present
    import csv

    def read_csv(path: str):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = []
            for row in reader:
                int_row = {}
                for k, v in row.items():
                    try:
                        int_row[k] = int(v)
                    except ValueError:
                        int_row[k] = v
                rows.append(int_row)
        return header, rows

    def find_performance_column(header):
        for col in header:
            if col.lower().startswith("performance"):
                return col
        raise KeyError("No performance column found in header.")


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def group_stats(rows, group_col, perf_col):
    """
    Return a dictionary mapping each group value to its stats:
    {'n': count, 'mean': mean, 'std': std}
    """
    groups = {}
    for row in rows:
        key = row[group_col]
        val = row[perf_col]
        groups.setdefault(key, []).append(val)
    stats = {}
    for key, values in groups.items():
        values_arr = np.array(values)
        stats[key] = {
            "n": len(values),
            "mean": np.mean(values_arr),
            "std": np.std(values_arr, ddof=0),
        }
    return stats


def analyze_group_difference(stats, overall_std, group1, group2, label):
    """
    Compare two groups and print analysis with effect size.
    """
    s1 = stats.get(group1)
    s2 = stats.get(group2)

    print(f"== {label} ==")
    if s1 is None or s2 is None:
        print(f"Insufficient data: one of the groups ({group1}, {group2}) is missing.")
        return

    mean_diff = s2["mean"] - s1["mean"]  # positive means group2 is better (lower perf)
    abs_diff = abs(mean_diff)

    # Cohen's d effect size
    d = mean_diff / overall_std if overall_std != 0 else np.nan

    print(f"Group {group1} (n={s1['n']}): mean={s1['mean']:.0f}, std={s1['std']:.0f}")
    print(f"Group {group2} (n={s2['n']}): mean={s2['mean']:.0f}, std={s2['std']:.0f}")
    print(f"Mean performance difference ({group2} - {group1}): {mean_diff:.1f}")
    print(f"Cohen's d effect size: {d:.2f}")

    # Decision rule (qualitative)
    if np.isnan(d):
        conclusion = "Unable to compute effect size due to zero variance."
    elif abs(d) >= 0.8:
        conclusion = "Large effect – the difference is statistically and practically meaningful."
    elif abs(d) >= 0.5:
        conclusion = "Moderate effect – the difference may be useful for decision making."
    else:
        conclusion = "Small effect – the evidence is inconclusive."
    print(f"Conclusion: {conclusion}\n")


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)

    # Overall performance statistics
    perf_values = np.array([row[perf_col] for row in rows])
    overall_std = np.std(perf_values, ddof=0)

    # Analyze KeepAlive (0/1)
    keepalive_stats = group_stats(rows, "keepAlive", perf_col)
    analyze_group_difference(
        keepalive_stats, overall_std, 0, 1, "APACHE-05: KeepAlive Comparison"
    )

    # Analyze accessLog (0/1)
    accesslog_stats = group_stats(rows, "accessLog", perf_col)
    analyze_group_difference(
        accesslog_stats, overall_std, 0, 1, "APACHE-06: AccessLog Comparison"
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache configuration comparison skill."
    )
    parser.add_argument(
        "csv_path",
        help="Path to Apache_AllMeasurements.csv",
    )
    args = parser.parse_args()

    try:
        main(args.csv_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

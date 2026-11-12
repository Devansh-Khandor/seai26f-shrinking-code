#!/usr/bin/env python3
"""
Skill 05: Blame analysis – identify settings and combinations that are strongly
associated with poor Apache performance.

Usage:
    python skill_05.py <csv_path>
"""

import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter

# Re‑use helpers from earlier skills
from skill_01 import (
    read_csv,
    compute_statistics,
    top_percentile_threshold,
    format_config,
)
from skill_02 import standard_deviation


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def get_feature_names(rows):
    """Return list of all column names except 'Performance-'."""
    return [k for k in rows[0].keys() if k != "Performance-"]


def compute_setting_stats(rows, performance):
    """
    For each setting and each value (0/1) compute the mean performance.
    Returns a dict: setting -> {value: mean_performance}
    """
    setting_stats = {}
    feature_names = get_feature_names(rows)

    for setting in feature_names:
        stats = {}
        for val in ("0", "1"):
            # Filter rows with this value
            vals = [
                float(r["Performance-"])
                for r in rows
                if r[setting] == val
            ]
            if vals:
                mean_val = sum(vals) / len(vals)
            else:
                mean_val = 0.0
            stats[val] = mean_val
        setting_stats[setting] = stats

    return setting_stats


def find_poor_configs(rows, performance, threshold):
    """
    Return list of tuples (index, row, perf) for configurations
    whose performance <= threshold (bottom 10%).
    """
    return [(idx, row, perf)
            for idx, (row, perf) in enumerate(zip(rows, performance))
            if perf <= threshold]


def analyze_config(row, setting_stats, overall_mean, top_n=3):
    """
    For a given configuration row, find the settings that are associated
    with worse performance than the overall mean.

    Returns a list of tuples:
        (setting, value, diff, mean_of_value)
    sorted by descending diff.
    """
    diffs = []
    for setting, val in row.items():
        if setting == "Performance-":
            continue
        mean_of_val = setting_stats[setting][val]
        diff = mean_of_val - overall_mean
        if diff > 0:          # worse than overall
            diffs.append((setting, val, diff, mean_of_val))

    # Sort by diff descending
    diffs.sort(key=lambda x: x[2], reverse=True)
    return diffs[:top_n]


def compute_combo_stats(poor_rows, feature_names):
    """
    Compute frequency and mean performance for all 2‑feature combinations
    among the poorly performing configurations.

    Returns:
        combo_counts: Counter of combo tuple -> count
        combo_perf: dict of combo tuple -> mean performance
    """
    combo_counts = Counter()
    combo_perf_sum = defaultdict(float)
    combo_perf_cnt = defaultdict(int)

    for row in poor_rows:
        vals = [row[f] for f in feature_names]
        combs = combinations(zip(feature_names, vals), 2)
        for c in combs:
            combo = tuple(sorted(c))
            combo_counts[combo] += 1
            # add performance
            perf = float(row["Performance-"])
            combo_perf_sum[combo] += perf
            combo_perf_cnt[combo] += 1

    combo_perf = {c: combo_perf_sum[c] / combo_perf_cnt[c]
                  for c in combo_perf_sum}
    return combo_counts, combo_perf


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    overall_mean, _, _, _ = compute_statistics(performance)
    feature_names = get_feature_names(rows)

    # ---- APACHE-09 ----
    print("== APACHE-09 ==")
    # Bottom 10% threshold
    threshold = top_percentile_threshold(performance, percent=10)
    poor_configs = find_poor_configs(rows, performance, threshold)

    if not poor_configs:
        print("No poorly performing configurations detected.")
    else:
        setting_stats = compute_setting_stats(rows, performance)
        # Show first 5 poor configs as example
        print(f"Threshold for bottom 10%: {threshold:.2f} ms")
        print(f"Number of poorly performing configs: {len(poor_configs)}")
        print("\nSample analysis of first 5 poor configurations:")
        for idx, row, perf in poor_configs[:5]:
            analysis = analyze_config(row, setting_stats, overall_mean)
            print(f"\nConfig {idx+1} (Performance={perf:.2f} ms):")
            if not analysis:
                print("  No settings found to be associated with worse performance.")
                continue
            print("  Settings most strongly associated with poor performance:")
            for setting, val, diff, mean_of_val in analysis:
                print(f"   - {setting}={val}: "
                      f"mean={mean_of_val:.2f} ms (diff={diff:.2f} ms)")

    print()

    # ---- APACHE-10 ----
    print("== APACHE-10 ==")
    # Prepare poor rows for combo analysis
    poor_rows = [row for _, row, _ in poor_configs]
    if not poor_rows:
        print("No poorly performing configurations to analyze combinations.")
        return

    combo_counts, combo_perf = compute_combo_stats(poor_rows, feature_names)

    # Filter combos that occur at least twice
    frequent_combos = [(c, combo_counts[c], combo_perf[c])
                       for c in combo_counts
                       if combo_counts[c] >= 2]

    if not frequent_combos:
        print("No combinations appear frequently among poorly performing configs.")
        return

    # Compute difference from overall mean
    combo_stats = []
    for combo, count, mean_perf in frequent_combos:
        diff = mean_perf - overall_mean
        combo_stats.append((combo, count, mean_perf, diff))

    # Sort by diff descending
    combo_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"Overall mean performance: {overall_mean:.2f} ms")
    print("\nTop combinations among poorly performing configurations:")
    for combo, count, mean_perf, diff in combo_stats[:5]:
        combo_str = ", ".join(f"{k}={v}" for k, v in combo)
        print(f"  - {combo_str}: "
              f"occurs {count}×, mean={mean_perf:.2f} ms "
              f"(diff={diff:.2f} ms)")

    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

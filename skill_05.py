#!/usr/bin/env python3
"""
skill_05.py

Implements the following Apache configuration analyses:

APACHE-09: For a poorly performing configuration, identify which of its enabled
           or disabled settings are most strongly associated with unfavorable
           performance in historical data.

APACHE-10: Identify combinations of server settings that frequently appear
           among poorly performing configurations and show how they differ
           from better‑performing combinations.

The script loads the CSV data, performs the two analyses, and prints
the results with the required header lines.
"""

import sys
import math
import itertools

# Reuse helper functions from earlier skills
from skill_01 import read_configurations, performance_stats


def compute_setting_stats(configs):
    """
    For each setting, compute the mean performance when enabled and disabled,
    and the difference (enabled_mean - disabled_mean).

    Returns a dict mapping setting name to a tuple:
        (mean_enabled, mean_disabled, diff)
    """
    setting_keys = [k for k in configs[0].keys() if k != 'Performance-']
    stats = {}

    for key in setting_keys:
        sum_enabled = 0.0
        cnt_enabled = 0
        sum_disabled = 0.0
        cnt_disabled = 0

        for cfg in configs:
            val = cfg[key]
            perf = cfg['Performance-']
            if val == 1.0:
                sum_enabled += perf
                cnt_enabled += 1
            else:
                sum_disabled += perf
                cnt_disabled += 1

        mean_enabled = sum_enabled / cnt_enabled if cnt_enabled else None
        mean_disabled = sum_disabled / cnt_disabled if cnt_disabled else None
        diff = (mean_enabled - mean_disabled) if (mean_enabled is not None and mean_disabled is not None) else None

        stats[key] = (mean_enabled, mean_disabled, diff)

    return stats


def report_apache_09(configs):
    """
    Identify the most strongly negatively associated settings with
    poor performance and report their states in the worst configuration.
    """
    print("== APACHE-09 ==")

    setting_stats = compute_setting_stats(configs)
    # Filter out settings without valid diff
    filtered = [(k, v[2]) for k, v in setting_stats.items() if v[2] is not None]

    if not filtered:
        print("No valid setting statistics available.")
        return

    # Sort by absolute difference descending (most impactful)
    top_settings = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:3]

    # Find the worst configuration
    worst_cfg = min(configs, key=lambda c: c['Performance-'])
    worst_perf = worst_cfg['Performance-']

    for setting, diff in top_settings:
        val = worst_cfg[setting]
        # Determine which side is bad
        if diff < 0:
            bad_state = 1  # enabled is bad
        elif diff > 0:
            bad_state = 0  # disabled is bad
        else:
            bad_state = None

        state_str = "enabled" if val == 1.0 else "disabled"
        if bad_state is None:
            match = " (no clear bad side)"
        else:
            match = "" if val == bad_state else " (matches not bad)"

        print(f"Setting: {setting}, Config state: {state_str}, Difference: {diff:.2f}{match}")

    print()


def compute_pair_counts_and_means(configs, poor_indices):
    """
    For each pair of settings that are enabled together in a poor configuration,
    count occurrences and compute the mean performance for configs where both
    are enabled.
    """
    setting_keys = [k for k in configs[0].keys() if k != 'Performance-']

    pair_counts = {}
    pair_perf_sums = {}
    pair_perf_counts = {}

    for idx in poor_indices:
        cfg = configs[idx]
        # list of enabled settings in this config
        enabled = [k for k in setting_keys if cfg[k] == 1.0]
        # generate all 2‑item combos
        for a, b in itertools.combinations(sorted(enabled), 2):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            # accumulate performance for this pair
            pair_perf_sums[pair] = pair_perf_sums.get(pair, 0.0) + cfg['Performance-']
            pair_perf_counts[pair] = pair_perf_counts.get(pair, 0) + 1

    # Compute mean performance for each pair
    pair_means = {pair: pair_perf_sums[pair] / pair_perf_counts[pair]
                  for pair in pair_counts}

    return pair_counts, pair_means


def report_apache_10(configs):
    """
    Identify frequently occurring combinations of enabled settings among
    poorly performing configurations and compare their mean performance
    to the overall mean.
    """
    print("== APACHE-10 ==")

    # Determine overall mean and stddev
    mean, _, _, stddev = performance_stats(configs)

    # Define poor threshold: mean - 1.5 * stddev
    threshold = mean - 1.5 * stddev

    poor_indices = [i for i, cfg in enumerate(configs) if cfg['Performance-'] <= threshold]

    if not poor_indices:
        print("No poorly performing configurations found.")
        print()
        return

    pair_counts, pair_means = compute_pair_counts_and_means(configs, poor_indices)

    # Sort pairs by count descending, take top 3
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    if not top_pairs:
        print("No frequent pairs found among poor configurations.")
        print()
        return

    for pair, count in top_pairs:
        mean_perf_pair = pair_means[pair]
        diff_from_overall = mean_perf_pair - mean
        a, b = pair
        print(f"Combination: {a}, {b} (count in poor configs: {count})")
        print(f"  Mean performance when both enabled: {mean_perf_pair:.2f}")
        print(f"  Difference from overall mean: {diff_from_overall:.2f}")

    print()


def main(csv_path):
    configs = read_configurations(csv_path)

    # Run the requested analyses
    report_apache_09(configs)
    report_apache_10(configs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

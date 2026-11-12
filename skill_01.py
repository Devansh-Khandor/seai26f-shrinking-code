#!/usr/bin/env python3
"""
Apache Server Configuration Analysis

Usage:
    python skill_01.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import statistics


def parse_csv(path: str):
    """Parse the CSV and return a list of configuration dicts and performance values."""
    configs = []
    performances = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert all values to integers and strip whitespace from keys
            config = {k.strip(): int(v.strip()) for k, v in row.items() if k.strip() != 'Performance-'}
            perf = int(row['Performance-'].strip())
            configs.append(config)
            performances.append(perf)

    return configs, performances


def summarize_performance(performances):
    """Return typical, best, and worst performance values."""
    typical = statistics.mean(performances)
    best = max(performances)
    worst = min(performances)
    return typical, best, worst


def find_favorable_configs(configs, performances, typical, stdev):
    """Identify configurations with unusually favorable performance."""
    threshold = typical + stdev  # values better than mean + 1 std dev
    fav = [c for c, p in zip(configs, performances) if p > threshold]

    if not fav:
        # Fallback: top 5% of configurations
        sorted_indices = sorted(range(len(performances)), key=lambda i: performances[i], reverse=True)
        n_top = max(1, len(performances) // 20)  # at least one
        top_indices = sorted_indices[:n_top]
        fav = [configs[i] for i in top_indices]

    return fav


def distinguishing_settings(configs, fav_configs):
    """Determine which settings are more prevalent in favorable configurations."""
    flags = configs[0].keys()
    overall_means = {flag: statistics.mean([c[flag] for c in configs]) for flag in flags}
    fav_means = {flag: statistics.mean([c[flag] for c in fav_configs]) for flag in flags}

    # Consider a setting distinguishing if the difference is > 0.1
    distinguishing = [
        flag
        for flag in flags
        if abs(fav_means[flag] - overall_means[flag]) > 0.1
    ]

    return distinguishing, fav_means, overall_means


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances = parse_csv(csv_path)

    typical, best, worst = summarize_performance(performances)
    stdev = statistics.stdev(performances) if len(performances) > 1 else 0

    # APACHE-01
    print("== APACHE-01 ==")
    print(f"Typical performance: {typical:.2f} ms")
    print(f"Best performance: {best} ms")
    print(f"Worst performance: {worst} ms")

    # APACHE-02
    fav_configs = find_favorable_configs(configs, performances, typical, stdev)
    distinguishing, fav_means, overall_means = distinguishing_settings(configs, fav_configs)

    print("\n== APACHE-02 ==")
    print(f"Number of configurations with unusually favorable performance: {len(fav_configs)}")
    print(f"Sample favorable configuration: {fav_configs[0]}")
    print("Distinguishing settings:")
    for flag in distinguishing:
        print(f"  {flag} (fav: {fav_means[flag]:.2f}, overall: {overall_means[flag]:.2f})")


if __name__ == "__main__":
    main()

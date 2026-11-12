#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 02

This script implements two additional analysis requests for the client:

  APACHE-03: Estimate the likely performance of a configuration before benchmarking.
  APACHE-04: Estimate the potential performance difference between two historically
             observed configurations that differ by a single setting, along with
             an uncertainty estimate.

The analysis uses a simple additive model derived from historical data.
"""

import csv
import sys
import statistics
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# 1. CSV Parsing
# --------------------------------------------------------------------------- #

def parse_csv(path: str) -> Tuple[List[Dict[str, int]], List[int], List[str]]:
    """
    Read the CSV file and return:
      - configs: list of configuration dicts (flag -> 0/1)
      - performances: list of corresponding performance values (ms)
      - flags: list of flag names (ordered as in the file)
    """
    configs = []
    performances = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        flags = [k.strip() for k in reader.fieldnames if k.strip() != 'Performance-']
        for row in reader:
            config = {k.strip(): int(v.strip()) for k, v in row.items() if k.strip() != 'Performance-'}
            perf = int(row['Performance-'].strip())
            configs.append(config)
            performances.append(perf)

    return configs, performances, flags


# --------------------------------------------------------------------------- #
# 2. Additive model building
# --------------------------------------------------------------------------- #

def build_additive_model(
    configs: List[Dict[str, int]],
    performances: List[int],
    flags: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the baseline mean performance and per‑flag marginal effects.
    Returns (baseline_mean, effects_dict)
    """
    baseline_mean = statistics.mean(performances)
    # For each flag, compute average performance when flag==1 and flag==0
    effects = {}
    for flag in flags:
        values_1 = [perf for cfg, perf in zip(configs, performances) if cfg[flag] == 1]
        values_0 = [perf for cfg, perf in zip(configs, performances) if cfg[flag] == 0]
        if not values_1 or not values_0:
            # If a flag never varies in the data, assume no effect
            effects[flag] = 0.0
        else:
            mean1 = statistics.mean(values_1)
            mean0 = statistics.mean(values_0)
            effects[flag] = mean1 - mean0
    return baseline_mean, effects


def predict_performance(
    config: Dict[str, int],
    baseline: float,
    effects: Dict[str, float],
) -> float:
    """Predict performance for a single configuration using the additive model."""
    prediction = baseline
    for flag, effect in effects.items():
        prediction += effect * config.get(flag, 0)
    return prediction


# --------------------------------------------------------------------------- #
# 3. Difference statistics for APACHE‑04
# --------------------------------------------------------------------------- #

def compute_difference_statistics(
    configs: List[Dict[str, int]],
    performances: List[int],
    flags: List[str],
) -> Dict[str, Tuple[float, float]]:
    """
    For each flag, compute the mean and standard deviation of performance differences
    between all pairs of configurations that differ only in that flag.
    Returns: dict flag -> (mean_diff, std_diff)
    """
    diff_stats: Dict[str, List[float]] = {flag: [] for flag in flags}

    # For each flag, group configurations by the values of all other flags
    for flag in flags:
        groups: Dict[Tuple[Tuple[str, int], ...], List[int]] = {}
        for cfg, perf in zip(configs, performances):
            key = tuple((f, cfg[f]) for f in flags if f != flag)
            groups.setdefault(key, []).append(perf)

        # Within each group, every pair differs only in `flag`
        for perf_list in groups.values():
            if len(perf_list) < 2:
                continue
            # Compute all pairwise differences
            for i in range(len(perf_list)):
                for j in range(i + 1, len(perf_list)):
                    diff = perf_list[i] - perf_list[j]
                    diff_stats[flag].append(diff)

    # Convert to (mean, std) tuples
    result: Dict[str, Tuple[float, float]] = {}
    for flag, diffs in diff_stats.items():
        if not diffs:
            continue
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        result[flag] = (mean_diff, std_diff)

    return result


# --------------------------------------------------------------------------- #
# 4. Main driver
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    # Build additive model
    baseline, effects = build_additive_model(configs, performances, flags)

    # ------------------------------------------------------------------ #
    # APACHE-03: Estimate performance
    # ------------------------------------------------------------------ #
    predictions = [predict_performance(cfg, baseline, effects) for cfg in configs]
    mae = statistics.mean([abs(p - a) for p, a in zip(predictions, performances)])
    rmse = (statistics.mean([(p - a) ** 2 for p, a in zip(predictions, performances)])) ** 0.5

    print("== APACHE-03 ==")
    print(f"Baseline mean performance: {baseline:.2f} ms")
    print(f"Mean absolute error of additive model: {mae:.2f} ms")
    print(f"Root mean squared error: {rmse:.2f} ms")
    print("Example estimate:")
    example_cfg = configs[0]
    example_pred = predictions[0]
    print(f"  Config: {example_cfg}")
    print(f"  Predicted: {example_pred:.2f} ms, Actual: {performances[0]} ms")
    print()

    # ------------------------------------------------------------------ #
    # APACHE-04: Estimate difference for single‑setting changes
    # ------------------------------------------------------------------ #
    diff_stats = compute_difference_statistics(configs, performances, flags)

    print("== APACHE-04 ==")
    if diff_stats:
        print("Estimated performance difference between configurations differing by one setting:")
        for flag, (mean_diff, std_diff) in diff_stats.items():
            print(f"  {flag}: mean difference {mean_diff:.2f} ms, std dev {std_diff:.2f} ms")
    else:
        print("  No settings exhibit variations in the data.")


if __name__ == "__main__":
    main()

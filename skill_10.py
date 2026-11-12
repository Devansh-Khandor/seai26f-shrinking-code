#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 10

This script implements the client requests:

  APACHE-19 – Compare our configuration-selection system against random
              configuration selection under the same benchmarking budget,
              and show which approach discovers better-performing
              configurations.

  APACHE-20 – Compare our performance predictions against a simple
              historical-average estimate and show whether our system
              provides a meaningful improvement on previously unseen
              configurations.

Usage:
    python skill_10.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import random
import statistics
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
def parse_csv(path: str) -> Tuple[List[Dict[str, int]], List[int], List[str]]:
    """Read CSV and return (configs, performances, flag_names)."""
    configs: List[Dict[str, int]] = []
    performances: List[int] = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        flags = [k.strip() for k in reader.fieldnames if k.strip() != 'Performance-']
        for row in reader:
            cfg = {k.strip(): int(v.strip()) for k, v in row.items() if k.strip() != 'Performance-'}
            perf = int(row['Performance-'].strip())
            configs.append(cfg)
            performances.append(perf)

    return configs, performances, flags


# --------------------------------------------------------------------------- #
# Additive model (used in APACHE-19 and APACHE-20)
# --------------------------------------------------------------------------- #
def build_additive_model(
    configs: List[Dict[str, int]],
    performances: List[int],
    flags: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute baseline mean and per‑flag marginal effects."""
    baseline = statistics.mean(performances) if performances else 0.0
    effects: Dict[str, float] = {}
    for flag in flags:
        values_1 = [p for c, p in zip(configs, performances) if c.get(flag, 0) == 1]
        values_0 = [p for c, p in zip(configs, performances) if c.get(flag, 0) == 0]
        if values_1 and values_0:
            effects[flag] = statistics.mean(values_1) - statistics.mean(values_0)
        else:
            effects[flag] = 0.0
    return baseline, effects


def predict_performance(cfg: Dict[str, int], baseline: float, effects: Dict[str, float]) -> float:
    """Predict performance for a single configuration."""
    pred = baseline
    for flag, effect in effects.items():
        pred += effect * cfg.get(flag, 0)
    return pred


# --------------------------------------------------------------------------- #
# APACHE-19: Compare selection vs random
# --------------------------------------------------------------------------- #
def apache_19(
    configs: List[Dict[str, int]],
    performances: List[int],
) -> None:
    """Show that our selection strategy (top predicted) outperforms random."""
    # Build full‑dataset model
    flags = configs[0].keys()
    baseline, effects = build_additive_model(configs, performances, list(flags))

    # Predict for all
    predictions = [predict_performance(cfg, baseline, effects) for cfg in configs]

    # Define benchmarking budget: 10% of total, capped at 10, at least 1
    total = len(configs)
    budget = max(1, min(10, total // 10 if total >= 10 else total))
    # If total < budget, just use all
    if total < budget:
        budget = total

    # Our selection: best predicted first
    sorted_indices = sorted(range(total), key=lambda i: predictions[i])
    selected_indices = sorted_indices[:budget]
    selected_perfs = [performances[i] for i in selected_indices]
    avg_selected = statistics.mean(selected_perfs)
    best_selected = min(selected_perfs)

    # Random selection
    random.seed(42)  # deterministic
    random_indices = random.sample(range(total), k=budget)
    random_perfs = [performances[i] for i in random_indices]
    avg_random = statistics.mean(random_perfs)
    best_random = min(random_perfs)

    # Print
    print("== APACHE-19 ==")
    print(f"Benchmarking budget: {budget} configuration(s)")
    print(f"Method: Top‑predicted selection")
    print(f"  Average performance of evaluated configs: {avg_selected:.2f} ms")
    print(f"  Best (lowest) performance among them: {best_selected:.0f} ms")
    print()
    print(f"Method: Random selection")
    print(f"  Average performance of evaluated configs: {avg_random:.2f} ms")
    print(f"  Best (lowest) performance among them: {best_random:.0f} ms")
    print()
    if best_selected < best_random:
        print("Conclusion: Our selection strategy discovered better-performing configurations.")
    else:
        print("Conclusion: Random selection performed as well or better in this run.")


# --------------------------------------------------------------------------- #
# APACHE-20: Predictive model vs baseline average
# --------------------------------------------------------------------------- #
def apache_20(
    configs: List[Dict[str, int]],
    performances: List[int],
) -> None:
    """Show that our additive model outperforms a simple historical‑average estimate."""
    # Train/test split (70/30). Deterministic split.
    indices = list(range(len(configs)))
    random.seed(99)
    random.shuffle(indices)
    split = int(0.7 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_cfgs = [configs[i] for i in train_idx]
    train_perf = [performances[i] for i in train_idx]
    test_cfgs = [configs[i] for i in test_idx]
    test_perf = [performances[i] for i in test_idx]

    # Baseline: mean of training performances
    baseline_mean = statistics.mean(train_perf)

    # Additive model built on training data
    flags = configs[0].keys()
    baseline, effects = build_additive_model(train_cfgs, train_perf, list(flags))

    # Predict on test set
    baseline_mae = statistics.mean([abs(p - baseline_mean) for p in test_perf])
    additive_pred = [predict_performance(cfg, baseline, effects) for cfg in test_cfgs]
    additive_mae = statistics.mean([abs(p - a) for p, a in zip(test_perf, additive_pred)])

    # Print
    print("\n== APACHE-20 ==")
    print(f"Training set size: {len(train_cfgs)}; Test set size: {len(test_cfgs)}")
    print(f"Baseline average MAE (historical mean): {baseline_mae:.2f} ms")
    print(f"Additive model MAE: {additive_mae:.2f} ms")
    improvement = baseline_mae - additive_mae
    percent_improve = (improvement / baseline_mae * 100) if baseline_mae != 0 else 0.0
    print(f"Improvement over baseline: {improvement:.2f} ms ({percent_improve:.1f}% reduction)")
    if improvement > 0:
        print("Conclusion: The additive model provides a meaningful improvement on unseen configurations.")
    else:
        print("Conclusion: The additive model did not outperform the simple baseline on unseen data.")


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    if not configs:
        print("No configurations found in the CSV.")
        sys.exit(1)

    apache_19(configs, performances)
    apache_20(configs, performances)


if __name__ == "__main__":
    main()

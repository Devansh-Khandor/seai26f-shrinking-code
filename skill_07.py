#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 07

This script implements two new analysis requests:

  APACHE-13 – Help find a low‑performance configuration while evaluating as few
              candidates as reasonably possible.
  APACHE-14 – With a limited benchmarking budget, recommend the order of
              configurations to test so promising settings are discovered early.

The approach is simple yet effective: build a lightweight additive performance
model from the historical data, then rank all configurations by their
predicted performance.  The best (lowest) predictions are likely to be
low‑performance configurations, so they can be tested first.
"""

import csv
import sys
import statistics
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# 1. CSV parsing
# --------------------------------------------------------------------------- #
def parse_csv(path: str) -> Tuple[List[Dict[str, int]], List[int], List[str]]:
    """Read CSV and return list of config dicts, performances, and flag names."""
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
# 2. Additive model
# --------------------------------------------------------------------------- #
def build_additive_model(
    configs: List[Dict[str, int]],
    performances: List[int],
    flags: List[str],
) -> Tuple[float, Dict[str, float]]:
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
# 3. Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    if not configs:
        print("No configurations found in the CSV.")
        sys.exit(1)

    baseline, effects = build_additive_model(configs, performances, flags)
    mean_perf = statistics.mean(performances)
    std_perf = statistics.stdev(performances) if len(performances) > 1 else 0.0

    # Predict performance for all configs
    predictions = [(idx, predict_performance(cfg, baseline, effects)) for idx, cfg in enumerate(configs)]

    # Sort by predicted performance ascending (lower is better)
    predictions.sort(key=lambda x: x[1])

    # ------------------------------------------------------------------ #
    # APACHE-13: Find a low‑performance configuration with minimal evaluations
    # ------------------------------------------------------------------ #
    print("== APACHE-13 ==")
    # We only evaluate the very best predicted configuration
    best_idx, best_pred = predictions[0]
    best_cfg = configs[best_idx]
    actual_perf = performances[best_idx]
    print(f"  Best predicted configuration to evaluate (1 candidate):")
    print(f"    Config: {best_cfg}")
    print(f"    Predicted performance: {best_pred:.2f} ms")
    print(f"    Actual performance (for reference): {actual_perf} ms")
    print(f"  Mean historical performance: {mean_perf:.2f} ms (std: {std_perf:.2f} ms)")

    # ------------------------------------------------------------------ #
    # APACHE-14: Recommend first configurations to test with limited budget
    # ------------------------------------------------------------------ #
    print("\n== APACHE-14 ==")
    # Define a reasonable budget: min(10% of total, 10) configurations
    budget = max(1, min(10, len(configs) // 10))
    print(f"  Limited benchmarking budget: evaluate {budget} configurations")
    print("  Recommended order (best predicted first):")
    for rank, (idx, pred) in enumerate(predictions[:budget], start=1):
        cfg = configs[idx]
        actual = performances[idx]
        improvement = mean_perf - pred
        print(f"    {rank}. Config: {cfg}")
        print(f"       Predicted: {pred:.2f} ms   (improvement: {improvement:.2f} ms over mean)")
        print(f"       Actual: {actual} ms")
    print(f"  Mean historical performance: {mean_perf:.2f} ms (std: {std_perf:.2f} ms)")

    # End of script


if __name__ == "__main__":
    main()

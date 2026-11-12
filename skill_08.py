#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 08

This script implements two new analysis requests for the client:

  APACHE-15 – For a poorly performing configuration, recommend a small change to its
              adjustable settings that could lead to a lower predicted performance
              value, using historically observed valid configurations as alternatives.

  APACHE-16 – Find the smallest number of configuration‑setting changes that could
              bring a poorly performing configuration closer to a specified
              performance target, while avoiding unsupported or invalid combinations.

Usage:
    python skill_08.py <path_to_Apache_AllMeasurements.csv>
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
# 2. Additive performance model
# --------------------------------------------------------------------------- #
def build_additive_model(
    configs: List[Dict[str, int]],
    performances: List[int],
    flags: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute baseline mean performance and per‑flag marginal effects."""
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
# 3. Utility helpers
# --------------------------------------------------------------------------- #
def hamming_distance(cfg1: Dict[str, int], cfg2: Dict[str, int], flags: List[str]) -> int:
    """Return number of differing flags between two configurations."""
    return sum(1 for f in flags if cfg1.get(f, 0) != cfg2.get(f, 0))


def format_config(cfg: Dict[str, int]) -> str:
    """Return a concise string representation of a configuration."""
    return "{" + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) + "}"


# --------------------------------------------------------------------------- #
# 4. Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    if not configs:
        print("No configurations found in the CSV.")
        sys.exit(1)

    # Build additive model once
    baseline, effects = build_additive_model(configs, performances, flags)

    # Identify the worst configuration (most poorly performing)
    worst_idx = max(range(len(performances)), key=lambda i: performances[i])
    worst_cfg = configs[worst_idx]
    worst_perf = performances[worst_idx]
    worst_pred = predict_performance(worst_cfg, baseline, effects)

    # ------------------------------------------------------------------ #
    # APACHE-15
    # ------------------------------------------------------------------ #
    print("== APACHE-15 ==")
    print(f"Worst configuration (actual performance: {worst_perf} ms):")
    print(f"  {format_config(worst_cfg)}")
    # Find all configurations that differ by exactly one setting
    single_step_cfgs = []
    for idx, cfg in enumerate(configs):
        if idx == worst_idx:
            continue
        if hamming_distance(worst_cfg, cfg, flags) == 1:
            single_step_cfgs.append((idx, cfg))

    # Evaluate their predicted performances
    better_candidates = []
    for idx, cfg in single_step_cfgs:
        pred = predict_performance(cfg, baseline, effects)
        if pred < worst_pred:
            better_candidates.append((idx, cfg, pred))

    if better_candidates:
        # Choose the one with the largest improvement
        best_idx, best_cfg, best_pred = max(better_candidates, key=lambda x: best_pred - x[2])
        # Determine which flag changed
        changed_flag = next(
            f for f in flags if best_cfg.get(f, 0) != worst_cfg.get(f, 0)
        )
        print(f"Suggested change: toggle '{changed_flag}'")
        print(f"Resulting configuration: {format_config(best_cfg)}")
        print(f"Predicted performance after change: {best_pred:.2f} ms")
    else:
        print("No single‑setting change found that improves the predicted performance.")
        print("Consider exploring changes involving two or more settings.")

    # ------------------------------------------------------------------ #
    # APACHE-16
    # ------------------------------------------------------------------ #
    print("\n== APACHE-16 ==")

    # Define a performance target: the mean of the best 10 % of configurations
    sorted_by_perf = sorted(zip(performances, configs), key=lambda x: x[0])
    best_10_percent = max(1, len(performances) // 10)
    target_perf = statistics.mean([p for p, _ in sorted_by_perf[:best_10_percent]])
    print(f"Target performance (mean of best {best_10_percent} configurations): {target_perf:.2f} ms")

    # Search for the configuration that (a) is present in the dataset,
    # (b) has performance <= target_perf (or if none, closest to target_perf),
    # (c) requires the fewest flag changes from the worst config.
    best_candidate = None
    best_distance = None
    best_candidate_perf = None

    for idx, (perf, cfg) in enumerate(zip(performances, configs)):
        if perf <= target_perf:
            dist = hamming_distance(worst_cfg, cfg, flags)
            if best_distance is None or dist < best_distance:
                best_candidate = cfg
                best_distance = dist
                best_candidate_perf = perf

    # If none meet the target, find the closest in performance
    if best_candidate is None:
        # Compute absolute difference to target
        diffs = [(abs(p - target_perf), hamming_distance(worst_cfg, cfg, flags), idx, p, cfg)
                 for idx, (p, cfg) in enumerate(zip(performances, configs))]
        diffs.sort(key=lambda x: (x[1], x[0]))  # first by distance, then by perf diff
        _, dist, idx, perf, cfg = diffs[0]
        best_candidate = cfg
        best_distance = dist
        best_candidate_perf = perf
        print("No configuration meets the target performance exactly;")
        print("selected the closest in performance and minimal changes.")

    print(f"Recommended configuration to achieve target:")
    print(f"  {format_config(best_candidate)}")
    print(f"  Actual performance: {best_candidate_perf} ms")
    print(f"  Number of setting changes from worst configuration: {best_distance}")

    # End of script


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 09

This script implements two new analysis requests for the client:

  APACHE-17 – Identify configurations whose recorded performance values are
              unusually different from what their settings and comparable
              historical configurations would lead us to expect.

  APACHE-18 – Flag configurations whose combinations of settings differ
              substantially from the most similar configurations in the
              historical benchmark records, even when their measured
              performance is not obviously poor.

The script is self‑contained, uses only the Python standard library,
and can be run as:

    python skill_09.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import statistics
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# 1. CSV parsing
# --------------------------------------------------------------------------- #
def parse_csv(path: str) -> Tuple[List[Dict[str, int]], List[int], List[str]]:
    """Read CSV and return list of configs, performances, and flag names."""
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
# 3. Hamming distance helper
# --------------------------------------------------------------------------- #
def hamming_distance(cfg1: Dict[str, int], cfg2: Dict[str, int], flags: List[str]) -> int:
    """Return number of differing flags between two configurations."""
    return sum(1 for f in flags if cfg1.get(f, 0) != cfg2.get(f, 0))


# --------------------------------------------------------------------------- #
# 4. Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    if not configs:
        print("No configurations found in the CSV.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # APACHE-17: Outlier performance relative to additive model
    # ------------------------------------------------------------------ #
    baseline, effects = build_additive_model(configs, performances, flags)

    predictions = [predict_performance(cfg, baseline, effects) for cfg in configs]
    residuals = [act - pred for act, pred in zip(performances, predictions)]

    res_mean = statistics.mean(residuals)
    res_std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0

    # Flag configurations with |residual| > mean + 2*std
    threshold = res_mean + 2 * res_std
    outliers = [
        (idx, cfg, perf, pred, resid)
        for idx, (cfg, perf, pred, resid) in enumerate(zip(configs, performances, predictions, residuals))
        if abs(resid) > threshold
    ]

    print("== APACHE-17 ==")
    print(f"Total configurations: {len(configs)}")
    print(f"Additive model residual mean: {res_mean:.2f} ms, std dev: {res_std:.2f} ms")
    print(f"Threshold for flagging (mean + 2*std): {threshold:.2f} ms")
    print(f"Number of flagged configurations: {len(outliers)}")
    if outliers:
        print("\nFlagged configurations (index, actual, predicted, residual):")
        for idx, cfg, act, pred, resid in outliers[:min(10, len(outliers))]:
            print(f"  #{idx:>3}: act={act:>4} ms, pred={pred:>6.2f} ms, resid={resid:>6.2f} ms, cfg={cfg}")
    else:
        print("  No configurations exceeded the threshold.")

    # ------------------------------------------------------------------ #
    # APACHE-18: Unusual setting combinations
    # ------------------------------------------------------------------ #
    # Compute nearest‑neighbour hamming distance for each configuration
    nearest_distances: List[int] = []
    nearest_perfs: List[int] = []
    nearest_cfgs: List[Dict[str, int]] = []

    for i, cfg in enumerate(configs):
        best_dist = None
        best_perf = None
        best_cfg = None
        for j, other_cfg in enumerate(configs):
            if i == j:
                continue
            dist = hamming_distance(cfg, other_cfg, flags)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_perf = performances[j]
                best_cfg = other_cfg
        nearest_distances.append(best_dist if best_dist is not None else 0)
        nearest_perfs.append(best_perf if best_perf is not None else 0)
        nearest_cfgs.append(best_cfg if best_cfg is not None else {})

    dist_mean = statistics.mean(nearest_distances)
    dist_std = statistics.stdev(nearest_distances) if len(nearest_distances) > 1 else 0.0
    dist_threshold = dist_mean + 2 * dist_std

    unusual = [
        (idx, cfg, dist, neighbor_perf)
        for idx, (cfg, dist, neighbor_perf) in enumerate(zip(configs, nearest_distances, nearest_perfs))
        if dist > dist_threshold
    ]

    print("\n== APACHE-18 ==")
    print(f"Nearest‑neighbour hamming distance mean: {dist_mean:.2f}, std dev: {dist_std:.2f}")
    print(f"Threshold for flagging (mean + 2*std): {dist_threshold:.2f}")
    print(f"Number of configurations with unusually large distance: {len(unusual)}")
    if unusual:
        print("\nUnusual configurations (index, distance, neighbour perf, cfg):")
        for idx, cfg, dist, n_perf in unusual[:min(10, len(unusual))]:
            print(f"  #{idx:>3}: dist={dist:>2}, neighbour_perf={n_perf:>4} ms, cfg={cfg}")
    else:
        print("  No configurations exceeded the distance threshold.")


if __name__ == "__main__":
    main()

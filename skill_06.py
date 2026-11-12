#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 06

This script implements two new analysis requests for the client:

  APACHE-11 – When the system recommends a configuration, explain its recommendation
              by comparing the configuration's settings and performance with
              relevant previously tested configurations.

  APACHE-12 – When two configurations receive different performance assessments,
              explain which settings differ between them and how those differences
              relate to the predicted or recorded outcomes.

Usage:
    python skill_06.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Iterable


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
# 2. Additive model (same as in skill_02)
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
    effects: Dict[str, float] = {}
    for flag in flags:
        values_1 = [perf for cfg, perf in zip(configs, performances) if cfg[flag] == 1]
        values_0 = [perf for cfg, perf in zip(configs, performances) if cfg[flag] == 0]
        if values_1 and values_0:
            mean1 = statistics.mean(values_1)
            mean0 = statistics.mean(values_0)
            effects[flag] = mean1 - mean0
        else:
            effects[flag] = 0.0
    return baseline_mean, effects


def predict_performance(cfg: Dict[str, int], baseline: float, effects: Dict[str, float]) -> float:
    """Predict performance for a single configuration using the additive model."""
    prediction = baseline
    for flag, effect in effects.items():
        prediction += effect * cfg.get(flag, 0)
    return prediction


# --------------------------------------------------------------------------- #
# 3. Recommended configuration
# --------------------------------------------------------------------------- #
def top_n_configs(
    configs: List[Dict[str, int]],
    performances: List[int],
    n: int,
) -> List[Tuple[Dict[str, int], int]]:
    """Return n best configurations (lower perf is better)."""
    if n <= 0:
        return []
    indices = sorted(range(len(performances)), key=lambda i: performances[i])
    return [(configs[i], performances[i]) for i in indices[:n]]


def most_common_config(configs: List[Dict[str, int]], flags: List[str]) -> Tuple[Dict[str, int], int]:
    """Return the most frequent configuration and its frequency."""
    counter: Counter[Tuple[int, ...]] = Counter()
    for cfg in configs:
        key = tuple(cfg[f] for f in flags)
        counter[key] += 1
    most_common_key, freq = counter.most_common(1)[0]
    common_cfg = {f: val for f, val in zip(flags, most_common_key)}
    return common_cfg, freq


# --------------------------------------------------------------------------- #
# 4. Explain differences between two configs
# --------------------------------------------------------------------------- #
def flag_difference(cfg1: Dict[str, int], cfg2: Dict[str, int], flags: List[str]) -> List[Tuple[str, int, int, float]]:
    """
    Return list of (flag, val1, val2, effect_diff)
    where effect_diff is the effect difference contributed by this flag
    to the overall predicted difference (using model effects).
    """
    diffs = []
    for flag in flags:
        v1 = cfg1.get(flag, 0)
        v2 = cfg2.get(flag, 0)
        if v1 != v2:
            effect_diff = v1 - v2  # +1 or -1
            diffs.append((flag, v1, v2, effect_diff))
    return diffs


def explain_config_pair(
    cfg1: Dict[str, int],
    cfg2: Dict[str, int],
    perf1: int,
    perf2: int,
    baseline: float,
    effects: Dict[str, float],
    flags: List[str],
) -> Iterable[str]:
    """Yield explanation lines for two configurations."""
    yield f"Configuration 1 (performance: {perf1} ms):"
    yield f"  Predicted: {predict_performance(cfg1, baseline, effects):.1f} ms"
    yield ""
    yield f"Configuration 2 (performance: {perf2} ms):"
    yield f"  Predicted: {predict_performance(cfg2, baseline, effects):.1f} ms"
    yield ""

    diff = perf1 - perf2
    diff_pred = predict_performance(cfg1, baseline, effects) - predict_performance(cfg2, baseline, effects)
    yield f"Observed performance difference: {diff} ms"
    yield f"Predicted performance difference: {diff_pred:.1f} ms"
    yield ""

    # Identify flag differences and their effect contribution
    flag_diffs = flag_difference(cfg1, cfg2, flags)
    if not flag_diffs:
        yield "No flag differences."
        return

    yield "Differences in flag settings:"
    for flag, v1, v2, eff in flag_diffs:
        sign = "+" if eff > 0 else "-"
        # effect contribution to predicted difference
        contrib = effects[flag] * eff
        yield f"  {flag}: cfg1={v1}, cfg2={v2} ({sign}{abs(eff)}) -> effect contribution {contrib:.1f} ms"
    yield ""


# --------------------------------------------------------------------------- #
# 5. Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    baseline, effects = build_additive_model(configs, performances, flags)

    # ------------------------------------------------------------------ #
    # APACHE-11: Recommendation explanation
    # ------------------------------------------------------------------ #
    total_configs = len(configs)
    mean_perf = statistics.mean(performances)
    std_perf = statistics.stdev(performances) if len(performances) > 1 else 0.0

    # top 5% best
    top_n = max(1, total_configs // 20)
    best_configs = top_n_configs(configs, performances, top_n)
    best_cfg, best_perf = best_configs[0]
    best_pred = predict_performance(best_cfg, baseline, effects)

    common_cfg, common_freq = most_common_config(configs, flags)

    print("== APACHE-11 ==")
    print(f"Total configurations: {total_configs}")
    print(f"Overall mean performance: {mean_perf:.1f} ms (std={std_perf:.1f} ms)")
    print(f"Recommended configuration (top {top_n}):")
    print(f"  Performance: {best_perf} ms, Predicted: {best_pred:.1f} ms")
    print(f"  Config: {best_cfg}")
    print(f"  This configuration is {best_perf - mean_perf:.1f} ms better than the average.")
    print(f"  It differs from the most common configuration (frequency {common_freq}) by the following flag(s):")
    diff_flags = [f for f in flags if best_cfg.get(f, 0) != common_cfg.get(f, 0)]
    for f in diff_flags:
        print(f"    - {f}: recommended={best_cfg[f]}, common={common_cfg[f]}")
    print(f"  The additive model attributes a total predicted difference of {best_pred - mean_perf:.1f} ms over the mean.")

    # ------------------------------------------------------------------ #
    # APACHE-12: Explanation of two configurations
    # ------------------------------------------------------------------ #
    # Pick best and worst configurations
    worst_cfg, worst_perf = top_n_configs(configs, performances, len(configs))[-1]
    worst_pred = predict_performance(worst_cfg, baseline, effects)

    print("\n== APACHE-12 ==")
    print(f"Comparing best and worst configurations:")
    for line in explain_config_pair(best_cfg, worst_cfg, best_perf, worst_perf,
                                    baseline, effects, flags):
        print(line)

    # End of script


if __name__ == "__main__":
    main()

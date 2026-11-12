#!/usr/bin/env python3
"""
Apache Server Configuration Analysis – Skill 05

This script implements two new analysis requests:

  APACHE-09 – For a poorly performing configuration, list which enabled or
              disabled settings are most strongly associated with poor
              performance in the historical data.

  APACHE-10 – Identify combinations of server settings that frequently
              appear among poorly performing configurations and compare
              them to the overall distribution.

Usage:
    python skill_05.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Helper functions
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

def flag_deltas(configs: List[Dict[str, int]],
                performances: List[int],
                flags: List[str]) -> Dict[str, float]:
    """Return mean performance difference for flag==1 minus flag==0."""
    deltas: Dict[str, float] = {}
    for flag in flags:
        perf1 = [p for c, p in zip(configs, performances) if c[flag] == 1]
        perf0 = [p for c, p in zip(configs, performances) if c[flag] == 0]
        if perf1 and perf0:
            deltas[flag] = statistics.mean(perf1) - statistics.mean(perf0)
        else:
            deltas[flag] = 0.0
    return deltas

def top_poor_flags(worst_cfg: Dict[str, int],
                   deltas: Dict[str, float],
                   top_n: int = 3) -> List[Tuple[str, int, float]]:
    """
    Return top flags whose setting in the worst configuration
    aligns with a higher mean performance (worse).
    Each entry is (flag, setting, delta magnitude).
    """
    results = []
    for flag, delta in deltas.items():
        setting = worst_cfg[flag]
        # Determine if this setting is the worse side
        if (setting == 1 and delta > 0) or (setting == 0 and delta < 0):
            results.append((flag, setting, abs(delta)))
    # Sort by magnitude descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def compute_pair_stats(configs: List[Dict[str, int]],
                       performances: List[int],
                       flags: List[str],
                       poor_indices: List[int]) -> List[Tuple[str, str, float, float, float]]:
    """
    For every pair of flags, compute the pattern percentages in poor
    configurations versus all configurations.
    Return list of tuples:
        (flag1, flag2, poor_percent, all_percent, diff)
    Sorted by diff descending.
    """
    total = len(configs)
    total_poor = len(poor_indices)
    pair_stats: List[Tuple[str, str, float, float, float]] = []

    # Precompute all patterns counts
    all_pattern_counts: Dict[Tuple[str, str], Counter] = {}
    for flag1, flag2 in combinations(flags, 2):
        all_pattern_counts[(flag1, flag2)] = Counter()

    for cfg in configs:
        for (flag1, flag2) in all_pattern_counts:
            pat = f"{cfg[flag1]}{cfg[flag2]}"
            all_pattern_counts[(flag1, flag2)][pat] += 1

    # Now compute for poor configs
    for flag1, flag2 in combinations(flags, 2):
        poor_counts = Counter()
        for idx in poor_indices:
            cfg = configs[idx]
            pat = f"{cfg[flag1]}{cfg[flag2]}"
            poor_counts[pat] += 1

        # Compare percentages
        for pat in ['00', '01', '10', '11']:
            all_c = all_pattern_counts[(flag1, flag2)].get(pat, 0)
            poor_c = poor_counts.get(pat, 0)
            if total == 0 or total_poor == 0:
                continue
            all_pct = (all_c / total) * 100
            poor_pct = (poor_c / total_poor) * 100
            diff = poor_pct - all_pct
            pair_stats.append((flag1, flag2, poor_pct, all_pct, diff))

    # Sort by diff descending
    pair_stats.sort(key=lambda x: x[4], reverse=True)
    return pair_stats

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    configs, performances, flags = parse_csv(csv_path)

    # Overall statistics
    mean_perf = statistics.mean(performances)
    std_perf = statistics.stdev(performances) if len(performances) > 1 else 0.0

    # Identify poorly performing configurations (top 10% worst)
    sorted_indices = sorted(range(len(performances)), key=lambda i: performances[i], reverse=True)
    threshold_index = max(1, len(performances) // 10)  # at least 1
    poor_indices = sorted_indices[:threshold_index]
    # Worst configuration
    worst_idx = sorted_indices[0]
    worst_cfg = configs[worst_idx]
    worst_perf = performances[worst_idx]

    # Compute flag deltas
    deltas = flag_deltas(configs, performances, flags)

    # --------------------
    # APACHE-09
    # --------------------
    print("== APACHE-09 ==")
    top_flags = top_poor_flags(worst_cfg, deltas, top_n=3)
    if not top_flags:
        print("  No clear flag associations found.")
    else:
        print(f"  Worst configuration performance: {worst_perf} ms")
        print("  Top settings most associated with poor performance:")
        for flag, setting, mag in top_flags:
            setting_str = "enabled" if setting == 1 else "disabled"
            sign = "+" if deltas[flag] > 0 else "-"
            print(f"    - {flag} ({setting_str}): delta {sign}{abs(deltas[flag]):.2f} ms")

    # --------------------
    # APACHE-10
    # --------------------
    print("\n== APACHE-10 ==")
    pair_stats = compute_pair_stats(configs, performances, flags, poor_indices)
    if not pair_stats:
        print("  No pairwise patterns identified.")
    else:
        # Show top 5 overrepresented patterns
        print("  Top patterns overrepresented in poor configurations:")
        displayed = 0
        for flag1, flag2, poor_pct, all_pct, diff in pair_stats:
            if diff <= 0:
                continue
            if displayed >= 5:
                break
            print(f"    - Flags {flag1} & {flag2} pattern {flag1}{flag2}:")
            print(f"        Poor configs: {poor_pct:.1f}%")
            print(f"        All configs: {all_pct:.1f}%")
            print(f"        Difference: +{diff:.1f}%")
            displayed += 1
        if displayed == 0:
            print("  No patterns are overrepresented in poor configurations.")

if __name__ == "__main__":
    main()

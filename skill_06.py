#!/usr/bin/env python3
"""
Apache configuration explanation and comparison.

This script implements:
  * APACHE-11: When the system recommends a configuration, explain its recommendation
                by comparing the configuration's settings and performance with relevant
                previously tested configurations.
  * APACHE-12: When two configurations receive different performance assessments,
                explain which settings differ between them and how those differences
                relate to the predicted or recorded outcomes.

Usage:
    python skill_06.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict
from statistics import mean

import numpy as np


# --------------------------------------------------------------------------- #
# 1. CSV reading and numeric conversion
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """Read CSV file and return list of rows as dicts with numeric values."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if not rows:
        print("No data found in CSV.")
        sys.exit(0)

    # Find performance column
    perf_col = None
    for col in rows[0].keys():
        if col.lower().startswith("performance"):
            perf_col = col
            break
    if perf_col is None:
        print("Performance column not found.")
        sys.exit(1)

    # Convert numeric values
    for r in rows:
        for k, v in r.items():
            if k == perf_col:
                r[k] = float(v)
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    r[k] = 0

    return rows, perf_col


# --------------------------------------------------------------------------- #
# 2. Helper functions
# --------------------------------------------------------------------------- #
def setting_cols(rows, perf_col):
    """Return list of setting column names (excluding performance)."""
    return [c for c in rows[0].keys() if c != perf_col]


def performance_summary(rows, perf_col):
    """Return mean, median, min, max of performance values."""
    perf = np.array([r[perf_col] for r in rows])
    return {
        "mean": float(np.mean(perf)),
        "median": float(np.median(perf)),
        "min": float(np.min(perf)),
        "max": float(np.max(perf)),
    }


def recommended_shortlist(rows, perf_col, top_n=5):
    """Return top_n configurations with lowest performance."""
    sorted_rows = sorted(rows, key=lambda r: r[perf_col])
    return sorted_rows[:top_n]


def config_key(r, setting_names):
    """Deterministic key for a configuration."""
    return "|".join(f"{n}={r[n]}" for n in sorted(setting_names))


def compare_configs(a, b, setting_names):
    """Return list of (setting, a_value, b_value) where they differ."""
    diffs = []
    for s in setting_names:
        if a[s] != b[s]:
            diffs.append((s, a[s], b[s]))
    return diffs


# --------------------------------------------------------------------------- #
# 3. APACHE-11: Recommendation explanation
# --------------------------------------------------------------------------- #
def apache_11(rows, perf_col, setting_names):
    """Explain recommendation for top configurations."""
    summary = performance_summary(rows, perf_col)
    best_perf = summary["min"]
    worst_perf = summary["max"]

    # Get recommended shortlist
    shortlist = recommended_shortlist(rows, perf_col, top_n=5)

    print("\n== APACHE-11 ==")
    print(f"Recommended top {len(shortlist)} configurations (lowest performance).")
    print(f"Overall average performance: {summary['mean']:.2f}")
    print(f"Overall best performance: {best_perf:.2f}")
    print(f"Overall worst performance: {worst_perf:.2f}\n")

    for idx, cfg in enumerate(shortlist, 1):
        perf = cfg[perf_col]
        diff_from_mean = perf - summary["mean"]
        print(f"{idx}. Performance={perf:.2f} (Δ={diff_from_mean:+.2f} from mean)")
        # Show a few key settings that differ from the average (use mean of each setting)
        # For simplicity, just list all settings
        settings_desc = ", ".join(f"{s}={cfg[s]}" for s in setting_names)
        print(f"   Settings: {settings_desc}")
        # Explain relative to best configuration
        print(f"   Compared to best config (perf={best_perf:.2f}), "
              f"this config has a performance difference of {perf - best_perf:+.2f}.\n")


# --------------------------------------------------------------------------- #
# 4. APACHE-12: Comparison of two configurations
# --------------------------------------------------------------------------- #
def apache_12(rows, perf_col, setting_names):
    """Explain differences between best and worst configurations."""
    sorted_rows = sorted(rows, key=lambda r: r[perf_col])
    best_cfg = sorted_rows[0]
    worst_cfg = sorted_rows[-1]

    best_perf = best_cfg[perf_col]
    worst_perf = worst_cfg[perf_col]
    perf_diff = worst_perf - best_perf

    print("\n== APACHE-12 ==")
    print(f"Best configuration (performance={best_perf:.2f}) vs "
          f"Worst configuration (performance={worst_perf:.2f})")
    print(f"Performance difference: {perf_diff:.2f} (worst - best)\n")

    diffs = compare_configs(best_cfg, worst_cfg, setting_names)
    if not diffs:
        print("No settings differ between the best and worst configurations.")
        return

    print("Settings differing between the two configurations:")
    for s, val_best, val_worst in diffs:
        print(f"  {s}: best={val_best} | worst={val_worst}")

    # Provide a quick interpretation: if a setting is 1 in best but 0 in worst,
    # it tends to improve performance, etc.
    print("\nInterpretation:")
    for s, val_best, val_worst in diffs:
        if val_best == 1 and val_worst == 0:
            print(f"  Enabling {s} (best config) appears to improve performance.")
        elif val_best == 0 and val_worst == 1:
            print(f"  Disabling {s} (best config) appears to improve performance.")
    print()


# --------------------------------------------------------------------------- #
# 5. Main driver
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)
    setting_names = setting_cols(rows, perf_col)

    apache_11(rows, perf_col, setting_names)
    apache_12(rows, perf_col, setting_names)


if __name__ == "__main__":
    main()

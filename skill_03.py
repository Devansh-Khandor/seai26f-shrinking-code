#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – extended for NASA‑05 and NASA‑06.

Usage:
    python skill_03.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Re‑use helpers from skill_01.py
# --------------------------------------------------------------------------- #

def load_data(csv_path: str) -> List[Dict[str, str]]:
    """Read CSV and return list of records (dicts)."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_float(value: str) -> float:
    """Convert a string to float, stripping spaces. Return NaN for empty strings."""
    try:
        return float(value.strip())
    except ValueError:
        return float('nan')


# --------------------------------------------------------------------------- #
# New helpers for group statistics and CI computation
# --------------------------------------------------------------------------- #

def group_numeric(records: List[Dict[str, str]],
                  group_field: str,
                  numeric_field: str) -> Dict[str, Dict[str, float]]:
    """
    Group numeric values by a categorical field.
    Returns mapping from group value to stats: mean, std, n.
    """
    groups: Dict[str, List[float]] = {}
    for r in records:
        key = r[group_field].strip()
        val = parse_float(r[numeric_field])
        if val != val:   # skip NaN
            continue
        groups.setdefault(key, []).append(val)

    stats: Dict[str, Dict[str, float]] = {}
    for key, values in groups.items():
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        stats[key] = {'mean': mean, 'std': std, 'n': n}
    return stats


def ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    """Return the 95 % confidence interval for a mean."""
    if n <= 1:
        return (mean, mean)
    margin = 1.96 * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def pairwise_no_overlap(ci1: Tuple[float, float], ci2: Tuple[float, float]) -> bool:
    """Return True if two CIs do not overlap."""
    return ci1[1] < ci2[0] or ci2[1] < ci1[0]


# --------------------------------------------------------------------------- #
# Main routine: NASA‑05 & NASA‑06
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)

    # -----------------------------------------------------------------------
    # NASA‑05 – Programmer capability vs effort
    # -----------------------------------------------------------------------
    print("== NASA-05 ==")
    eff_stats = group_numeric(records, 'acap', 'Effort-')
    overall_eff_vals = [parse_float(r['Effort-']) for r in records if parse_float(r['Effort-']) == parse_float(r['Effort-'])]
    overall_eff_mean = statistics.mean(overall_eff_vals)
    overall_eff_std = statistics.stdev(overall_eff_vals) if len(overall_eff_vals) > 1 else 0.0

    # Report group means
    print("Programmer capability (acap) group-wise mean effort:")
    for key in sorted(eff_stats.keys()):
        m = eff_stats[key]['mean']
        n = eff_stats[key]['n']
        print(f"  acap {key}: mean effort {m:.1f} (n={n})")

    # Determine if any pair of groups has non‑overlapping CIs
    groups = sorted(eff_stats.keys())
    ci_map = {k: ci95(eff_stats[k]['mean'], eff_stats[k]['std'], eff_stats[k]['n']) for k in groups}
    meaningful = False
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if pairwise_no_overlap(ci_map[groups[i]], ci_map[groups[j]]):
                meaningful = True
                break
        if meaningful:
            break

    if meaningful:
        print("Development‑effort differences across acap levels are statistically meaningful and "
              "can inform future planning.")
    else:
        print("Development‑effort differences across acap levels are not statistically significant; "
              "available evidence is insufficient for confident planning.")

    # -----------------------------------------------------------------------
    # NASA‑06 – Complexity vs defects
    # -----------------------------------------------------------------------
    print("\n== NASA-06 ==")
    defect_stats = group_numeric(records, 'plex', 'Defects-')
    overall_def_vals = [parse_float(r['Defects-']) for r in records if parse_float(r['Defects-']) == parse_float(r['Defects-'])]
    overall_def_mean = statistics.mean(overall_def_vals)
    overall_def_std = statistics.stdev(overall_def_vals) if len(overall_def_vals) > 1 else 0.0

    # Report group means
    print("Project complexity (plex) group-wise mean defect count:")
    for key in sorted(defect_stats.keys()):
        m = defect_stats[key]['mean']
        n = defect_stats[key]['n']
        print(f"  plex {key}: mean defects {m:.1f} (n={n})")

    # Determine if any pair of groups has non‑overlapping CIs
    groups = sorted(defect_stats.keys())
    ci_map = {k: ci95(defect_stats[k]['mean'], defect_stats[k]['std'], defect_stats[k]['n']) for k in groups}
    reliable = False
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if pairwise_no_overlap(ci_map[groups[i]], ci_map[groups[j]]):
                reliable = True
                break
        if reliable:
            break

    if reliable:
        print("Defect outcomes vary reliably across complexity levels; planning can safely rely on "
              "complexity as a predictor.")
    else:
        print("Defect outcomes do not differ reliably across complexity levels; available evidence "
              "is insufficient to make confident predictions.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>

This script reads the NASA data CSV and answers two client requests:
    NASA-05 – Does programmer capability (acap) influence development effort?
    NASA-06 – Does project complexity (cplx) influence defect outcomes?

Only the Python standard library is used.  The script prints the results
with the required headers and at least three lines of output per request.
"""

import csv
import sys
from statistics import mean


def _parse_float(value):
    """Convert a string to float; return None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _group_stats(rows, group_col, value_col):
    """
    Group rows by the categorical column `group_col` and compute the mean
    of the numeric column `value_col`.  Returns a dict:
        group -> (mean_value, count)
    """
    groups = {}
    for r in rows:
        key = r.get(group_col)
        val = r.get(value_col)
        if key is None or val is None:
            continue
        groups.setdefault(key, []).append(val)
    stats = {}
    for key, vals in groups.items():
        stats[key] = (mean(vals), len(vals))
    return stats


def _overall_mean(rows, value_col):
    """Return the overall mean of a numeric column, ignoring None."""
    vals = [r[value_col] for r in rows if r[value_col] is not None]
    return mean(vals) if vals else None


def _max_min_diff(stats):
    """Return max and min mean values from the stats dict."""
    if not stats:
        return None, None
    means = [m for m, _ in stats.values()]
    return max(means), min(means)


def _percentage_diff(max_val, min_val):
    """Return percentage difference between max and min relative to their average."""
    if max_val is None or min_val is None:
        return None
    avg = (max_val + min_val) / 2
    if avg == 0:
        return 0.0
    return abs(max_val - min_val) / avg * 100


def _interpretation(perc_diff, threshold=20.0):
    """Return a qualitative assessment based on a percentage difference."""
    if perc_diff is None:
        return "Insufficient evidence to assess impact."
    if perc_diff >= threshold:
        return "Significant impact; consider in future planning."
    else:
        return "Impact appears modest; evidence is inconclusive."


def _print_group_summary(title, stats, overall_mean, percent_diff, interpretation):
    """Print a formatted summary for a request."""
    print(f"== {title} ==")
    # Header
    print(f"{'Group':>10} | {'Mean':>10} | {'N':>3}")
    print("-" * 30)
    for group, (m, n) in sorted(stats.items()):
        print(f"{group:>10} | {m:10.2f} | {n:>3}")
    print()
    if overall_mean is not None:
        print(f"Overall mean {overall_mean:.2f}")
    else:
        print("Overall mean: N/A")
    if percent_diff is not None:
        print(f"Max-min difference is {percent_diff:.1f}% of the average of the extremes.")
    print(interpretation)
    print("\n" + "=" * 50 + "\n")


def main(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = {
                "idX": r.get("idX"),
                "acap": r.get("acap"),
                "cplx": r.get("cplx"),
                "Effort-": _parse_float(r.get("Effort-")),
                "Defects-": _parse_float(r.get("Defects-")),
            }
            rows.append(parsed)

    # NASA-05: Programmer capability vs effort
    stats_eff = _group_stats(rows, "acap", "Effort-")
    overall_eff = _overall_mean(rows, "Effort-")
    max_eff, min_eff = _max_min_diff(stats_eff)
    perc_eff = _percentage_diff(max_eff, min_eff)
    interp_eff = _interpretation(perc_eff, threshold=20.0)
    _print_group_summary("NASA-05", stats_eff, overall_eff, perc_eff, interp_eff)

    # NASA-06: Complexity vs defects
    stats_def = _group_stats(rows, "cplx", "Defects-")
    overall_def = _overall_mean(rows, "Defects-")
    max_def, min_def = _max_min_diff(stats_def)
    perc_def = _percentage_diff(max_def, min_def)
    interp_def = _interpretation(perc_def, threshold=20.0)
    _print_group_summary("NASA-06", stats_def, overall_def, perc_def, interp_def)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

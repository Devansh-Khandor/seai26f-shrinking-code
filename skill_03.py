#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <path to Apache_AllMeasurements.csv>
"""

import json
import sys
import math

# Reuse helpers from skill_01
import skill_01 as s01


def compute_group_stats(rows, goal_col, setting):
    """Return dict mapping setting value to stats {n, median, mean, std}."""
    groups = {}
    # Identify distinct values (skip None)
    values = set(row[setting] for row in rows if row[setting] is not None)
    for val in sorted(values):
        subset = [r for r in rows if r[setting] == val]
        # Compute stats for the goal column in this subset
        try:
            stats = s01.compute_stats(subset, goal_col)
        except ValueError:
            # No valid goal values in this group
            continue
        n = len([r for r in subset if r[goal_col] is not None])
        mean = stats['mean']
        median = stats['median']
        # Compute std dev (population)
        if n > 1:
            vals = [r[goal_col] for r in subset if r[goal_col] is not None]
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        else:
            std = 0.0
        groups[val] = {
            'n': n,
            'median': median,
            'mean': mean,
            'std': std,
        }
    return groups


def compute_comparisons(groups):
    """Return list of comparisons with verdict based on non‑overlapping 95% CIs."""
    vals = list(groups.keys())
    comps = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a, b = vals[i], vals[j]
            ga = groups[a]
            gb = groups[b]
            # 95% CI: mean ± 1.96*std/sqrt(n)
            ca_low = ga['mean'] - 1.96 * ga['std'] / math.sqrt(ga['n']) if ga['n'] > 0 else ga['mean']
            ca_high = ga['mean'] + 1.96 * ga['std'] / math.sqrt(ga['n']) if ga['n'] > 0 else ga['mean']
            cb_low = gb['mean'] - 1.96 * gb['std'] / math.sqrt(gb['n']) if gb['n'] > 0 else gb['mean']
            cb_high = gb['mean'] + 1.96 * gb['std'] / math.sqrt(gb['n']) if gb['n'] > 0 else gb['mean']
            # If intervals do not overlap → "different"
            if ca_low > cb_high or cb_low > ca_high:
                verdict = "different"
            else:
                verdict = "same"
            comps.append({"a": a, "b": b, "verdict": verdict})
    return comps


def print_group_stats(section, groups):
    print(f"== {section} ==")
    for val in sorted(groups.keys()):
        g = groups[val]
        print(f"{section.split('-')[1]} {val}: n={g['n']}, median={s01.format_number(g['median'])}, mean={s01.format_number(g['mean'])}")


def print_comparisons(comps, label):
    print()
    for c in comps:
        print(f"Comparison {c['a']} vs {c['b']}: verdict={c['verdict']}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    # Goal column: there should be only one
    goal_cols = [c for c in header if s01.is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    # ----- APACHE-05 -----
    keep_alive_groups = compute_group_stats(rows, goal_col, "keepAlive")
    keep_alive_comps = compute_comparisons(keep_alive_groups)

    print_group_stats("APACHE-05", keep_alive_groups)
    print_comparisons(keep_alive_comps, "APACHE-05")

    # Prepare JSON answer for APACHE-05
    ans05_groups = {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in keep_alive_groups.items()}
    ans05_comps = [{"a": c["a"], "b": c["b"], "verdict": c["verdict"]} for c in keep_alive_comps]
    apache05_answer = {"groups": ans05_groups, "comparisons": ans05_comps}
    print("\nANSWER APACHE-05", json.dumps(apache05_answer, separators=(",", ":")))

    # ----- APACHE-06 -----
    access_log_groups = compute_group_stats(rows, goal_col, "accessLog")
    access_log_comps = compute_comparisons(access_log_groups)

    print_group_stats("APACHE-06", access_log_groups)
    print_comparisons(access_log_comps, "APACHE-06")

    # Prepare JSON answer for APACHE-06
    ans06_groups = {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in access_log_groups.items()}
    ans06_comps = [{"a": c["a"], "b": c["b"], "verdict": c["verdict"]} for c in access_log_comps]
    apache06_answer = {"groups": ans06_groups, "comparisons": ans06_comps}
    print("\nANSWER APACHE-06", json.dumps(apache06_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()

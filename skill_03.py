#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv path>

Adds two new analysis sections:
  * CAR-05: Compare fuel efficiency across cylinder counts.
  * CAR-06: Compare overall performance between two model-year ranges.

The script prints each section header, a few descriptive lines, and a machine‑checked
ANSWER line in JSON format.
"""

import sys
import json
import itertools
import math
import numpy as np

# Reuse helpers from the earlier skill
from skill_01 import read_csv, parse_float, normalize

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def collect_group_values(rows, group_idx, value_idx):
    """
    Return a list of (group_value, value) tuples where the value exists.
    """
    result = []
    for row in rows:
        grp = parse_float(row[group_idx])
        val = parse_float(row[value_idx])
        if grp is not None and val is not None:
            result.append((grp, val))
    return result


def compute_stats_from_values(vals):
    """
    Compute mean, median, std (population) for a list of floats.
    Returns a dict with mean, median, std, n.
    """
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None}
    arr = np.array(vals)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0))
    }


def ci_bounds(mean, std, n, level=0.95):
    """
    Return lower and upper bounds for a normal confidence interval.
    If n <= 1, return mean for both bounds.
    """
    if n <= 1:
        return mean, mean
    z = 1.96  # 95% approx
    se = std / math.sqrt(n)
    return mean - z * se, mean + z * se


def intervals_overlap(a_low, a_high, b_low, b_high):
    """Check if two closed intervals overlap."""
    return not (a_high < b_low or b_high < a_low)


def compute_row_score(row, mins, maxs):
    """
    Compute a combined performance score for a single row.
    Uses the same formula as in skill_01.compute_top_rows.
    """
    lbs = parse_float(row[mins['lbs_idx']])
    acc = parse_float(row[mins['acc_idx']])
    mpg = parse_float(row[mins['mpg_idx']])
    if lbs is None or acc is None or mpg is None:
        return None

    norm_lbs = normalize(lbs, mins['lbs_min'], maxs['lbs_max'])
    norm_acc = normalize(acc, mins['acc_min'], maxs['acc_max'])
    norm_mpg = normalize(mpg, mins['mpg_min'], maxs['mpg_max'])

    return (1 - norm_lbs) + (1 - norm_acc) + norm_mpg


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)

    # Indices for columns of interest
    idx_clndrs = header.index('Clndrs')
    idx_mpg = header.index('Mpg+')
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_model = header.index('Model')

    # ------------------------------------------------------------------ #
    # CAR-05: Cylinder count comparison
    # ------------------------------------------------------------------ #
    print("== CAR-05 ==")

    # Group rows by Clndrs and collect Mpg+ values
    group_pairs = collect_group_values(data, idx_clndrs, idx_mpg)

    # Organise by cylinder count
    groups = {}
    for grp_val, mpg_val in group_pairs:
        groups.setdefault(grp_val, []).append(mpg_val)

    # Compute stats per group
    group_stats = {}
    for grp_val, mpg_vals in groups.items():
        stats = compute_stats_from_values(mpg_vals)
        group_stats[str(grp_val)] = {
            "n": stats["n"],
            "median": round(stats["median"], 3) if stats["median"] is not None else None,
            "mean": round(stats["mean"], 3) if stats["mean"] is not None else None
        }

    # Print human‑readable summary
    for grp_val, stats in sorted(group_stats.items(), key=lambda x: int(x[0])):
        print(f"Cylinder {grp_val}: n={stats['n']}, median Mpg+={stats['median']}, mean Mpg+={stats['mean']}")

    # Compute pairwise comparisons
    comparisons = []
    # Pre‑compute confidence intervals for each group
    group_cis = {}
    for grp_val, mpg_vals in groups.items():
        n = len(mpg_vals)
        if n == 0:
            continue
        mean = np.mean(mpg_vals)
        std = np.std(mpg_vals, ddof=0)
        low, high = ci_bounds(mean, std, n)
        group_cis[grp_val] = (low, high)

    grp_keys = list(groups.keys())
    for a, b in itertools.combinations(grp_keys, 2):
        low_a, high_a = group_cis[a]
        low_b, high_b = group_cis[b]
        verdict = "different" if not intervals_overlap(low_a, high_a, low_b, high_b) else "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Machine‑checked answer for CAR‑05
    answer_car05 = {"groups": group_stats, "comparisons": comparisons}
    print(f"ANSWER CAR-05 {json.dumps(answer_car05, separators=(',', ':'), sort_keys=True)}")

    # ------------------------------------------------------------------ #
    # CAR-06: Performance comparison between two model ranges
    # ------------------------------------------------------------------ #
    print("\n== CAR-06 ==")

    # Determine two non‑overlapping inclusive Model ranges (split at median unique value)
    unique_models = sorted(set(parse_float(row[idx_model]) for row in data if parse_float(row[idx_model]) is not None))
    mid = len(unique_models) // 2
    first_range = (unique_models[0], unique_models[mid - 1] if mid > 0 else unique_models[0])
    second_range = (unique_models[mid], unique_models[-1])

    # Function to get rows in a range
    def rows_in_range(r_from, r_to):
        return [row for row in data if parse_float(row[idx_model]) is not None and r_from <= parse_float(row[idx_model]) <= r_to]

    rows_first = rows_in_range(*first_range)
    rows_second = rows_in_range(*second_range)

    # Compute global mins and maxs for scoring
    # Ensure we only consider rows where all goal columns are present
    def scoreable_rows(rows):
        return [row for row in rows if parse_float(row[idx_lbs]) is not None and parse_float(row[idx_acc]) is not None and parse_float(row[idx_mpg]) is not None]

    scoreable_all = scoreable_rows(data)
    mins = {
        'lbs_idx': idx_lbs,
        'acc_idx': idx_acc,
        'mpg_idx': idx_mpg,
        'lbs_min': min(parse_float(r[idx_lbs]) for r in scoreable_all),
        'acc_min': min(parse_float(r[idx_acc]) for r in scoreable_all),
        'mpg_min': min(parse_float(r[idx_mpg]) for r in scoreable_all)
    }
    maxs = {
        'lbs_idx': idx_lbs,
        'acc_idx': idx_acc,
        'mpg_idx': idx_mpg,
        'lbs_max': max(parse_float(r[idx_lbs]) for r in scoreable_all),
        'acc_max': max(parse_float(r[idx_acc]) for r in scoreable_all),
        'mpg_max': max(parse_float(r[idx_mpg]) for r in scoreable_all)
    }

    # Compute scores for each group
    def compute_scores(rows):
        scores = []
        for row in rows:
            sc = compute_row_score(row, mins, maxs)
            if sc is not None:
                scores.append(sc)
        return scores

    scores_first = compute_scores(rows_first)
    scores_second = compute_scores(rows_second)

    # Compute stats per group
    def stats_from_scores(scores):
        if not scores:
            return {"n": 0, "mean": None, "std": None}
        arr = np.array(scores)
        return {"n": int(len(arr)), "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}

    stats_first = stats_from_scores(scores_first)
    stats_second = stats_from_scores(scores_second)

    # Print human‑readable summary
    print(f"Group 1: Model {first_range[0]}–{first_range[1]}, n={stats_first['n']}")
    print(f"Group 2: Model {second_range[0]}–{second_range[1]}, n={stats_second['n']}")
    print(f"Group 1 mean score={round(stats_first['mean'],3) if stats_first['mean'] is not None else None}")
    print(f"Group 2 mean score={round(stats_second['mean'],3) if stats_second['mean'] is not None else None}")

    # Determine verdict using confidence intervals
    if stats_first['n'] == 0 or stats_second['n'] == 0:
        verdict = "same"
    else:
        low1, high1 = ci_bounds(stats_first['mean'], stats_first['std'], stats_first['n'])
        low2, high2 = ci_bounds(stats_second['mean'], stats_second['std'], stats_second['n'])
        verdict = "different" if not intervals_overlap(low1, high1, low2, high2) else "same"

    # Machine‑checked answer for CAR‑06
    answer_car06 = {
        "groups": [
            {"from": first_range[0], "to": first_range[1], "n": stats_first["n"]},
            {"from": second_range[0], "to": second_range[1], "n": stats_second["n"]},
        ],
        "verdict": verdict
    }
    print(f"ANSWER CAR-06 {json.dumps(answer_car06, separators=(',', ':'), sort_keys=True)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

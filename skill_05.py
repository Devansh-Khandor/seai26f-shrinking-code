#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv path>

Adds two new analysis sections:
  * CAR-09: Identify vehicles with unusually poor fuel efficiency and rank characteristics
            most strongly associated with that outcome.
  * CAR-10: Find the common characteristics of vehicles with unfavorable overall performance
            and compare them to better performing vehicles.

The script prints each section header, a few descriptive lines, and a machine‑checked ANSWER line
in JSON format.
"""

import sys
import json
import numpy as np
from collections import Counter

# Reuse helpers from previous skills
from skill_01 import read_csv, parse_float, normalize

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def compute_row_score(row, mins, maxs, idx_lbs, idx_acc, idx_mpg):
    """
    Compute the combined performance score for a single row.
    Returns None if any goal is missing.
    """
    lbs = parse_float(row[idx_lbs])
    acc = parse_float(row[idx_acc])
    mpg = parse_float(row[idx_mpg])
    if lbs is None or acc is None or mpg is None:
        return None

    norm_lbs = normalize(lbs, mins['lbs_min'], maxs['lbs_max'])
    norm_acc = normalize(acc, mins['acc_min'], maxs['acc_max'])
    norm_mpg = normalize(mpg, mins['mpg_min'], maxs['mpg_max'])

    return (1 - norm_lbs) + (1 - norm_acc) + norm_mpg


def collect_scores(rows, idx_lbs, idx_acc, idx_mpg):
    """
    Return a list of (row_num, score) tuples for rows with all goal values present.
    """
    # Determine min and max for normalization
    lbs_vals = [parse_float(r[idx_lbs]) for r in rows if parse_float(r[idx_lbs]) is not None]
    acc_vals = [parse_float(r[idx_acc]) for r in rows if parse_float(r[idx_acc]) is not None]
    mpg_vals = [parse_float(r[idx_mpg]) for r in rows if parse_float(r[idx_mpg]) is not None]

    mins = {
        'lbs_min': float(np.min(lbs_vals)),
        'acc_min': float(np.min(acc_vals)),
        'mpg_min': float(np.min(mpg_vals))
    }
    maxs = {
        'lbs_max': float(np.max(lbs_vals)),
        'acc_max': float(np.max(acc_vals)),
        'mpg_max': float(np.max(mpg_vals))
    }

    scored = []
    for i, row in enumerate(rows, start=1):
        sc = compute_row_score(row, mins, maxs, idx_lbs, idx_acc, idx_mpg)
        if sc is not None:
            scored.append((i, sc))
    return scored


def get_mean(values):
    return float(np.mean(values)) if values else None


def get_mode(values):
    if not values:
        return None
    cnt = Counter(values)
    max_count = max(cnt.values())
    modes = [v for v, c in cnt.items() if c == max_count]
    return min(modes)  # deterministic


def compute_typical(rows, idxs):
    """
    Compute typical (mean for numeric) for each feature index across rows.
    """
    typ = {}
    for idx in idxs:
        vals = [parse_float(r[idx]) for r in rows if parse_float(r[idx]) is not None]
        if vals:
            typ[idx] = get_mean(vals)
    return typ


def compute_feature_correlation(rows, feature_idx, mpg_idx):
    """
    Compute Pearson correlation coefficient between a feature and Mpg+.
    Returns absolute correlation.
    """
    feat_vals = []
    mpg_vals = []
    for r in rows:
        fv = parse_float(r[feature_idx])
        mv = parse_float(r[mpg_idx])
        if fv is not None and mv is not None:
            feat_vals.append(fv)
            mpg_vals.append(mv)
    if len(feat_vals) < 2:
        return 0.0
    return abs(float(np.corrcoef(feat_vals, mpg_vals)[0, 1]))


# --------------------------------------------------------------------------- #
# Main logic for CAR-09 and CAR-10
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)

    # Indices for goal columns
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Indices for non‑goal columns (excluding X columns)
    non_goal_idxs = [i for i, name in enumerate(header)
                     if not name.endswith('+') and not name.endswith('-')
                     and not name.endswith('X')]

    # ------------------------------------------------------------------ #
    # CAR-09: Poor fuel efficiency
    # ------------------------------------------------------------------ #
    print("== CAR-09 ==")

    # Gather mpg values
    mpg_rows = [(i, parse_float(r[idx_mpg]))
                for i, r in enumerate(data, start=1)
                if parse_float(r[idx_mpg]) is not None]

    # Sort by mpg ascending to find poorest
    mpg_rows.sort(key=lambda x: x[1])
    poor_rows = [i for i, _ in mpg_rows[:5]]  # at least 5

    print(f"Poor Mpg+ rows (lowest 5): {poor_rows}")

    # Correlation ranking for each non‑goal feature
    corrs = []
    for idx in non_goal_idxs:
        corr = compute_feature_correlation(data, idx, idx_mpg)
        if corr > 0:
            corrs.append((corr, header[idx]))
    corrs.sort(reverse=True)
    ranking = [name for _, name in corrs[:3]]  # at least 3

    print(f"Top 3 features most correlated with poor Mpg+: {ranking}")

    # Machine‑checked answer
    answer_car09 = {"poor": poor_rows, "ranking": ranking}
    print(f"ANSWER CAR-09 {json.dumps(answer_car09, separators=(',', ':'), sort_keys=True)}")

    # ------------------------------------------------------------------ #
    # CAR-10: Unfavorable vs favorable overall performance
    # ------------------------------------------------------------------ #
    print("\n== CAR-10 ==")

    # Compute scores for all rows
    scored_rows = collect_scores(data, idx_lbs, idx_acc, idx_mpg)

    # Sort by score (lower is worse)
    scored_rows.sort(key=lambda x: x[1])
    bad_rows = [i for i, _ in scored_rows[:5]]   # bottom 5
    good_rows = [i for i, _ in scored_rows[-5:]]  # top 5

    print(f"Bad overall performance rows (worst 5): {bad_rows}")
    print(f"Good overall performance rows (best 5): {good_rows}")

    # Compute typical values for each feature in bad and good groups
    bad_group_rows = [data[i-1] for i in bad_rows]
    good_group_rows = [data[i-1] for i in good_rows]

    bad_typ = compute_typical(bad_group_rows, non_goal_idxs)
    good_typ = compute_typical(good_group_rows, non_goal_idxs)

    # Compute differences
    differences = []
    for idx in non_goal_idxs:
        feature_name = header[idx]
        bad_val = bad_typ.get(idx)
        good_val = good_typ.get(idx)
        if bad_val is None or good_val is None:
            continue
        # numeric difference
        diff = abs(bad_val - good_val)
        differences.append((diff, feature_name, bad_val, good_val))
    differences.sort(reverse=True)
    diff_list = [{
        "feature": f,
        "bad": round(b, 3) if isinstance(b, float) else b,
        "good": round(g, 3) if isinstance(g, float) else g
    } for _, f, b, g in differences[:3]]  # at least 3

    print(f"Top 3 feature differences between bad and good groups: {diff_list}")

    # Machine‑checked answer
    answer_car10 = {
        "bad": bad_rows,
        "good": good_rows,
        "differences": diff_list
    }
    print(f"ANSWER CAR-10 {json.dumps(answer_car10, separators=(',', ':'), sort_keys=True)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

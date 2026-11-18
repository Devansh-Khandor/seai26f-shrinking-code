#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <csv path>

Adds two new analysis sections:
  * CAR-13: Identify a vehicle with a strong balance of fuel efficiency, weight,
            and acceleration while evaluating as few candidate vehicles as
            reasonably possible.
  * CAR-14: With a limited inspection budget, recommend which vehicles should be
            evaluated first so that we can discover promising inventory options early.

The script prints each section header, a few descriptive lines, and a machine‑checked
ANSWER line in JSON format.
"""

import sys
import json
import math
import numpy as np

# Reuse helpers from skill_01
from skill_01 import read_csv, parse_float, normalize

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def compute_score(row, mins, maxs, idx_lbs, idx_acc, idx_mpg):
    """
    Compute the combined performance score for a single row.
    Return None if any goal is missing.
    Score is higher for better performance.
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


def compute_all_goals_stats(rows, idx_lbs, idx_acc, idx_mpg):
    """
    Return mins and maxs for goal columns computed over the provided rows.
    """
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
    return mins, maxs


def get_feature_indices(header):
    """
    Return a list of column indices that are not goal columns (do not end in '+' or '-')
    and are not input columns (do not end in 'X').
    """
    indices = []
    for i, name in enumerate(header):
        if name.endswith('+') or name.endswith('-') or name.endswith('X'):
            continue
        indices.append(i)
    return indices


def compute_feature_min_max(rows, feature_idxs):
    """
    Compute per‑feature min and max across rows, ignoring missing values.
    """
    mins = {}
    maxs = {}
    for idx in feature_idxs:
        vals = [parse_float(r[idx]) for r in rows if parse_float(r[idx]) is not None]
        if vals:
            mins[idx] = float(np.min(vals))
            maxs[idx] = float(np.max(vals))
        else:
            mins[idx] = maxs[idx] = 0.0
    return mins, maxs


def compute_proxy_score(row, mins, maxs, feature_idxs):
    """
    Compute a proxy score for a row based on non‑goal numeric features.
    Lower feature values are considered better (e.g., lower horsepower, lower volume).
    """
    score = 0.0
    count = 0
    for idx in feature_idxs:
        val = parse_float(row[idx])
        if val is None:
            continue
        minv = mins[idx]
        maxv = maxs[idx]
        if maxv == minv:
            norm = 0.0
        else:
            # better when lower -> use (max - val)
            norm = (maxv - val) / (maxv - minv)
        score += norm
        count += 1
    # Normalize by number of features considered to avoid bias when some features missing
    return score / count if count else 0.0


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)
    n_rows = len(data)

    # Goal column indices
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # ------------------------------------------------------------------ #
    # CAR-13: Evaluate few vehicles, find best balanced one
    # ------------------------------------------------------------------ #
    eval_count = min(30, n_rows)
    evaluated_rows = data[:eval_count]
    evaluated_row_nums = list(range(1, eval_count + 1))

    # Compute min/max over evaluated rows for goals
    mins, maxs = compute_all_goals_stats(evaluated_rows, idx_lbs, idx_acc, idx_mpg)

    # Compute score for each evaluated row
    best_row = None
    best_score = -math.inf
    for rn, row in zip(evaluated_row_nums, evaluated_rows):
        sc = compute_score(row, mins, maxs, idx_lbs, idx_acc, idx_mpg)
        if sc is not None and sc > best_score:
            best_score = sc
            best_row = rn

    # Print section
    print("== CAR-13 ==")
    print(f"Evaluated rows: {evaluated_row_nums}")
    if best_row is not None:
        print(f"Best row: {best_row} (Score={best_score:.3f})")
    else:
        print("No complete rows among evaluated set.")
    # Machine‑checked answer
    answer_car13 = {"evaluated": evaluated_row_nums, "best": best_row}
    print(f"ANSWER CAR-13 {json.dumps(answer_car13, separators=(',', ':'), sort_keys=True)}")

    # ------------------------------------------------------------------ #
    # CAR-14: Recommend evaluation order with budget
    # ------------------------------------------------------------------ #
    budget = min(30, n_rows)
    feature_idxs = get_feature_indices(header)
    feature_mins, feature_maxs = compute_feature_min_max(data, feature_idxs)

    # Compute proxy scores for all rows
    all_scores = []
    for rn, row in enumerate(data, start=1):
        sc = compute_proxy_score(row, feature_mins, feature_maxs, feature_idxs)
        all_scores.append((sc, rn))

    # Sort by proxy score descending
    all_scores.sort(reverse=True, key=lambda x: x[0])
    order = [rn for _, rn in all_scores[:budget]]

    # Print section
    print("\n== CAR-14 ==")
    print(f"Evaluation order (top {budget}): {order}")
    # Machine‑checked answer
    answer_car14 = {"order": order}
    print(f"ANSWER CAR-14 {json.dumps(answer_car14, separators=(',', ':'), sort_keys=True)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

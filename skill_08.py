#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <csv path>

Adds two new analysis sections:
  * CAR-15: For a poorly rated vehicle, show alternative vehicles in the historical
            inventory with similar characteristics but better recorded performance,
            and explain which differences could account for the improvement.
  * CAR-16: For a vehicle that does not meet our preferred performance targets,
            identify the closest available alternative that meets more of those
            targets while changing as few vehicle characteristics as possible.

The script prints each section header, a few descriptive lines, and a machine‑checked
ANSWER line in JSON format.
"""

import sys
import json
import math
import numpy as np

# Reuse helpers from earlier skills
from skill_01 import read_csv, parse_float
from skill_07 import get_feature_indices

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def compute_goals_min_max(rows, idx_lbs, idx_acc, idx_mpg):
    """Return dict of mins and maxs for goal columns."""
    lbs_vals = [parse_float(r[idx_lbs]) for r in rows if parse_float(r[idx_lbs]) is not None]
    acc_vals = [parse_float(r[idx_acc]) for r in rows if parse_float(r[idx_acc]) is not None]
    mpg_vals = [parse_float(r[idx_mpg]) for r in rows if parse_float(r[idx_mpg]) is not None]
    return {
        'lbs_min': float(np.min(lbs_vals)),
        'acc_min': float(np.min(acc_vals)),
        'mpg_min': float(np.min(mpg_vals)),
        'lbs_max': float(np.max(lbs_vals)),
        'acc_max': float(np.max(acc_vals)),
        'mpg_max': float(np.max(mpg_vals)),
    }

def compute_score(row, mins, maxs, idx_lbs, idx_acc, idx_mpg):
    """Compute the combined performance score for a single row."""
    lbs = parse_float(row[idx_lbs])
    acc = parse_float(row[idx_acc])
    mpg = parse_float(row[idx_mpg])
    if lbs is None or acc is None or mpg is None:
        return None

    norm_lbs = (lbs - mins['lbs_min']) / (maxs['lbs_max'] - mins['lbs_min']) if maxs['lbs_max'] > mins['lbs_min'] else 0.0
    norm_acc = (acc - mins['acc_min']) / (maxs['acc_max'] - mins['acc_min']) if maxs['acc_max'] > mins['acc_min'] else 0.0
    norm_mpg = (mpg - mins['mpg_min']) / (maxs['mpg_max'] - mins['mpg_min']) if maxs['mpg_max'] > mins['mpg_min'] else 0.0

    return (1 - norm_lbs) + (1 - norm_acc) + norm_mpg

def compute_all_scores(rows, idx_lbs, idx_acc, idx_mpg):
    """Return list of (row_num, score) for rows with complete goal data."""
    minsmax = compute_goals_min_max(rows, idx_lbs, idx_acc, idx_mpg)
    scores = []
    for i, r in enumerate(rows, start=1):
        sc = compute_score(r, minsmax, minsmax, idx_lbs, idx_acc, idx_mpg)
        if sc is not None:
            scores.append((i, sc))
    return scores, minsmax

def compute_feature_min_max(rows, feature_idxs):
    """Compute per‑feature min and max across rows, ignoring missing values."""
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

def normalized_distance(row_a, row_b, feature_idxs, mins, maxs):
    """Compute Euclidean distance in normalized feature space."""
    dist2 = 0.0
    for idx in feature_idxs:
        va = parse_float(row_a[idx])
        vb = parse_float(row_b[idx])
        if va is None or vb is None:
            continue
        denom = maxs[idx] - mins[idx]
        if denom == 0:
            norm_diff = 0.0
        else:
            norm_diff = (va - vb) / denom
        dist2 += norm_diff * norm_diff
    return math.sqrt(dist2)

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)

    # Goal column indices
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Compute all scores
    all_scores, minsmax = compute_all_scores(data, idx_lbs, idx_acc, idx_mpg)
    if not all_scores:
        print("No complete rows for scoring.")
        sys.exit(1)

    # Find best and worst rows
    best_row_num, best_score = max(all_scores, key=lambda x: x[1])
    worst_row_num, worst_score = min(all_scores, key=lambda x: x[1])

    # Retrieve row data
    best_row = data[best_row_num - 1]
    worst_row = data[worst_row_num - 1]

    # ------------------------------------------------------------------ #
    # CAR-15: Similar but better alternatives
    # ------------------------------------------------------------------ #
    print("== CAR-15 ==")
    # Identify input feature indices (not goals, not X)
    feature_idxs = get_feature_indices(header)
    feat_mins, feat_maxs = compute_feature_min_max(data, feature_idxs)

    # Find alternatives: rows with better score and minimal distance
    alternatives = []
    for rn, sc in all_scores:
        if rn == worst_row_num:
            continue
        if sc <= worst_score:
            continue
        dist = normalized_distance(worst_row, data[rn - 1], feature_idxs, feat_mins, feat_maxs)
        alternatives.append((dist, rn))

    # Sort by distance
    alternatives.sort(key=lambda x: x[0])
    # Pick first alternative that exists
    if alternatives:
        best_alt_num = alternatives[0][1]
        alt_row = data[best_alt_num - 1]
        # Compute changes
        changes = {}
        for idx in feature_idxs:
            val_w = parse_float(worst_row[idx])
            val_a = parse_float(alt_row[idx])
            if val_w is None or val_a is None:
                continue
            if val_w != val_a:
                changes[header[idx]] = [val_w, val_a]
        alternative_objs = [{
            "row": best_alt_num,
            "changes": changes
        }]
    else:
        alternative_objs = []

    print(f"Poor vehicle: Row {worst_row_num}")
    print(f"Alternatives found: {len(alternative_objs)}")
    if alternative_objs:
        print(f"Example alternative: Row {alternative_objs[0]['row']}")
    else:
        print("No better alternatives found.")

    answer_car15 = {
        "row": worst_row_num,
        "alternatives": alternative_objs
    }
    print(f"ANSWER CAR-15 {json.dumps(answer_car15, separators=(',', ':'), sort_keys=True)}")

    # ------------------------------------------------------------------ #
    # CAR-16: Closest better alternative meeting more targets
    # ------------------------------------------------------------------ #
    print("\n== CAR-16 ==")

    # Define targets: use best row's goal values
    targets = {
        "Lbs-": parse_float(best_row[idx_lbs]),
        "Acc+": parse_float(best_row[idx_acc]),
        "Mpg+": parse_float(best_row[idx_mpg]),
    }

    # Count targets met for a row
    def count_targets(row):
        cnt = 0
        lbs = parse_float(row[idx_lbs])
        acc = parse_float(row[idx_acc])
        mpg = parse_float(row[idx_mpg])
        if lbs is not None and lbs <= targets["Lbs-"]:
            cnt += 1
        if acc is not None and acc >= targets["Acc+"];
        if acc is not None and acc >= targets["Acc+"]:
            cnt += 1
        if mpg is not None and mpg >= targets["Mpg+"]:
            cnt += 1
        return cnt

    # Use the worst row as the candidate row that lacks targets
    row_num = worst_row_num
    row_data = worst_row
    row_cnt = count_targets(row_data)

    # Find candidates that meet more targets
    candidates = []
    for rn, sc in all_scores:
        if rn == row_num:
            continue
        cand_row = data[rn - 1]
        cand_cnt = count_targets(cand_row)
        if cand_cnt > row_cnt:
            # Count number of changed input columns
            changes = 0
            for idx in feature_idxs:
                v1 = parse_float(row_data[idx])
                v2 = parse_float(cand_row[idx])
                if v1 is None or v2 is None:
                    continue
                if v1 != v2:
                    changes += 1
            candidates.append((cand_cnt, changes, rn, cand_row))

    # Select best candidate: more targets, then fewer changes, then lower row number
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        best_candidate = candidates[0]
        alt_row_num = best_candidate[2]
        alt_row = best_candidate[3]
        # Compute changed columns
        changed_cols = []
        for idx in feature_idxs:
            v1 = parse_float(row_data[idx])
            v2 = parse_float(alt_row[idx])
            if v1 is None or v2 is None:
                continue
            if v1 != v2:
                changed_cols.append(header[idx])
    else:
        alt_row_num = None
        changed_cols = []

    print(f"Row missing some targets: Row {row_num}")
    if alt_row_num is not None:
        print(f"Closest better alternative: Row {alt_row_num}")
    else:
        print("No alternative found meeting more targets.")
    print(f"Targets thresholds: {targets}")

    answer_car16 = {
        "row": row_num,
        "targets": targets,
        "alternative": alt_row_num,
        "changed": changed_cols
    }
    print(f"ANSWER CAR-16 {json.dumps(answer_car16, separators=(',', ':'), sort_keys=True)}")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

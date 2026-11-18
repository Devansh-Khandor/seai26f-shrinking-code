#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <csv path>

Adds two new analysis sections:
  * CAR-17: Flag vehicle records whose input combinations are unusually different
            from the rest of the inventory so staff can review them.
  * CAR-18: Identify vehicles whose reported fuel efficiency or weight differs
            substantially from what similar vehicles would lead us to expect,
            so we can check for possible recording mistakes.

The script prints each section header, a few descriptive lines, and a machine‑checked
ANSWER line in JSON format.
"""

import sys
import json
import math
import numpy as np

# Reuse helpers from previous skills
from skill_01 import read_csv, parse_float
from skill_07 import get_feature_indices

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

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

def normalize_value(val, minimum, maximum):
    """Linearly normalize a value to [0,1]; if min==max return 0.0."""
    if maximum == minimum:
        return 0.0
    return (val - minimum) / (maximum - minimum)

def normalized_vector(row, feature_idxs, mins, maxs):
    """Return a list of normalized values for the given row over feature_idxs."""
    vec = []
    for idx in feature_idxs:
        val = parse_float(row[idx])
        if val is None:
            return None  # incomplete
        vec.append(normalize_value(val, mins[idx], maxs[idx]))
    return np.array(vec)

def euclidean_distance(v1, v2):
    """Euclidean distance between two 1‑D numpy arrays."""
    return np.linalg.norm(v1 - v2)

# --------------------------------------------------------------------------- #
# CAR-17: Unusual input combinations
# --------------------------------------------------------------------------- #

def car_17(header, data, feature_idxs):
    """Return flagged rows and scores for CAR‑17."""
    mins, maxs = compute_feature_min_max(data, feature_idxs)

    # Compute normalized vectors and distances to mean
    vectors = []
    row_numbers = []
    for i, row in enumerate(data, start=1):
        vec = normalized_vector(row, feature_idxs, mins, maxs)
        if vec is None:
            continue  # skip rows with missing input
        vectors.append(vec)
        row_numbers.append(i)

    if not vectors:
        return [], {}

    vecs = np.vstack(vectors)
    mean_vec = np.mean(vecs, axis=0)
    distances = np.linalg.norm(vecs - mean_vec, axis=1)

    # Statistical threshold: mean + 2*std
    d_mean = np.mean(distances)
    d_std = np.std(distances, ddof=0)
    threshold = d_mean + 2 * d_std

    flagged = []
    scores = {}
    for rn, dist in zip(row_numbers, distances):
        if dist > threshold:
            flagged.append(rn)
            scores[str(rn)] = float(dist)

    return sorted(flagged), scores

# --------------------------------------------------------------------------- #
# CAR-18: Unexpected Mpg+ or Lbs- compared to similar vehicles
# --------------------------------------------------------------------------- #

def car_18(header, data, feature_idxs):
    """Return flagged rows with unexpected Mpg+ or Lbs- for CAR‑18."""
    mins, maxs = compute_feature_min_max(data, feature_idxs)

    # Precompute normalized vectors for all rows (skip incomplete)
    vectors = {}
    for i, row in enumerate(data, start=1):
        vec = normalized_vector(row, feature_idxs, mins, maxs)
        if vec is not None:
            vectors[i] = vec

    if not vectors:
        return []

    flagged_entries = []

    idx_lbs = header.index('Lbs-')
    idx_mpg = header.index('Mpg+')

    # For each row, find nearest neighbors among other rows
    all_rows = set(vectors.keys())
    for rn, vec in vectors.items():
        # Build list of other rows and distances
        neighbors = []
        for other_rn in all_rows:
            if other_rn == rn:
                continue
            d = euclidean_distance(vec, vectors[other_rn])
            neighbors.append((d, other_rn))
        neighbors.sort(key=lambda x: x[0])

        # Take the 5 nearest (or fewer if not enough)
        k = min(5, len(neighbors))
        nearest = [other for _, other in neighbors[:k]]

        # Gather neighbor values for the two target columns
        mpg_vals = []
        lbs_vals = []
        for nr in nearest:
            val_mpg = parse_float(data[nr-1][idx_mpg])
            val_lbs = parse_float(data[nr-1][idx_lbs])
            if val_mpg is not None:
                mpg_vals.append(val_mpg)
            if val_lbs is not None:
                lbs_vals.append(val_lbs)

        # For each column, compute mean and std
        for col_name, vals in (('Mpg+', mpg_vals), ('Lbs-', lbs_vals)):
            actual = parse_float(data[rn-1][header.index(col_name)])
            if actual is None or not vals:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=0))
            # If std is zero, flag if actual differs from mean
            if std_val == 0:
                if actual != mean_val:
                    flagged_entries.append({
                        "row": rn,
                        "column": col_name,
                        "actual": float(actual),
                        "expected": round(mean_val, 3)
                    })
                continue
            # Flag if deviation > 2*std
            if abs(actual - mean_val) > 2 * std_val:
                flagged_entries.append({
                    "row": rn,
                    "column": col_name,
                    "actual": float(actual),
                    "expected": round(mean_val, 3)
                })
        # Stop if we have reached 40
        if len(flagged_entries) >= 40:
            break

    # Sort entries by row number
    flagged_entries.sort(key=lambda x: x["row"])
    return flagged_entries[:40]

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)

    feature_idxs = get_feature_indices(header)

    # -------------------------------- CAR‑17 --------------------------------
    print("== CAR-17 ==")
    flagged_17, scores_17 = car_17(header, data, feature_idxs)
    print(f"Flagged rows: {flagged_17}")
    # Provide at least three lines of results
    for i, rn in enumerate(flagged_17[:3], start=1):
        print(f"{i}. Row {rn} unusualness score={scores_17[str(rn)]:.4f}")
    # Machine‑checked answer
    answer_car17 = {"flagged": flagged_17, "scores": scores_17}
    print(f"ANSWER CAR-17 {json.dumps(answer_car17, separators=(',', ':'), sort_keys=True)}")

    # -------------------------------- CAR‑18 --------------------------------
    print("\n== CAR-18 ==")
    flagged_18 = car_18(header, data, feature_idxs)
    print(f"Flagged entries: {len(flagged_18)}")
    for i, entry in enumerate(flagged_18[:3], start=1):
        print(f"{i}. Row {entry['row']} {entry['column']} actual={entry['actual']} expected={entry['expected']}")
    # Machine‑checked answer
    answer_car18 = {"flagged": flagged_18}
    print(f"ANSWER CAR-18 {json.dumps(answer_car18, separators=(',', ':'), sort_keys=True)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_09.py - CAR-17 and CAR-18: outlier detection and anomaly flagging

Usage:
    python skill_09.py <csv_path>
"""

import sys
import csv
import json
import math
from collections import defaultdict

GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']      # Lbs- minimize, others maximize
NON_GOAL_COLUMNS = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']
ALL_COLUMNS = NON_GOAL_COLUMNS + GOAL_COLUMNS

# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts with numeric conversion.
    Missing values are represented as None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_row = {}
            for col in ALL_COLUMNS:
                val = row[col]
                if val == '?':
                    new_row[col] = None
                else:
                    try:
                        new_row[col] = float(val)
                    except ValueError:
                        new_row[col] = None
            rows.append(new_row)
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def mean(lst):
    return sum(lst) / len(lst) if lst else None

def stddev(lst):
    n = len(lst)
    if n < 2:
        return None
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (n - 1))

# ----------------------------------------------------------------------
# CAR-17: Unusual combination flagging
# ----------------------------------------------------------------------
def compute_column_stats(rows):
    """Return mean and stddev per column across non-None values."""
    stats = {}
    for col in ALL_COLUMNS:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            stats[col] = {
                'mean': mean(vals),
                'std': stddev(vals) or 0.0  # avoid zero std
            }
    return stats

def compute_unusual_score(row, stats):
    """Compute a simple standardized Euclidean distance to the mean."""
    z2 = 0.0
    count = 0
    for col in ALL_COLUMNS:
        val = row[col]
        if val is None or col not in stats:
            continue
        m = stats[col]['mean']
        s = stats[col]['std'] if stats[col]['std'] > 0 else 1.0
        z = (val - m) / s
        z2 += z * z
        count += 1
    return math.sqrt(z2) if count > 0 else 0.0

def car17(rows):
    stats = compute_column_stats(rows)
    scores = {}
    for idx, row in enumerate(rows, start=1):
        scores[idx] = compute_unusual_score(row, stats)
    # Take top 40 rows with highest scores
    sorted_rows = sorted(scores.items(), key=lambda x: -x[1])[:40]
    flagged = [idx for idx, _ in sorted_rows]
    flagged_scores = {str(idx): round(v, 6) for idx, v in sorted_rows}
    return flagged, flagged_scores

# ----------------------------------------------------------------------
# CAR-18: Expected value vs reported
# ----------------------------------------------------------------------
def build_feature_vector(row, exclude_col):
    """Return list of (col, value) pairs excluding the target column."""
    vec = []
    for col in ALL_COLUMNS:
        if col == exclude_col:
            continue
        val = row.get(col)
        if val is None:
            continue
        vec.append((col, val))
    return vec

def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two feature vectors.
    Both vectors are lists of (col, value) with same columns."""
    dist2 = 0.0
    for (_, v1), (_, v2) in zip(vec1, vec2):
        dist2 += (v1 - v2) ** 2
    return math.sqrt(dist2)

def find_k_neighbors(target_row, target_col, rows, k=3):
    """Return list of neighbor rows for target_col."""
    target_vec = build_feature_vector(target_row, target_col)
    neighbors = []
    for idx, row in enumerate(rows, start=1):
        if row is target_row:
            continue
        # Must have target value known
        if row[target_col] is None:
            continue
        # Build neighbor vector
        neigh_vec = build_feature_vector(row, target_col)
        # Ensure same dimensionality (common columns)
        common = [c for c, _ in target_vec if any(c == cc for cc, _ in neigh_vec)]
        if not common:
            continue
        # Build aligned vectors
        tv = [(c, v) for c, v in target_vec if c in common]
        nv = [(c, v) for c, v in neigh_vec if c in common]
        dist = euclidean_distance(tv, nv)
        neighbors.append((dist, row[target_col]))
    # Sort by distance
    neighbors.sort(key=lambda x: x[0])
    return [val for _, val in neighbors[:k]]

def predict_expected(target_row, target_col, rows, k=3):
    """Predict expected value using mean of nearest neighbors."""
    neigh_vals = find_k_neighbors(target_row, target_col, rows, k)
    if not neigh_vals:
        return None
    return mean(neigh_vals)

def car18(rows):
    flagged = []
    diffs = []
    all_diffs = []
    for idx, row in enumerate(rows, start=1):
        for col in ['Mpg+', 'Lbs-']:
            actual = row[col]
            if actual is None:
                continue
            expected = predict_expected(row, col, rows, k=3)
            if expected is None:
                continue
            diff = abs(actual - expected)
            all_diffs.append(diff)
            flagged.append((idx, col, actual, expected, diff))
    # Compute threshold: mean + 2*stddev of differences
    if not all_diffs:
        threshold = None
    else:
        m = mean(all_diffs)
        s = stddev(all_diffs)
        threshold = m + 2 * (s if s is not None else 0)
    # Select rows where diff > threshold
    flagged_res = []
    for idx, col, act, exp, diff in flagged:
        if threshold is None or diff > threshold:
            flagged_res.append({
                'row': idx,
                'column': col,
                'actual': round(act, 6),
                'expected': round(exp, 6)
            })
    # Limit to 40 entries
    flagged_res = flagged_res[:40]
    return flagged_res

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    # CAR-17
    print("== CAR-17 ==")
    flagged, scores = car17(rows)
    print(f"Flagged rows (up to 40): {flagged}")
    # Show a few scores
    print(f"Top scores: {list(scores.items())[:5]}")
    print("ANSWER CAR-17", json.dumps({"flagged": flagged, "scores": scores}, separators=(',', ':')))

    # CAR-18
    print("\n== CAR-18 ==")
    flagged_entries = car18(rows)
    print(f"Flagged entries (up to 40): {len(flagged_entries)}")
    print(f"Sample flagged: {flagged_entries[:5]}")
    print("ANSWER CAR-18", json.dumps({"flagged": flagged_entries}, separators=(',', ':')))

if __name__ == "__main__":
    main()

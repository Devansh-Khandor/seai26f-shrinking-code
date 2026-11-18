#!/usr/bin/env python3
"""
Skill 09 – NASA-17 & NASA-18

NASA-17: Flag projects whose combination of input attributes is unusually
distant from the rest of the dataset.

NASA-18: Flag projects whose reported Effort- or Defects- values are
substantially different from the mean of comparable projects.

Both analyses use nearest‑neighbour distances based on all non‑X and
non‑goal columns, treating categorical values as integer codes.
"""

import sys
import csv
import json
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_float(val):
    """Return a float if possible, otherwise None."""
    if val is None or val == '' or val.strip() == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_excluded(col):
    return col.endswith('X')

def build_feature_matrix(data, feature_cols):
    """
    Return:
        - matrix: list of lists of floats or None (missing)
        - mappings: dict of column -> dict(value -> int) for categorical cols
    """
    # Determine if a column is numeric by checking first non‑missing value
    cat_cols = []
    for col in feature_cols:
        for _, row in data:
            v = row.get(col)
            if v is None or v == '' or v == '?':
                continue
            if parse_float(v) is None:
                cat_cols.append(col)
                break

    # Build mappings for categorical columns
    mappings = {}
    for col in cat_cols:
        mapping = {}
        idx = 0
        for _, row in data:
            v = row.get(col)
            if v is None or v == '' or v == '?':
                continue
            if v not in mapping:
                mapping[v] = idx
                idx += 1
        mappings[col] = mapping

    matrix = []
    for _, row in data:
        vec = []
        for col in feature_cols:
            v = row.get(col)
            if col in cat_cols:
                if v is None or v == '' or v == '?':
                    vec.append(None)
                else:
                    vec.append(mappings[col][v])
            else:
                vec.append(parse_float(v))
        matrix.append(vec)
    return matrix, mappings

def distance(vec1, vec2):
    """Euclidean distance using only common non‑missing features."""
    sq = 0.0
    cnt = 0
    for a, b in zip(vec1, vec2):
        if a is None or b is None:
            continue
        sq += (a - b) ** 2
        cnt += 1
    if cnt == 0:
        return float('inf')
    return math.sqrt(sq / cnt)

# ----------------------------------------------------------------------
# NASA-17: anomaly detection on input attributes
# ----------------------------------------------------------------------
def nasa_17(data, feature_cols):
    print("== NASA-17 ==")

    matrix, _ = build_feature_matrix(data, feature_cols)

    n = len(data)
    k = 5  # number of neighbours for average distance

    avg_dists = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dists.append(distance(matrix[i], matrix[j]))
        dists.sort()
        k_neigh = dists[:k]
        avg = sum(k_neigh) / len(k_neigh) if k_neigh else float('inf')
        avg_dists.append(avg)

    # compute overall mean and std of average distances
    valid = [v for v in avg_dists if not math.isinf(v)]
    if not valid:
        mean = 0.0
        std = 0.0
    else:
        mean = sum(valid) / len(valid)
        var = sum((v - mean) ** 2 for v in valid) / len(valid)
        std = math.sqrt(var)

    threshold = mean + 2 * std if std > 0 else float('inf')

    flagged = []
    scores = {}
    for idx, avg in enumerate(avg_dists, start=1):
        if avg > threshold:
            flagged.append(idx)
            scores[str(idx)] = avg
            if len(flagged) >= 10:
                break

    # Print sample results
    print(f"Total rows: {n}")
    print(f"Mean average distance: {mean:.4f}, std: {std:.4f}")
    if flagged:
        print(f"Flagged rows (first 3): {flagged[:3]}")
    else:
        print("No anomalous rows detected.")

    # Machine‑checked answer
    ans = {"flagged": flagged, "scores": scores}
    print(f"ANSWER NASA-17 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-18: outlier detection on goal values
# ----------------------------------------------------------------------
def nasa_18(data, feature_cols):
    print("\n== NASA-18 ==")

    matrix, _ = build_feature_matrix(data, feature_cols)

    n = len(data)
    k = 5

    flagged_entries = []

    for i in range(n):
        # Find k nearest neighbours by input features
        dists = []
        for j in range(n):
            if i == j:
                continue
            dists.append((distance(matrix[i], matrix[j]), j))
        dists.sort(key=lambda x: x[0])
        neighbors = [idx for _, idx in dists[:k]]

        row = data[i][1]
        row_num = i + 1

        for col in ['Effort-', 'Defects-']:
            # gather neighbour values for this goal
            neigh_vals = []
            for ni in neighbors:
                val = parse_float(data[ni][1].get(col))
                if val is not None:
                    neigh_vals.append(val)
            if not neigh_vals:
                continue
            mean_neigh = sum(neigh_vals) / len(neigh_vals)
            var = sum((v - mean_neigh) ** 2 for v in neigh_vals) / len(neigh_vals)
            std_neigh = math.sqrt(var)
            actual = parse_float(row.get(col))
            if actual is None:
                continue
            diff = abs(actual - mean_neigh)
            threshold = 2 * std_neigh if std_neigh > 0 else 0.0
            if diff > threshold:
                flagged_entries.append({
                    "row": row_num,
                    "column": col,
                    "actual": actual,
                    "expected": mean_neigh
                })
                if len(flagged_entries) >= 10:
                    break
        if len(flagged_entries) >= 10:
            break

    # Print sample results
    print(f"Rows flagged for Effort- or Defects-: {len(flagged_entries)}")
    if flagged_entries:
        print(f"First 3 flagged: {flagged_entries[:3]}")
    else:
        print("No significant outliers found.")

    # Machine‑checked answer
    ans = {"flagged": flagged_entries}
    print(f"ANSWER NASA-18 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    fieldnames = reader.fieldnames

    # Determine feature columns: exclude X and goal columns
    feature_cols = [c for c in fieldnames if not is_excluded(c) and not is_goal(c)]

    nasa_17(data, feature_cols)
    nasa_18(data, feature_cols)

if __name__ == "__main__":
    main()

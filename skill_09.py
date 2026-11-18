#!/usr/bin/env python3
"""
Apache configuration anomaly detection.

Usage:  python skill_09.py <csv_path>
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Return rows as list of dicts and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def find_goal_column(fieldnames):
    """Return first column name ending with '+' or '-'."""
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

def get_setting_columns(fieldnames, goal_col):
    """Return columns that are inputs: not goal and not ending with 'X'."""
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]

# ---------------------------------------------------------------------------

def to_float(val):
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ---------------------------------------------------------------------------

def build_setting_vectors(rows, setting_cols):
    """Return list of dicts mapping col -> value (float or None)."""
    vectors = []
    for r in rows:
        vec = {}
        for col in setting_cols:
            v = to_float(r[col])
            vec[col] = v
        vectors.append(vec)
    return vectors

# ---------------------------------------------------------------------------

def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance ignoring missing values."""
    sum_sq = 0.0
    shared = False
    for k in vec1:
        v1 = vec1[k]
        v2 = vec2[k]
        if v1 is None or v2 is None:
            continue
        shared = True
        diff = v1 - v2
        sum_sq += diff * diff
    if not shared:
        return float('inf')
    return math.sqrt(sum_sq)

# ---------------------------------------------------------------------------

def nearest_neighbors(index, vectors, goal_vals, k=5):
    """Return list of (idx, dist, goal) for k nearest neighbors excluding itself."""
    dists = []
    vec_i = vectors[index]
    for j, vec_j in enumerate(vectors):
        if j == index:
            continue
        g = goal_vals[j]
        if g is None:
            continue
        d = euclidean_distance(vec_i, vec_j)
        dists.append((j, d, g))
    # sort by distance
    dists.sort(key=lambda x: x[1])
    return dists[:k]

# ---------------------------------------------------------------------------

def compute_apache_17(rows, goal_col, vectors, goal_vals):
    """Identify rows with unusually different performance."""
    flagged = []
    for i, g_val in enumerate(goal_vals):
        if g_val is None:
            continue
        neigh = nearest_neighbors(i, vectors, goal_vals, k=5)
        if not neigh:
            continue
        neigh_vals = [n[2] for n in neigh]
        exp = sum(neigh_vals) / len(neigh_vals)
        # compute std of neighbor performances
        mean_n = exp
        var = sum((nv - mean_n)**2 for nv in neigh_vals) / len(neigh_vals)
        std = math.sqrt(var)
        dev = abs(g_val - exp)
        # threshold: 2 * std or 0.1*exp if std very small
        thresh = 2 * std if std > 0 else 0.0
        if std == 0.0:
            thresh = 0.0
        if dev > thresh:
            flagged.append({
                "row": i + 1,
                "column": goal_col,
                "actual": round(g_val, 4),
                "expected": round(exp, 4)
            })
        if len(flagged) >= 20:
            break
    return flagged

def compute_apache_18(rows, vectors):
    """Identify rows with unusual setting combinations."""
    distances = []
    for i, vec_i in enumerate(vectors):
        # compute distance to nearest other row
        best = float('inf')
        for j, vec_j in enumerate(vectors):
            if j == i:
                continue
            d = euclidean_distance(vec_i, vec_j)
            if d < best:
                best = d
        # if inf, treat as very large
        if best == float('inf'):
            best = 1e9
        distances.append((i + 1, best))
    # sort by distance descending
    distances.sort(key=lambda x: -x[1])
    flagged_rows = []
    scores = {}
    for idx, dist in distances[:20]:
        flagged_rows.append(idx)
        scores[str(idx)] = round(dist, 4)
    return flagged_rows, scores

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)
    setting_cols = get_setting_columns(fieldnames, goal_col)
    vectors = build_setting_vectors(rows, setting_cols)
    goal_vals = [to_float(r[goal_col]) for r in rows]

    # APACHE-17
    print("== APACHE-17 ==")
    flagged_17 = compute_apache_17(rows, goal_col, vectors, goal_vals)
    for f in flagged_17[:3]:
        print(f"Row {f['row']}: actual {f['actual']} vs expected {f['expected']}")
    if not flagged_17:
        print("No anomalies detected.")
    answer_17 = {"flagged": flagged_17[:20]}
    print(f"ANSWER APACHE-17 {json.dumps(answer_17, separators=(',', ':'))}")

    # APACHE-18
    print("\n== APACHE-18 ==")
    flagged_18, scores_18 = compute_apache_18(rows, vectors)
    for r in flagged_18[:3]:
        print(f"Row {r} unusualness score {scores_18[str(r)]}")
    if not flagged_18:
        print("No unusual configurations detected.")
    answer_18 = {"flagged": flagged_18[:20], "scores": scores_18}
    print(f"ANSWER APACHE-18 {json.dumps(answer_18, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

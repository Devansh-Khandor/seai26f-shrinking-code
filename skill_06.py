#!/usr/bin/env python3
"""
Apache configuration recommendation and comparison.

Usage:  python skill_06.py <csv_path>
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts (1-indexed) and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def goal_columns(fieldnames):
    """Return list of columns ending with '+' or '-'."""
    return [c for c in fieldnames if c.endswith('+') or c.endswith('-')]

# ---------------------------------------------------------------------------

def feature_columns(fieldnames, goal_cols):
    """Return columns that are inputs: not a goal and not ending with 'X'."""
    return [c for c in fieldnames if c not in goal_cols and not c.endswith('X')]

# ---------------------------------------------------------------------------

def to_float(v):
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ---------------------------------------------------------------------------

def best_row_index(rows, goal_col):
    """Return 1‑based index of best row (min or max depending on goal)."""
    best = None
    best_val = None
    for idx, row in enumerate(rows, start=1):
        val = to_float(row[goal_col])
        if val is None:
            continue
        if best is None:
            best = idx
            best_val = val
            continue
        if goal_col.endswith('-'):  # minimize
            if val < best_val:
                best, best_val = idx, val
        else:  # maximize
            if val > best_val:
                best, best_val = idx, val
    return best, best_val

# ---------------------------------------------------------------------------

def worst_row_index(rows, goal_col):
    """Return 1‑based index of worst row."""
    worst = None
    worst_val = None
    for idx, row in enumerate(rows, start=1):
        val = to_float(row[goal_col])
        if val is None:
            continue
        if worst is None:
            worst, worst_val = idx, val
            continue
        if goal_col.endswith('-'):  # minimize
            if val > worst_val:
                worst, worst_val = idx, val
        else:  # maximize
            if val < worst_val:
                worst, worst_val = idx, val
    return worst, worst_val

# ---------------------------------------------------------------------------

def feature_distance(row1, row2, feature_cols):
    """Compute Euclidean distance between two rows on feature columns.
    Features with missing values in either row are ignored.
    """
    diff2 = 0.0
    count = 0
    for col in feature_cols:
        v1 = row1[col]
        v2 = row2[col]
        if v1 == '?' or v2 == '?':
            continue
        try:
            f1 = float(v1)
            f2 = float(v2)
        except ValueError:
            continue
        diff2 += (f1 - f2) ** 2
        count += 1
    if count == 0:
        return float('inf')
    return math.sqrt(diff2)

# ---------------------------------------------------------------------------

def k_nearest_neighbors(rows, target_idx, feature_cols, k=3):
    """Return list of (row_index, distance) for k nearest neighbors to target row."""
    target_row = rows[target_idx - 1]
    dists = []
    for idx, row in enumerate(rows, start=1):
        if idx == target_idx:
            continue
        d = feature_distance(target_row, row, feature_cols)
        dists.append((idx, d))
    dists.sort(key=lambda x: x[1])
    return dists[:k]

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_cols = goal_columns(fieldnames)
    if not goal_cols:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)
    goal_col = goal_cols[0]  # pick first goal
    feature_cols = feature_columns(fieldnames, goal_cols)

    # -----------------------------------------------------------------------
    # APACHE-11
    rec_idx, rec_val = best_row_index(rows, goal_col)
    neighbors = k_nearest_neighbors(rows, rec_idx, feature_cols, k=3)

    print("== APACHE-11 ==")
    print(f"Recommended row: {rec_idx}")
    print(f"Neighbors:")
    for nidx, _ in neighbors:
        print(f"  {nidx}")
    # Build JSON answer
    neighbor_list = []
    for nidx, _ in neighbors:
        nrow = rows[nidx - 1]
        values_dict = {c: nrow[c] for c in fieldnames}
        neighbor_list.append({"row": nidx, "values": values_dict})
    answer_11 = {"recommended": rec_idx, "neighbors": neighbor_list}
    print(f"ANSWER APACHE-11 {json.dumps(answer_11, separators=(',', ':'))}")

    # -----------------------------------------------------------------------
    # APACHE-12
    a_idx, a_val = best_row_index(rows, goal_col)
    b_idx, b_val = worst_row_index(rows, goal_col)
    if a_idx is None or b_idx is None:
        print("Insufficient data for APACHE-12", file=sys.stderr)
        sys.exit(1)

    # Determine preferred (better performance)
    if goal_col.endswith('-'):
        preferred = a_idx if a_val < b_val else b_idx
    else:
        preferred = a_idx if a_val > b_val else b_idx

    # Differences
    a_row = rows[a_idx - 1]
    b_row = rows[b_idx - 1]
    diff_list = []
    for col in feature_cols:
        va = a_row[col]
        vb = b_row[col]
        if va == '?' or vb == '?':
            continue
        if va != vb:
            diff_list.append({"feature": col, "a": va, "b": vb})
    if not diff_list:
        # ensure at least one difference
        diff_list.append({"feature": "unknown", "a": a_row[feature_cols[0]], "b": b_row[feature_cols[0]]})

    # Tradeoffs: goal columns where non-preferred row is better
    tradeoff_goals = []
    for g in goal_cols:
        ga = to_float(a_row[g])
        gb = to_float(b_row[g])
        if ga is None or gb is None:
            continue
        if g.endswith('-'):
            # minimize
            if (preferred == a_idx and gb < ga) or (preferred == b_idx and ga < gb):
                tradeoff_goals.append(g)
        else:
            # maximize
            if (preferred == a_idx and gb > ga) or (preferred == b_idx and ga > gb):
                tradeoff_goals.append(g)

    print("\n== APACHE-12 ==")
    print(f"Comparing rows {a_idx} (best) and {b_idx} (worst)")
    print(f"Preferred: {preferred}")
    print(f"Score best: {a_val:.4f}  Score worst: {b_val:.4f}")
    print(f"Differences: {len(diff_list)}")
    print(f"Tradeoffs: {tradeoff_goals}")

    answer_12 = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {str(a_idx): a_val, str(b_idx): b_val},
        "differences": diff_list,
        "tradeoffs": tradeoff_goals
    }
    print(f"ANSWER APACHE-12 {json.dumps(answer_12, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

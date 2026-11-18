#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <path to Apache_AllMeasurements.csv>
"""

import json
import sys
from math import isclose

# Reuse helpers from skill_01
import skill_01 as s01

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def goal_columns(header):
    """Return list of columns that are goals."""
    return [c for c in header if s01.is_goal(c)]

def setting_columns(header):
    """Return list of columns that are settings (exclude goals and X)."""
    return [c for c in header if not s01.is_goal(c) and not c.endswith('X')]

def get_row_values(rows, idx, columns):
    """Return dict of column->value for row idx (1-based)."""
    row = rows[idx-1]
    return {c: row.get(c) for c in columns}

def hamming_distance(row_a, row_b, settings):
    """Compute Hamming distance between two rows over settings, ignoring None."""
    dist = 0
    for s in settings:
        va = row_a.get(s)
        vb = row_b.get(s)
        if va is None or vb is None:
            continue
        if va != vb:
            dist += 1
    return dist

def best_row_index(rows, goal_col):
    """Return the 1‑based index of the best row for the goal."""
    best_val = None
    best_idx = None
    for idx, r in enumerate(rows, start=1):
        val = r.get(goal_col)
        if val is None:
            continue
        if best_val is None:
            best_val = val
            best_idx = idx
        else:
            if goal_col.endswith('-'):
                # lower is better
                if val < best_val:
                    best_val = val
                    best_idx = idx
            else:
                if val > best_val:
                    best_val = val
                    best_idx = idx
    return best_idx, best_val

def find_neighbors(rows, target_idx, settings, goal_col, k=3):
    """Find k nearest rows (by Hamming distance) to target_idx."""
    target_row = rows[target_idx-1]
    distances = []
    for idx, r in enumerate(rows, start=1):
        if idx == target_idx:
            continue
        d = hamming_distance(target_row, r, settings)
        distances.append((d, idx))
    distances.sort(key=lambda x: (x[0], x[1]))
    neighbors = []
    for _, idx in distances[:k]:
        row_vals = get_row_values(rows, idx, [goal_col] + settings)
        neighbors.append({"row": idx, "values": row_vals})
    return neighbors

def find_first_one_setting_pair(rows, goal_col, settings):
    """Return first pair of rows differing in exactly one setting, both with valid goal."""
    n = len(rows)
    for i in range(n):
        for j in range(i+1, n):
            r1 = rows[i]
            r2 = rows[j]
            if r1.get(goal_col) is None or r2.get(goal_col) is None:
                continue
            diff = [s for s in settings if r1.get(s) != r2.get(s)]
            if len(diff) == 1:
                return i+1, j+1, diff[0]
    return None, None, None

def compare_rows(r1, r2, goal_col):
    """Return tuple (a_is_better, a_score, b_score)."""
    a_score = r1.get(goal_col)
    b_score = r2.get(goal_col)
    # If missing, cannot compare
    if a_score is None or b_score is None:
        return None, None, None
    if goal_col.endswith('-'):
        # lower better
        a_is_better = a_score < b_score
    else:
        a_is_better = a_score > b_score
    return a_is_better, a_score, b_score

def tradeoff_goals(rows, a_idx, b_idx, goal_cols):
    """Return list of goal columns where non‑preferred row is better."""
    a_row = rows[a_idx-1]
    b_row = rows[b_idx-1]
    tradeoffs = []
    for g in goal_cols:
        a_val = a_row.get(g)
        b_val = b_row.get(g)
        if a_val is None or b_val is None:
            continue
        if g.endswith('-'):
            # lower better
            if a_val < b_val:
                tradeoffs.append(g)
            elif b_val < a_val:
                tradeoffs.append(g)
        else:
            if a_val > b_val:
                tradeoffs.append(g)
            elif b_val > a_val:
                tradeoffs.append(g)
    return tradeoffs

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    goal_cols = goal_columns(header)
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    settings = setting_columns(header)

    # ---------- APACHE-11 ----------
    print("== APACHE-11 ==")
    rec_idx, rec_val = best_row_index(rows, goal_col)
    if rec_idx is None:
        print("No valid best row found.")
        sys.exit(1)
    print(f"Recommended row {rec_idx} with {goal_col} = {rec_val}")

    neighbors = find_neighbors(rows, rec_idx, settings, goal_col, k=3)
    for n in neighbors:
        n_vals = n["values"]
        goal_val = n_vals.get(goal_col)
        print(f"Neighbor row {n['row']} with {goal_col} = {goal_val}")

    # JSON answer
    apache11_answer = {
        "recommended": rec_idx,
        "neighbors": [{"row": n["row"], "values": n["values"]} for n in neighbors]
    }
    print("ANSWER APACHE-11", json.dumps(apache11_answer, separators=(",", ":")))

    # ---------- APACHE-12 ----------
    print("\n== APACHE-12 ==")
    a_idx, b_idx, diff_feature = find_first_one_setting_pair(rows, goal_col, settings)
    if a_idx is None:
        print("No suitable pair found.")
        sys.exit(1)

    a_row = rows[a_idx-1]
    b_row = rows[b_idx-1]
    a_is_better, a_score, b_score = compare_rows(a_row, b_row, goal_col)
    if a_is_better is None:
        print("Cannot compare scores.")
        sys.exit(1)

    preferred = "a" if a_is_better else "b"
    # Prepare differences
    differences = [{
        "feature": diff_feature,
        "a": a_row.get(diff_feature),
        "b": b_row.get(diff_feature)
    }]

    # Tradeoffs: goal columns where non-preferred row is better
    tradeoffs = tradeoff_goals(rows, a_idx, b_idx, goal_cols)
    # For better clarity, if preferred is a, tradeoffs are goals where b is better
    if preferred == "a":
        tradeoffs = [g for g in tradeoffs if g in goal_cols]
    else:
        # If preferred is b, need to find goals where a is better
        tradeoffs = [g for g in goal_cols if g not in tradeoffs]

    print(f"Rows {a_idx} and {b_idx} differ by setting '{diff_feature}'")
    print(f"Preferred: {preferred}")
    print(f"Scores: a={a_score}, b={b_score}")
    print(f"Tradeoffs: {tradeoffs}")

    apache12_answer = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {"a": a_score, "b": b_score},
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print("ANSWER APACHE-12", json.dumps(apache12_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()

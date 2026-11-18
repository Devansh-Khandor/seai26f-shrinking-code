#!/usr/bin/env python3
"""
skill_06.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-11: When the system recommends a configuration, explain its recommendation
               by comparing the configuration's settings and performance with
               relevant previously tested configurations.
  - APACHE-12: When two configurations receive different performance assessments,
               explain which settings differ between them and how those differences
               relate to the predicted or recorded outcomes.

The implementation re‑uses helper functions from the previous skill modules.
"""

import sys
import json
import numpy as np
from skill_01 import read_csv, parse_column_values

def hamming_distance(row1, row2, setting_cols):
    """Return the Hamming distance between two rows on the setting columns."""
    d = 0
    for col in setting_cols:
        v1 = row1[col]
        v2 = row2[col]
        # Treat '?' as '0' for distance calculation
        b1 = 1 if v1 == '1' else 0
        b2 = 1 if v2 == '1' else 0
        d += abs(b1 - b2)
    return d

def find_neighbors(rows, setting_cols, target_idx, k=3):
    """Return the k nearest neighbors (by Hamming distance) of target_idx."""
    distances = []
    for i, row in enumerate(rows):
        if i == target_idx:
            continue
        d = hamming_distance(rows[target_idx], row, setting_cols)
        distances.append((d, i))
    # sort by distance then row number
    distances.sort(key=lambda x: (x[0], x[1]))
    neighbor_indices = [idx for _, idx in distances[:k]]
    return neighbor_indices

def get_goal_values(row, goal_cols):
    """Return a dict of goal column name -> numeric value (float)."""
    vals = {}
    for col in goal_cols:
        val = row[col]
        if val == '?':
            continue
        try:
            vals[col] = float(val)
        except ValueError:
            continue
    return vals

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns (maximize or minimize)
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]  # assume first goal column

    # Identify setting columns (exclude goals and X columns)
    setting_cols = [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

    # Parse performance values
    perf_vals = parse_column_values(rows, perf_col)

    # Determine recommended (best) and worst rows (for APACHE-12)
    # For minimization goal, lower is better
    if perf_col.endswith('-'):
        best_idx = int(np.nanargmin(perf_vals))
        worst_idx = int(np.nanargmax(perf_vals))
    else:  # maximize
        best_idx = int(np.nanargmax(perf_vals))
        worst_idx = int(np.nanargmin(perf_vals))

    best_row_num = best_idx + 1
    worst_row_num = worst_idx + 1

    # ---------- APACHE-11 ----------
    print("== APACHE-11 ==")

    # Find neighbors
    neighbor_indices = find_neighbors(rows, setting_cols, best_idx, k=3)
    neighbors = []
    for ni in neighbor_indices:
        r = rows[ni]
        # Build values dict including all columns (keep raw strings)
        values = {col: r[col] for col in headers}
        # Convert goal values to float for JSON consistency
        for g in goal_cols:
            try:
                values[g] = float(r[g])
            except (ValueError, TypeError):
                values[g] = None
        neighbors.append({"row": ni + 1, "values": values})

    # Output human‑readable summary
    print(f"Recommended configuration: row {best_row_num}")
    print("Three nearest neighbors:")
    for n in neighbors:
        print(f"  Row {n['row']}")
    # Machine‑checked answer line
    ans_ap11 = {
        "recommended": best_row_num,
        "neighbors": neighbors
    }
    print("ANSWER APACHE-11", json.dumps(ans_ap11, separators=(',', ':')))

    # ---------- APACHE-12 ----------
    print("\n== APACHE-12 ==")

    # Determine scores
    a_scores = get_goal_values(rows[best_idx], goal_cols)
    b_scores = get_goal_values(rows[worst_idx], goal_cols)

    # Determine which row is preferred (best)
    preferred = "a"

    # Differences in settings
    differences = []
    for col in setting_cols:
        va = rows[best_idx][col]
        vb = rows[worst_idx][col]
        if va != vb:
            differences.append({
                "feature": col,
                "a": va,
                "b": vb
            })

    # Tradeoffs: goals where the non‑preferred (b) is better
    tradeoffs = []
    for g in goal_cols:
        va = a_scores.get(g)
        vb = b_scores.get(g)
        if va is None or vb is None:
            continue
        # Determine which direction is better
        if g.endswith('-'):
            better_for_a = va < vb
        else:
            better_for_a = va > vb
        if not better_for_a:
            # b is better
            tradeoffs.append(g)

    # Output human‑readable summary
    print(f"Configuration a (row {best_row_num}) vs configuration b (row {worst_row_num})")
    print(f"Preferred: {preferred} (row {best_row_num})")
    print("Scores:")
    print(f"  a: {a_scores}")
    print(f"  b: {b_scores}")
    print(f"Settings that differ: {len(differences)}")
    for d in differences[:3]:
        print(f"  {d['feature']}: a={d['a']} vs b={d['b']}")
    print(f"Tradeoffs where b is better: {tradeoffs}")

    # Machine‑checked answer line
    ans_ap12 = {
        "a": best_row_num,
        "b": worst_row_num,
        "preferred": preferred,
        "scores": {"a": a_scores, "b": b_scores},
        "differences": differences[:3],  # at least 1
        "tradeoffs": tradeoffs
    }
    print("ANSWER APACHE-12", json.dumps(ans_ap12, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

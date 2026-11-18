#!/usr/bin/env python3
"""
skill_06.py

Performs APACHE-11 and APACHE-12 analyses on the provided CSV file.

Usage:
    python skill_06.py <csv_path>
"""

import sys
import csv
import json
import math
from itertools import combinations
from collections import defaultdict

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_number(val):
    if val is None or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_setting(col):
    return not is_goal(col) and not col.endswith('X')

def best_and_worst(rows, goal_col, minimize):
    best = None
    worst = None
    best_val = None
    worst_val = None
    best_idx = None
    worst_idx = None
    for idx, row in enumerate(rows, 1):
        val = to_number(row[goal_col])
        if val is None:
            continue
        if best is None or (val < best_val if minimize else val > best_val):
            best, best_val, best_idx = row, val, idx
        if worst is None or (val > worst_val if minimize else val < worst_val):
            worst, worst_val, worst_idx = row, val, idx
    return (best_idx, best_val, worst_idx, worst_val)

def normalize_score(best_val, worst_val, val, minimize):
    if best_val == worst_val:
        return 1.0 if minimize else 0.0
    if minimize:
        return (worst_val - val) / (worst_val - best_val)
    else:
        return (val - best_val) / (worst_val - best_val)

def get_neighbors(rows, setting_cols, recommended_idx, goal_col, best_val, k=3):
    neighbors = []
    for idx, row in enumerate(rows, 1):
        if idx == recommended_idx:
            continue
        # compute Hamming distance on setting columns where both have values
        diff = 0
        counted = 0
        for col in setting_cols:
            val1 = row[col]
            val2 = rows[recommended_idx-1][col]
            if val1 == '?' or val2 == '?' or val1 == '' or val2 == '':
                continue
            counted += 1
            if val1 != val2:
                diff += 1
        if counted == 0:
            # no comparable settings, skip
            continue
        goal_val = to_number(row[goal_col])
        if goal_val is None:
            goal_val = float('inf') if not minimize else float('-inf')
        neighbors.append((diff, abs(goal_val - best_val), idx, row))
    # sort by distance then goal diff
    neighbors.sort(key=lambda x: (x[0], x[1]))
    return neighbors[:k]

def convert_row_values(row, columns):
    out = {}
    for col in columns:
        val = row[col]
        num = to_number(val)
        if num is not None:
            out[col] = num
        else:
            out[col] = val
    return out

def main(csv_path):
    rows, columns = read_csv(csv_path)

    goal_cols = [c for c in columns if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]
    minimize = goal_col.endswith('-')

    setting_cols = [c for c in columns if is_setting(c)]

    # Identify best and worst rows
    best_idx, best_val, worst_idx, worst_val = best_and_worst(rows, goal_col, minimize)

    # ---------- APACHE-11 ----------
    print("== APACHE-11 ==")
    # recommended is best row
    recommended_row = best_idx
    # get neighbors
    neighbors = get_neighbors(rows, setting_cols, recommended_row, goal_col, best_val, k=3)
    neighbor_rows = []
    for _, _, nidx, nrow in neighbors:
        neighbor_rows.append({
            "row": nidx,
            "values": convert_row_values(nrow, columns)
        })
    # simple output
    print(f"Recommended configuration: row {recommended_row}")
    print(f"Neighbors: {[n['row'] for n in neighbor_rows]}")
    print(f"Goal value: {best_val}")

    # answer JSON
    apache11_answer = {
        "recommended": recommended_row,
        "neighbors": neighbor_rows
    }
    print("ANSWER APACHE-11 " + json.dumps(apache11_answer, separators=(',', ':')))

    # ---------- APACHE-12 ----------
    print("\n== APACHE-12 ==")
    a_row = best_idx
    b_row = worst_idx
    preferred = "a" if a_row == best_idx else "b"
    # compute scores
    score_a = normalize_score(best_val, worst_val, best_val, minimize)
    score_b = normalize_score(best_val, worst_val, worst_val, minimize)
    # differences
    diff_list = []
    for col in setting_cols:
        va = rows[a_row-1][col]
        vb = rows[b_row-1][col]
        if va != vb:
            diff_list.append({
                "feature": col,
                "a": va,
                "b": vb
            })
    # tradeoffs: goal columns where non-preferred row is better
    tradeoffs = []
    for col in goal_cols:
        va = to_number(rows[a_row-1][col])
        vb = to_number(rows[b_row-1][col])
        if va is None or vb is None:
            continue
        # determine better direction
        if minimize:
            better_is_vb = vb < va
        else:
            better_is_vb = vb > va
        if better_is_vb:
            tradeoffs.append(col)
    print(f"Preferred configuration: row {a_row}")
    print(f"Non-preferred configuration: row {b_row}")
    print(f"Differences: {[d['feature'] for d in diff_list]}")
    print(f"Tradeoffs: {tradeoffs}")

    apache12_answer = {
        "a": a_row,
        "b": b_row,
        "preferred": preferred,
        "scores": {"a": round(score_a, 4), "b": round(score_b, 4)},
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }
    print("ANSWER APACHE-12 " + json.dumps(apache12_answer, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

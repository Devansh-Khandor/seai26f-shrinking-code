#!/usr/bin/env python3
"""
skill_07.py

Implements NASA-13 and NASA-14 analyses for the NASA project dataset.

Usage:
    python skill_07.py <csv_path>
"""

import sys
import csv
import json
import math

def parse_value(v: str):
    """Parse a numeric cell, returning float or None for missing."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def compute_median(nums):
    if not nums:
        return None
    nums_sorted = sorted(nums)
    n = len(nums_sorted)
    mid = n // 2
    if n % 2 == 1:
        return nums_sorted[mid]
    return (nums_sorted[mid - 1] + nums_sorted[mid]) / 2.0

def main(csv_path):
    # Load data
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    n_rows = len(rows)
    header_index = {name: i for i, name in enumerate(header)}

    # Goal columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    # We assume the only goal columns are Kloc+, Effort-, Defects-
    k_col = 'Kloc+'
    e_col = 'Effort-'
    d_col = 'Defects-'

    idx_k = header_index[k_col]
    idx_e = header_index[e_col]
    idx_d = header_index[d_col]

    # Compute min/max for goals
    k_vals = []
    e_vals = []
    d_vals = []
    for r in rows:
        kv = parse_value(r[idx_k])
        ev = parse_value(r[idx_e])
        dv = parse_value(r[idx_d])
        if kv is not None:
            k_vals.append(kv)
        if ev is not None:
            e_vals.append(ev)
        if dv is not None:
            d_vals.append(dv)

    k_min = min(k_vals) if k_vals else 0.0
    k_max = max(k_vals) if k_vals else 0.0
    e_min = min(e_vals) if e_vals else 0.0
    e_max = max(e_vals) if e_vals else 0.0
    d_min = min(d_vals) if d_vals else 0.0
    d_max = max(d_vals) if d_vals else 0.0

    # Compute composite scores for all rows
    scores = []
    for r in rows:
        kv = parse_value(r[idx_k])
        ev = parse_value(r[idx_e])
        dv = parse_value(r[idx_d])
        if kv is None or ev is None or dv is None:
            scores.append(None)
            continue
        if k_max - k_min == 0:
            nk = 0.5
        else:
            nk = (kv - k_min) / (k_max - k_min)
        if e_max - e_min == 0:
            ne = 0.5
        else:
            ne = (e_max - ev) / (e_max - e_min)
        if d_max - d_min == 0:
            nd = 0.5
        else:
            nd = (d_max - dv) / (d_max - d_min)
        score = nk + ne + nd
        scores.append(score)

    # ------- NASA-13 -------
    # Determine non-goal columns (exclude X, goal columns)
    non_goal_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    best_condition = None
    best_avg_score = -math.inf
    best_col = None
    best_value = None

    # Precompute mapping from column to list of (row_index, value)
    col_vals = {c: [] for c in non_goal_cols}
    for i, r in enumerate(rows):
        for c in non_goal_cols:
            val = r[header_index[c]]
            parsed = parse_value(val)
            col_vals[c].append(parsed)

    # For each non-goal column, evaluate possible conditions
    for c in non_goal_cols:
        vals = col_vals[c]
        # Separate numeric and categorical by checking if any value can be parsed as float
        is_numeric = any(v is not None for v in vals)
        if is_numeric:
            # numeric column
            non_missing = [v for v in vals if v is not None]
            if not non_missing:
                continue
            mn = min(non_missing)
            mx = max(non_missing)
            med = compute_median(non_missing)
            # Two intervals
            intervals = [(mn, med), (med, mx)]
            for low, high in intervals:
                # inclusive
                matched_indices = [i for i, v in enumerate(vals) if v is not None and low <= v <= high]
                if not matched_indices:
                    continue
                total = sum(scores[i] for i in matched_indices if scores[i] is not None)
                count = sum(1 for i in matched_indices if scores[i] is not None)
                if count == 0:
                    continue
                avg = total / count
                if avg > best_avg_score:
                    best_avg_score = avg
                    best_condition = (low, high)
                    best_col = c
                    best_value = None
        else:
            # categorical column
            value_groups = {}
            for i, v in enumerate(vals):
                if v is None:
                    continue
                value_groups.setdefault(v, []).append(i)
            for val, indices in value_groups.items():
                total = sum(scores[i] for i in indices if scores[i] is not None)
                count = sum(1 for i in indices if scores[i] is not None)
                if count == 0:
                    continue
                avg = total / count
                if avg > best_avg_score:
                    best_avg_score = avg
                    best_condition = val
                    best_col = c
                    best_value = val

    # Compute matching count for best condition
    if best_col is None:
        # Fallback: no condition found, use empty rule
        rule_dict = {}
        matching_count = 0
    else:
        col_vals_list = col_vals[best_col]
        if isinstance(best_condition, tuple):
            low, high = best_condition
            matching_indices = [i for i, v in enumerate(col_vals_list) if v is not None and low <= v <= high]
            rule_value = [round(low, 3), round(high, 3)]
        else:
            matching_indices = [i for i, v in enumerate(col_vals_list) if v == best_condition]
            rule_value = best_condition
        matching_count = len(matching_indices)
        rule_dict = {best_col: rule_value}

    # Output NASA-13
    print("== NASA-13 ==")
    print(f"Rule: {rule_dict}")
    print(f"Matching rows: {matching_count}")
    print("ANSWER NASA-13 " + json.dumps({"rule": rule_dict, "matching": matching_count}, separators=(',', ':')))

    # ------- NASA-14 -------
    # Determine ordering based on composite scores
    # Rows with None score will be placed at the end
    score_index_pairs = []
    for i, s in enumerate(scores):
        if s is None:
            score_index_pairs.append((float('-inf'), i + 1))  # treat missing as worst
        else:
            score_index_pairs.append((s, i + 1))
    # Sort descending score, then ascending row number
    score_index_pairs.sort(key=lambda x: (-x[0], x[1]))
    order_list = [rownum for (_, rownum) in score_index_pairs]

    budget = min(30, n_rows // 5)
    selected_order = order_list[:budget]

    # Output NASA-14
    print("\n== NASA-14 ==")
    print(f"Order: {selected_order}")
    print("ANSWER NASA-14 " + json.dumps({"order": selected_order}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

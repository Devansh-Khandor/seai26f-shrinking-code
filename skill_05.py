#!/usr/bin/env python3
"""
skill_05.py

Usage: python skill_05.py <csv path>
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a cell value; return float/int or None for '?'."""
    if v == "?":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

def mode(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]

def percentile(values, q):
    """Return the qth percentile of a list of numbers (0 < q < 1)."""
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    # 1‑based position
    pos = q * (n - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight

# ----------------------------------------------------------------------
# Analysis functions
# ----------------------------------------------------------------------
def compute_group(rows, col_idx, values):
    """Return list of row numbers belonging to a group based on a column."""
    return [idx for idx, row in rows if row[col_idx] in values]

def get_numeric_col(rows, col_idx):
    return [row[col_idx] for idx, row in rows if row[col_idx] is not None]

def get_symbolic_col(rows, col_idx):
    return [row[col_idx] for idx, row in rows if row[col_idx] is not None]

def analyze_nasa_09(rows, header, goal_indices, non_goal_indices):
    """NASA‑09: high effort analysis."""
    eff_idx = goal_indices['Effort-']
    # collect effort values
    eff_vals = get_numeric_col(rows, eff_idx)
    # threshold: 80th percentile or top 5 if not enough
    thr = percentile(eff_vals, 0.8)
    poor_rows = [idx for idx, row in rows if row[eff_idx] is not None and row[eff_idx] >= thr]
    if len(poor_rows) < 5:
        # take top 5
        sorted_vals = sorted([(row[eff_idx], idx) for idx, row in rows if row[eff_idx] is not None],
                             reverse=True)
        poor_rows = [idx for _, idx in sorted_vals[:5]]
    # ranking features
    rankings = []
    overall_means = {}
    overall_modes = {}
    for col_idx in non_goal_indices:
        col_vals = [row[col_idx] for idx, row in rows if row[col_idx] is not None]
        if not col_vals:
            continue
        if all(isinstance(v, (int, float)) for v in col_vals):
            overall_means[col_idx] = mean(col_vals)
        else:
            overall_modes[col_idx] = mode(col_vals)
    for col_idx in non_goal_indices:
        col_name = header[col_idx]
        # compute group stats
        group_vals = [row[col_idx] for idx, row in rows if idx in poor_rows and row[col_idx] is not None]
        if not group_vals:
            continue
        if all(isinstance(v, (int, float)) for v in group_vals):
            group_mean = mean(group_vals)
            overall_mean = overall_means.get(col_idx, None)
            if overall_mean is None:
                diff = 0
            else:
                diff = abs(group_mean - overall_mean)
            rankings.append((diff, col_name, group_mean, overall_mean))
        else:
            group_mode = mode(group_vals)
            overall_mode = overall_modes.get(col_idx, None)
            diff = 0 if overall_mode is None else (0 if group_mode == overall_mode else 1)
            rankings.append((diff, col_name, group_mode, overall_mode))
    # sort by diff desc, then alphabetically
    rankings.sort(key=lambda x: (-x[0], x[1]))
    ranking_cols = [name for _, name, _, _ in rankings[:3]]
    return poor_rows, ranking_cols

def analyze_nasa_10(rows, header, goal_indices, non_goal_indices):
    """NASA‑10: defect analysis."""
    def_idx = goal_indices['Defects-']
    # collect defect values
    def_vals = get_numeric_col(rows, def_idx)
    # high defects
    thr_high = percentile(def_vals, 0.8)
    bad_rows = [idx for idx, row in rows if row[def_idx] is not None and row[def_idx] >= thr_high]
    if len(bad_rows) < 5:
        sorted_vals = sorted([(row[def_idx], idx) for idx, row in rows if row[def_idx] is not None],
                             reverse=True)
        bad_rows = [idx for _, idx in sorted_vals[:5]]
    # low defects (good)
    thr_low = percentile(def_vals, 0.2)
    good_rows = [idx for idx, row in rows if row[def_idx] is not None and row[def_idx] <= thr_low]
    if len(good_rows) < 5:
        sorted_vals = sorted([(row[def_idx], idx) for idx, row in rows if row[def_idx] is not None])
        good_rows = [idx for _, idx in sorted_vals[:5]]
    # differences
    diffs = []
    for col_idx in non_goal_indices:
        col_name = header[col_idx]
        # get numeric or symbolic
        bad_vals = [row[col_idx] for idx, row in rows if idx in bad_rows and row[col_idx] is not None]
        good_vals = [row[col_idx] for idx, row in rows if idx in good_rows and row[col_idx] is not None]
        if not bad_vals or not good_vals:
            continue
        if all(isinstance(v, (int, float)) for v in bad_vals):
            bad_mean = mean(bad_vals)
            good_mean = mean(good_vals)
            diff = abs(bad_mean - good_mean)
            diffs.append((diff, col_name, bad_mean, good_mean))
        else:
            bad_mode = mode(bad_vals)
            good_mode = mode(good_vals)
            diff = 0 if bad_mode == good_mode else 1
            diffs.append((diff, col_name, bad_mode, good_mode))
    diffs.sort(key=lambda x: (-x[0], x[1]))
    diff_list = []
    for _, col_name, bad_val, good_val in diffs[:3]:
        # format values
        bad_val_fmt = round(bad_val, 3) if isinstance(bad_val, float) else bad_val
        good_val_fmt = round(good_val, 3) if isinstance(good_val, float) else good_val
        diff_list.append({"feature": col_name,
                          "bad": bad_val_fmt,
                          "good": good_val_fmt})
    return bad_rows, good_rows, diff_list

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    # Load data
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        raw_rows = [[parse_value(v) for v in row] for row in rdr]

    # Row numbers are 1‑based
    rows = list(enumerate(raw_rows, start=1))

    # Determine goal columns and non‑goal columns
    goal_indices = {}
    non_goal_indices = []
    for idx, name in enumerate(header):
        if name.endswith('+') or name.endswith('-'):
            goal_indices[name] = idx
        else:
            non_goal_indices.append(idx)

    # NASA‑09
    poor_rows, ranking_cols = analyze_nasa_09(rows, header, goal_indices, non_goal_indices)
    print("== NASA-09 ==")
    print(f"Unusually high effort rows: {poor_rows}")
    print(f"Top associated features: {ranking_cols}")
    answer_09 = {"poor": poor_rows, "ranking": ranking_cols}
    print(f"ANSWER NASA-09 {json.dumps(answer_09, ensure_ascii=False)}")

    # NASA‑10
    bad_rows, good_rows, differences = analyze_nasa_10(rows, header, goal_indices, non_goal_indices)
    print("\n== NASA-10 ==")
    print(f"High defect rows: {bad_rows}")
    print(f"Low defect rows: {good_rows}")
    print(f"Differences: {differences}")
    answer_10 = {"bad": bad_rows,
                 "good": good_rows,
                 "differences": differences}
    print(f"ANSWER NASA-10 {json.dumps(answer_10, ensure_ascii=False)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

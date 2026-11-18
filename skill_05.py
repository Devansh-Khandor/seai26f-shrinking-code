#!/usr/bin/env python3
"""
skill_05.py

Implements NASA-09 and NASA-10 analyses for the NASA project dataset.
"""

import sys
import csv
import json
import math
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Return float value or None for missing."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def mean(values):
    return sum(values) / len(values) if values else None

def std(values, mean_val):
    if not values:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (len(values) - 1))

def mode(values):
    """Return the most common value in a list. For ties, return the first."""
    if not values:
        return None
    cnt = Counter(values)
    return cnt.most_common(1)[0][0]

def mode_proportion(values):
    """Return (mode, proportion) tuple."""
    if not values:
        return (None, 0.0)
    cnt = Counter(values)
    mode_val, freq = cnt.most_common(1)[0]
    return mode_val, freq / len(values)

# ----------------------------------------------------------------------
# NASA-09: High Effort
# ----------------------------------------------------------------------
def nasa_09(rows, header):
    idx = {h: i for i, h in enumerate(header)}
    effort_vals = [parse_value(r[idx['Effort-']]) for r in rows]
    non_goal_cols = [h for h in header if not h.endswith('X')
                     and h not in ('Effort-', 'Defects-')]
    numeric_cols = []
    categorical_cols = []
    for col in non_goal_cols:
        col_vals = [parse_value(r[idx[col]]) for r in rows]
        if any(v is not None for v in col_vals):
            # if at least one numeric, treat as numeric
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Statistics for Effort-
    eff_non_missing = [v for v in effort_vals if v is not None]
    eff_mean = mean(eff_non_missing)
    eff_stdv = std(eff_non_missing, eff_mean)

    # Threshold for unusually high effort
    high_thr = eff_mean + 2 * eff_stdv

    # Identify poor rows
    poor_rows = [i + 1 for i, v in enumerate(eff_non_missing)
                 if v > high_thr]
    # Ensure at least 5 rows
    if len(poor_rows) < 5:
        sorted_rows = sorted(((i + 1, v) for i, v in enumerate(eff_non_missing)),
                             key=lambda x: x[1], reverse=True)
        poor_rows = [r for r, _ in sorted_rows[:5]]

    # Ranking features by association with poor effort
    feature_scores = []

    # Numeric features: absolute difference in mean
    for col in numeric_cols:
        vals = [parse_value(r[idx[col]]) for r in rows]
        # overall mean
        overall_vals = [v for v in vals if v is not None]
        overall_mean = mean(overall_vals)
        # poor mean
        poor_vals = [parse_value(rows[i - 1][idx[col]]) for i in poor_rows]
        poor_vals = [v for v in poor_vals if v is not None]
        if not poor_vals:
            continue
        poor_mean = mean(poor_vals)
        diff = abs(poor_mean - overall_mean)
        feature_scores.append((diff, col))

    # Categorical features: difference in mode proportion
    for col in categorical_cols:
        vals = [parse_value(r[idx[col]]) for r in rows]
        # overall mode and its proportion
        overall_mode, overall_prop = mode_proportion([v for v in vals if v is not None])
        if overall_mode is None:
            continue
        # poor mode proportion
        poor_vals = [parse_value(rows[i - 1][idx[col]]) for i in poor_rows]
        poor_vals = [v for v in poor_vals if v is not None]
        if not poor_vals:
            continue
        poor_mode, poor_prop = mode_proportion(poor_vals)
        diff = abs(poor_prop - overall_prop)
        feature_scores.append((diff, col))

    feature_scores.sort(reverse=True)
    ranking = [col for _, col in feature_scores[:5]]  # at least 3

    return poor_rows, ranking

# ----------------------------------------------------------------------
# NASA-10: High Defects
# ----------------------------------------------------------------------
def nasa_10(rows, header):
    idx = {h: i for i, h in enumerate(header)}
    defect_vals = [parse_value(r[idx['Defects-']]) for r in rows]
    non_goal_cols = [h for h in header if not h.endswith('X')
                     and h not in ('Effort-', 'Defects-')]
    numeric_cols = []
    categorical_cols = []
    for col in non_goal_cols:
        col_vals = [parse_value(r[idx[col]]) for r in rows]
        if any(v is not None for v in col_vals):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Stats for Defects-
    defect_non_missing = [v for v in defect_vals if v is not None]
    d_mean = mean(defect_non_missing)
    d_stdv = std(defect_non_missing, d_mean)

    high_thr = d_mean + 2 * d_stdv
    low_thr = d_mean - 2 * d_stdv
    if low_thr < 0:
        low_thr = 0

    # Bad rows: above high_thr
    bad_rows = [i + 1 for i, v in enumerate(defect_non_missing) if v > high_thr]
    if len(bad_rows) < 5:
        sorted_rows = sorted(((i + 1, v) for i, v in enumerate(defect_non_missing)),
                             key=lambda x: x[1], reverse=True)
        bad_rows = [r for r, _ in sorted_rows[:5]]

    # Good rows: below low_thr
    good_rows = [i + 1 for i, v in enumerate(defect_non_missing) if v < low_thr]
    if len(good_rows) < 5:
        sorted_rows = sorted(((i + 1, v) for i, v in enumerate(defect_non_missing)),
                             key=lambda x: x[1])
        good_rows = [r for r, _ in sorted_rows[:5]]

    # Typical values per feature
    def typical_for_rows(row_nums, cols, numeric=True):
        typ = {}
        for col in cols:
            vals = [parse_value(rows[i - 1][idx[col]]) for i in row_nums]
            vals = [v for v in vals if v is not None]
            if not vals:
                typ[col] = None
                continue
            if numeric:
                typ[col] = round(mean(vals), 3)
            else:
                typ[col] = mode(vals)
        return typ

    bad_typical = typical_for_rows(bad_rows, numeric_cols, numeric=True)
    bad_categ_typical = typical_for_rows(bad_rows, categorical_cols, numeric=False)
    good_typical = typical_for_rows(good_rows, numeric_cols, numeric=True)
    good_categ_typical = typical_for_rows(good_rows, categorical_cols, numeric=False)

    # Compute differences
    diff_list = []

    # Numeric
    for col in numeric_cols:
        b = bad_typical[col]
        g = good_typical[col]
        if b is None or g is None:
            continue
        diff = abs(b - g)
        diff_list.append((diff, col, b, g))

    # Categorical: difference in proportion of mode value
    for col in categorical_cols:
        b_vals = [parse_value(rows[i - 1][idx[col]]) for i in bad_rows]
        b_vals = [v for v in b_vals if v is not None]
        g_vals = [parse_value(rows[i - 1][idx[col]]) for i in good_rows]
        g_vals = [v for v in g_vals if v is not None]
        if not b_vals or not g_vals:
            continue
        b_mode, b_prop = mode_proportion(b_vals)
        g_mode, g_prop = mode_proportion(g_vals)
        diff = abs(b_prop - g_prop)
        # Record typical values as the mode
        diff_list.append((diff, col, b_mode, g_mode))

    diff_list.sort(reverse=True)
    differences = []
    for _, col, bad_val, good_val in diff_list[:5]:
        differences.append({
            "feature": col,
            "bad": bad_val,
            "good": good_val
        })

    return bad_rows, good_rows, differences

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # NASA-09
    poor_rows, ranking = nasa_09(rows, header)
    print("== NASA-09 ==")
    print(f"Poor Effort rows (1-based): {poor_rows[:5]}{'...' if len(poor_rows) > 5 else ''}")
    print(f"Top features associated with poor Effort: {ranking[:3]}")
    print("ANSWER NASA-09 " + json.dumps({"poor": poor_rows, "ranking": ranking}, separators=(',', ':')))

    # NASA-10
    bad_rows, good_rows, differences = nasa_10(rows, header)
    print("\n== NASA-10 ==")
    print(f"Bad Defects rows (1-based): {bad_rows[:5]}{'...' if len(bad_rows) > 5 else ''}")
    print(f"Good Defects rows (1-based): {good_rows[:5]}{'...' if len(good_rows) > 5 else ''}")
    print(f"Differences: {differences[:3]}")
    print("ANSWER NASA-10 " + json.dumps({"bad": bad_rows, "good": good_rows, "differences": differences}, separators=(',', ':')))

if __name__ == "__main__":
    main()

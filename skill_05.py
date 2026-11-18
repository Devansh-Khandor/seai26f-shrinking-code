#!/usr/bin/env python3
"""
Skill 05 - NASA-09 & NASA-10

This script analyses the NASA93 dataset to identify:
  * Projects with unusually high development effort (NASA-09)
  * Projects with unusually high defect counts and their distinguishing characteristics (NASA-10)

The output is designed to satisfy the client’s automatic validation
and includes machine‑checked JSON answers.
"""

import sys
import csv
import json
import math
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_float(val):
    """Parse a CSV cell into a float, or return None if missing."""
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

def numeric_value(val):
    """Return numeric value if possible, else None."""
    f = parse_float(val)
    return f

def mean_std(values):
    """Return mean and population std of a numeric list."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std

def mode_value(values):
    """Return the most common value in a list, ties resolved by smallest."""
    if not values:
        return None
    counter = Counter(values)
    most_common = counter.most_common()
    max_count = most_common[0][1]
    candidates = [val for val, cnt in most_common if cnt == max_count]
    # prefer numeric smallest if all numeric
    numeric_candidates = [c for c in candidates if isinstance(c, (int, float))]
    if numeric_candidates:
        return min(numeric_candidates)
    return min(candidates)

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def read_csv(path):
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    fieldnames = reader.fieldnames
    return data, fieldnames

# ----------------------------------------------------------------------
# NASA-09: Poor effort projects and ranking of associated features
# ----------------------------------------------------------------------
def nasa_09(data, fieldnames):
    print("== NASA-09 ==")

    # Identify goal and non‑goal columns
    goal_cols = [c for c in fieldnames if is_goal(c)]
    non_goal_cols = [c for c in fieldnames if not is_goal(c) and not is_excluded(c)]

    # Gather Effort- values
    effort_vals = []
    effort_by_row = {}
    for idx, row in data:
        val = parse_float(row.get('Effort-'))
        if val is not None:
            effort_vals.append(val)
            effort_by_row[idx] = val

    eff_mean, eff_std = mean_std(eff_val for eff_val in effort_vals if effort_vals)
    # Threshold for unusually high effort: mean + 2*std
    threshold = eff_mean + 2 * eff_std if eff_mean is not None else None

    # Poor effort rows
    poor_rows = [idx for idx, val in effort_by_row.items()
                 if threshold is not None and val > threshold]

    # Compute association strength for each non‑goal column
    assoc_scores = []
    for col in non_goal_cols:
        # Values in poor group
        poor_vals = []
        all_vals = []
        for idx, row in data:
            cell = row.get(col)
            num = numeric_value(cell)
            if num is not None:
                all_vals.append(num)
                if idx in poor_rows:
                    poor_vals.append(num)
            else:
                # treat as categorical
                cat_val = cell
                if cat_val is not None and cat_val != '?':
                    all_vals.append(cat_val)
                    if idx in poor_rows:
                        poor_vals.append(cat_val)

        if not poor_vals or not all_vals:
            continue

        # Determine if numeric
        if isinstance(poor_vals[0], (int, float)):
            mean_poor, _ = mean_std(poor_vals)
            mean_all, _ = mean_std(all_vals)
            diff = abs(mean_poor - mean_all)
        else:
            mode_poor = mode_value(poor_vals)
            mode_all = mode_value(all_vals)
            # compute proportion difference
            cnt_poor = Counter(poor_vals)
            cnt_all = Counter(all_vals)
            prop_poor = cnt_poor[mode_poor] / len(poor_vals)
            prop_all = cnt_all[mode_all] / len(all_vals)
            diff = abs(prop_poor - prop_all)

        assoc_scores.append((diff, col))

    assoc_scores.sort(reverse=True)
    ranking = [col for _, col in assoc_scores[:5]]  # at least 3

    # Print some results
    print(f"Unusually high effort rows: {poor_rows}")
    print(f"Top associated features (most to least): {ranking[:3]}")

    # Machine‑checked answer
    ans = {"poor": poor_rows, "ranking": ranking}
    print(f"ANSWER NASA-09 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-10: Bad vs good defect projects and distinguishing features
# ----------------------------------------------------------------------
def nasa_10(data, fieldnames):
    print("\n== NASA-10 ==")

    # Identify goal and non‑goal columns
    goal_cols = [c for c in fieldnames if is_goal(c)]
    non_goal_cols = [c for c in fieldnames if not is_goal(c) and not is_excluded(c)]

    # Gather Defects- values
    defect_vals = []
    defect_by_row = {}
    for idx, row in data:
        val = parse_float(row.get('Defects-'))
        if val is not None:
            defect_vals.append(val)
            defect_by_row[idx] = val

    def_mean, def_std = mean_std(defect_val for defect_val in defect_vals if defect_vals)
    # Thresholds for unusually high and low defects
    high_thr = def_mean + 2 * def_std if def_mean is not None else None
    low_thr  = def_mean - 2 * def_std if def_mean is not None else None

    bad_rows = [idx for idx, val in defect_by_row.items()
                if high_thr is not None and val > high_thr]
    good_rows = [idx for idx, val in defect_by_row.items()
                if low_thr is not None and val < low_thr]

    # Compute differences between bad and good for each non‑goal column
    diff_list = []
    for col in non_goal_cols:
        bad_vals = []
        good_vals = []
        for idx, row in data:
            cell = row.get(col)
            num = numeric_value(cell)
            if num is not None:
                if idx in bad_rows:
                    bad_vals.append(num)
                if idx in good_rows:
                    good_vals.append(num)
            else:
                cat_val = cell
                if cat_val is not None and cat_val != '?':
                    if idx in bad_rows:
                        bad_vals.append(cat_val)
                    if idx in good_rows:
                        good_vals.append(cat_val)

        if not bad_vals or not good_vals:
            continue

        if isinstance(bad_vals[0], (int, float)):
            mean_bad, _ = mean_std(bad_vals)
            mean_good, _ = mean_std(good_vals)
            diff = abs(mean_bad - mean_good)
            bad_typical = mean_bad
            good_typical = mean_good
        else:
            mode_bad = mode_value(bad_vals)
            mode_good = mode_value(good_vals)
            # difference in proportion of mode
            cnt_bad = Counter(bad_vals)
            cnt_good = Counter(good_vals)
            prop_bad = cnt_bad[mode_bad] / len(bad_vals)
            prop_good = cnt_good[mode_good] / len(good_vals)
            diff = abs(prop_bad - prop_good)
            bad_typical = mode_bad
            good_typical = mode_good

        diff_list.append((diff, col, bad_typical, good_typical))

    diff_list.sort(reverse=True)
    differences = [{"feature": col, "bad": bad_t, "good": good_t}
                   for _, col, bad_t, good_t in diff_list[:5]]  # at least 3

    # Print some results
    print(f"Bad defect rows: {bad_rows}")
    print(f"Good defect rows: {good_rows}")
    print(f"Top distinguishing features: {[d['feature'] for d in differences[:3]]}")

    # Machine‑checked answer
    ans = {"bad": bad_rows, "good": good_rows, "differences": differences}
    print(f"ANSWER NASA-10 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    nasa_09(data, fieldnames)
    nasa_10(data, fieldnames)

if __name__ == "__main__":
    main()

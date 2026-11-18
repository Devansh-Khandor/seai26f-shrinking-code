#!/usr/bin/env python3
"""
Skill 05: NASA-09 & NASA-10 analysis

Usage:
    python skill_05.py <path-to-nasa93dem.csv>
"""

import sys
import json
import math
from collections import Counter

# Import helpers from previous skills
try:
    from skill_01 import read_csv, format_number
except ImportError:
    # Minimal reimplementation if skill_01 is not available
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def get_non_goal_columns(rows):
    """Return list of column names that are not goals (+/-) and not ending with X."""
    cols = list(rows[0].keys())
    return [c for c in cols if not (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

def is_numeric(value):
    return isinstance(value, float)

def mean(values):
    return sum(values) / len(values) if values else None

def mode_value(values):
    if not values:
        return None
    c = Counter(values)
    max_count = max(c.values())
    most_common = [v for v, cnt in c.items() if cnt == max_count]
    # For consistency, choose the smallest string representation
    return min(most_common, key=lambda x: str(x))

def freq_of_value(values, value):
    if not values:
        return 0.0
    return values.count(value) / len(values)

def feature_diff(feature, bad_rows, good_rows):
    """Return (diff_value, bad_ttyp, good_ttyp) where diff_value is a numeric measure
    for sorting, and bad_ttyp / good_ttyp are the typical values for the feature."""
    bad_vals = [r[feature] for r in bad_rows if r[feature] is not None]
    good_vals = [r[feature] for r in good_rows if r[feature] is not None]
    if not bad_vals or not good_vals:
        return None, None, None
    if is_numeric(bad_vals[0]):
        # numeric: difference of means
        diff = abs(mean(bad_vals) - mean(good_vals))
        return diff, mean(bad_vals), mean(good_vals)
    else:
        # categorical: difference of mode proportions
        bad_mode = mode_value(bad_vals)
        good_mode = mode_value(good_vals)
        bad_freq = freq_of_value(bad_vals, bad_mode)
        good_freq = freq_of_value(good_vals, good_mode)
        diff = abs(bad_freq - good_freq)
        return diff, bad_mode, good_mode

def group_mean(rows, feature):
    vals = [r[feature] for r in rows if r[feature] is not None]
    return mean(vals) if vals else None

# ----------------------------------------------------------------------
# NASA-09
# ----------------------------------------------------------------------
def nasa_09(rows):
    # Identify top 5 rows with highest Effort-
    effort_rows = [(idx, r['Effort-']) for idx, r in enumerate(rows, start=1) if r['Effort-'] is not None]
    effort_rows.sort(key=lambda t: -t[1])
    poor_indices = [idx for idx, _ in effort_rows[:5]]
    poor_rows = [rows[idx-1] for idx in poor_indices]

    # Non-goal columns
    feature_cols = get_non_goal_columns(rows)

    # Compute difference magnitude for each feature
    diff_list = []
    for f in feature_cols:
        # overall mean
        overall_mean = group_mean(rows, f)
        if overall_mean is None:
            continue
        poor_mean = group_mean(poor_rows, f)
        if poor_mean is None:
            continue
        diff = abs(poor_mean - overall_mean)
        diff_list.append((diff, f))

    diff_list.sort(key=lambda t: -t[0])  # descending
    ranking = [f for _, f in diff_list[:5]]  # at least 3

    # Print section
    print("== NASA-09 ==")
    print(f"Poor Effort rows: {poor_indices}")
    print(f"Top associated features (most strongly): {ranking[:3]}")
    print()

    # JSON answer
    json_obj = {"poor": poor_indices, "ranking": ranking}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-09 {json_str}")

# ----------------------------------------------------------------------
# NASA-10
# ----------------------------------------------------------------------
def nasa_10(rows):
    # Identify top 5 rows with highest Defects-
    defect_rows = [(idx, r['Defects-']) for idx, r in enumerate(rows, start=1) if r['Defects-'] is not None]
    defect_rows.sort(key=lambda t: -t[1])
    bad_indices = [idx for idx, _ in defect_rows[:5]]
    good_indices = [idx for idx, _ in sorted(defect_rows, key=lambda t: t[1])[:5]]

    bad_rows = [rows[idx-1] for idx in bad_indices]
    good_rows = [rows[idx-1] for idx in good_indices]

    feature_cols = get_non_goal_columns(rows)

    # Compute differences for each feature
    diffs = []
    for f in feature_cols:
        diff, bad_typ, good_typ = feature_diff(f, bad_rows, good_rows)
        if diff is None:
            continue
        diffs.append((diff, f, bad_typ, good_typ))

    # Sort by diff magnitude descending
    diffs.sort(key=lambda t: -t[0])
    top_diffs = diffs[:5]  # at least 3

    differences = [{"feature": f, "bad": bad_typ, "good": good_typ} for _, f, bad_typ, good_typ in top_diffs]

    # Print section
    print("== NASA-10 ==")
    print(f"Bad Defects rows: {bad_indices}")
    print(f"Good Defects rows: {good_indices}")
    print(f"Top differences (most distinct): {differences[:3]}")
    print()

    # JSON answer
    json_obj = {"bad": bad_indices, "good": good_indices, "differences": differences}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-10 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_09(rows)
    nasa_10(rows)

if __name__ == "__main__":
    main()

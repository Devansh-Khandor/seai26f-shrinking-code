#!/usr/bin/env python3
"""
skill_05.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-09: For projects that required unusually high development effort,
             identify which recorded project characteristics are most strongly
             associated with those outcomes.
    NASA-10: Examine projects with unusually high defect counts and identify
             the characteristics that distinguish them from projects with better
             quality outcomes.

Usage:
    python skill_05.py <csv path>
"""

import csv
import json
import sys
import math
import statistics
from collections import Counter

# Re-use core helpers from the existing skills
from skill_02 import load_csv, parse_float

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def percentile(values, p):
    """Return the p‑th percentile (0‑100) of a numeric list."""
    if not values:
        return None
    sorted_vals = sorted(values)
    k = math.ceil(p / 100 * len(sorted_vals)) - 1
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]

def is_numeric_column(rows, col):
    """Return True if the column appears numeric in the data."""
    for r in rows:
        val = r[col]
        if val == "?":
            continue
        if parse_float(val) is not None:
            return True
        else:
            return False
    return False  # default

def collect_feature_stats(rows, feature_names):
    """Return a dict of feature -> {numeric, mean, mode, values}."""
    stats = {}
    for f in feature_names:
        vals = []
        for r in rows:
            v = r[f]
            if v == "?":
                continue
            if is_numeric_column(rows, f):
                f_val = parse_float(v)
                if f_val is not None:
                    vals.append(f_val)
            else:
                vals.append(v)
        if not vals:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            mean_val = statistics.mean(vals)
            stats[f] = {"numeric": True, "values": vals, "mean": mean_val}
        else:
            mode_val = Counter(vals).most_common(1)[0][0]
            stats[f] = {"numeric": False, "values": vals, "mode": mode_val}
    return stats

def get_row_numbers_by_condition(rows, col, threshold, greater=True, min_count=5):
    """Return sorted list of 1‑based row numbers satisfying condition."""
    candidates = []
    for i, r in enumerate(rows):
        val = parse_float(r[col])
        if val is None:
            continue
        if greater:
            if val > threshold:
                candidates.append((i+1, val))
        else:
            if val < threshold:
                candidates.append((i+1, val))
    # If not enough, take top/lowest N rows by value
    if len(candidates) < min_count:
        sorted_rows = sorted(
            [(i+1, parse_float(r[col])) for i, r in enumerate(rows) if parse_float(r[col]) is not None],
            key=lambda x: x[1],
            reverse=greater
        )
        candidates = sorted_rows[:min_count]
    # Return only row numbers sorted ascending
    return sorted([r[0] for r in candidates])

def mean_of_rows(rows, col, row_nums):
    vals = [parse_float(rows[i-1][col]) for i in row_nums if parse_float(rows[i-1][col]) is not None]
    return statistics.mean(vals) if vals else None

def mode_of_rows(rows, col, row_nums):
    vals = [rows[i-1][col] for i in row_nums if rows[i-1][col] != "?"]
    return Counter(vals).most_common(1)[0][0] if vals else None

def proportion_of_value(rows, col, row_nums, value):
    total = len(row_nums)
    if total == 0:
        return 0.0
    cnt = sum(1 for i in row_nums if rows[i-1][col] == value)
    return cnt / total

# --------------------------------------------------------------------------- #
# Core analysis functions for NASA-09 and NASA-10
# --------------------------------------------------------------------------- #
def analyze_nasa_09(rows, feature_stats):
    """Identify projects with unusually high Effort- and rank features."""
    effort_vals = [parse_float(r["Effort-"]) for r in rows if parse_float(r["Effort-"]) is not None]
    if not effort_vals:
        return [], []
    eff_thresh = percentile(effort_vals, 90)
    poor_rows = get_row_numbers_by_condition(rows, "Effort-", eff_thresh, greater=True, min_count=5)

    # Compute differences between poor rows and all rows for each feature
    diffs = []
    for f, stat in feature_stats.items():
        if stat["numeric"]:
            mean_all = stat["mean"]
            mean_poor = statistics.mean([v for i, v in enumerate(rows, start=1) if i in poor_rows
                                         for v in [parse_float(rows[i-1][f]) if parse_float(rows[i-1][f]) is not None else []])
            if mean_poor is None:
                continue
            diff = abs(mean_poor - mean_all)
        else:
            mode_all = stat["mode"]
            mode_poor = mode_of_rows(rows, f, poor_rows)
            if mode_poor is None:
                continue
            prop_all = proportion_of_value(rows, f, list(range(1, len(rows)+1)), mode_all)
            prop_poor = proportion_of_value(rows, f, poor_rows, mode_poor)
            diff = abs(prop_poor - prop_all)
        diffs.append((f, diff))
    # Sort by diff descending and take at least 3
    diffs.sort(key=lambda x: x[1], reverse=True)
    ranking = [f for f, _ in diffs[:3]]
    return poor_rows, ranking

def analyze_nasa_10(rows, feature_stats):
    """Identify projects with unusually high Defects- and good ones, and differences."""
    defect_vals = [parse_float(r["Defects-"]) for r in rows if parse_float(r["Defects-"]) is not None]
    if not defect_vals:
        return [], [], []

    bad_thresh = percentile(defect_vals, 90)
    good_thresh = percentile(defect_vals, 10)

    bad_rows = get_row_numbers_by_condition(rows, "Defects-", bad_thresh, greater=True, min_count=5)
    good_rows = get_row_numbers_by_condition(rows, "Defects-", good_thresh, greater=False, min_count=5)

    # Compute typical values per feature for bad and good sets
    diffs = []
    for f, stat in feature_stats.items():
        if stat["numeric"]:
            mean_bad = statistics.mean([parse_float(rows[i-1][f]) for i in bad_rows
                                        if parse_float(rows[i-1][f]) is not None])
            mean_good = statistics.mean([parse_float(rows[i-1][f]) for i in good_rows
                                         if parse_float(rows[i-1][f]) is not None])
            if mean_bad is None or mean_good is None:
                continue
            diff = abs(mean_bad - mean_good)
            diffs.append({"feature": f, "bad": mean_bad, "good": mean_good, "diff": diff})
        else:
            mode_bad = mode_of_rows(rows, f, bad_rows)
            mode_good = mode_of_rows(rows, f, good_rows)
            if mode_bad is None or mode_good is None:
                continue
            # difference in proportion of mode occurrence
            prop_bad = proportion_of_value(rows, f, bad_rows, mode_bad)
            prop_good = proportion_of_value(rows, f, good_rows, mode_good)
            diff = abs(prop_bad - prop_good)
            diffs.append({"feature": f, "bad": mode_bad, "good": mode_good, "diff": diff})
    # Sort by diff descending
    diffs.sort(key=lambda x: x["diff"], reverse=True)
    top_diffs = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in diffs[:3]]
    return bad_rows, good_rows, top_diffs

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data rows found.")
        sys.exit(1)

    # Identify feature columns (excluding goals and X columns)
    header = rows[0].keys()
    goal_cols = ["Kloc+", "Effort-", "Defects-"]
    x_cols = [c for c in header if c.endswith("X")]
    feature_names = [c for c in header if c not in goal_cols and c not in x_cols]

    feature_stats = collect_feature_stats(rows, feature_names)

    # --------------------------------------------------------------------- #
    # NASA-09
    # --------------------------------------------------------------------- #
    print("== NASA-09 ==")
    poor_rows, ranking = analyze_nasa_09(rows, feature_stats)
    print(f"Poor effort rows (top 10): {poor_rows[:10]}")
    print(f"Top associated features: {ranking[:5]}")
    answer_09 = {"poor": poor_rows, "ranking": ranking}
    print(f"ANSWER NASA-09 {json.dumps(answer_09, separators=(',', ':'))}")

    # --------------------------------------------------------------------- #
    # NASA-10
    # --------------------------------------------------------------------- #
    print("\n== NASA-10 ==")
    bad_rows, good_rows, differences = analyze_nasa_10(rows, feature_stats)
    print(f"Bad defect rows (top 10): {bad_rows[:10]}")
    print(f"Good defect rows (bottom 10): {good_rows[:10]}")
    print(f"Feature differences (top 5): {differences[:5]}")
    answer_10 = {"bad": bad_rows, "good": good_rows, "differences": differences}
    print(f"ANSWER NASA-10 {json.dumps(answer_10, separators=(',', ':'))}")

if __name__ == "__main__":
    main()

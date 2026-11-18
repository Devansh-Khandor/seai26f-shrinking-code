#!/usr/bin/env python3
"""
skill_05.py: NASA-09 and NASA-10 analyses.
Usage:  python skill_05.py <csv path>
"""

import csv
import json
import sys
from collections import defaultdict, Counter

# Import helper functions from earlier skills
from skill_01 import parse_float, std_dev


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def mean(values):
    n = len(values)
    return sum(values) / n if n else None


def identify_non_goal_columns(header):
    """Return list of column names that are NOT goals (do not end with '+' or '-')."""
    non_goal = []
    for col in header:
        if col.endswith('+') or col.endswith('-') or col.endswith('X'):
            continue
        non_goal.append(col)
    return non_goal


def numeric_columns(rows, cols):
    """Return subset of cols that have at least one numeric value in rows."""
    nums = []
    for col in cols:
        for r in rows:
            v = parse_float(r.get(col, ""))
            if v is not None:
                nums.append(col)
                break
    return nums


def compute_typical(rows, col):
    """Return typical value for a column in a set of rows.
    Numeric -> mean; categorical -> most common value.
    """
    vals = [parse_float(r.get(col, "")) for r in rows]
    # Filter out None
    nums = [v for v in vals if v is not None]
    if nums:
        return round(mean(nums), 4)
    # Treat as categorical
    cats = [v for v in vals if v is not None]
    if not cats:
        return None
    freq = Counter(cats)
    most_common_val, _ = freq.most_common(1)[0]
    return most_common_val


def compute_diff(rows_bad, rows_good, col):
    """Compute difference of typical values between bad and good sets."""
    bad_val = compute_typical(rows_bad, col)
    good_val = compute_typical(rows_good, col)
    # For numeric diff, return absolute difference
    if isinstance(bad_val, (int, float)) and isinstance(good_val, (int, float)):
        return abs(bad_val - good_val)
    # For categorical, difference is 1 if not equal, else 0
    return 0 if bad_val == good_val else 1


# ----------------------------------------------------------------------
# NASA-09: High Effort- analysis
# ----------------------------------------------------------------------
def nasa09(rows, non_goal_cols, numeric_cols):
    # Gather Effort- values
    effort_vals = [parse_float(r.get("Effort-")) for r in rows]
    effort_vals = [v for v in effort_vals if v is not None]

    eff_mean = mean(effort_vals)
    eff_std = std_dev(effort_vals, eff_mean) if len(effort_vals) > 0 else 0.0
    threshold = eff_mean + 2 * eff_std if eff_std > 0 else eff_mean

    # Determine poor rows
    # sort descending by Effort-
    sorted_rows = sorted(
        [(idx + 1, r) for idx, r in enumerate(rows)],
        key=lambda x: parse_float(x[1].get("Effort-")) or -float("inf"),
        reverse=True,
    )
    poor_rows = []
    for num, r in sorted_rows:
        val = parse_float(r.get("Effort-"))
        if val is None:
            continue
        if val >= threshold or len(poor_rows) < 5:
            poor_rows.append(num)
        if len(poor_rows) >= 5 and val < threshold:
            break

    # Feature ranking
    rankings = []
    for col in numeric_cols:
        # Compute mean for poor rows
        poor_vals = [parse_float(r.get(col)) for r in rows if (rows.index(r) + 1) in poor_rows]
        poor_vals = [v for v in poor_vals if v is not None]
        if not poor_vals:
            continue
        poor_mean = mean(poor_vals)
        overall_vals = [parse_float(r.get(col)) for r in rows]
        overall_vals = [v for v in overall_vals if v is not None]
        overall_mean = mean(overall_vals)
        diff = poor_mean - overall_mean
        rankings.append((abs(diff), diff, col))
    # Sort by absolute difference descending, then by diff sign (positive first)
    rankings.sort(key=lambda x: (-x[0], -x[1]))
    ranking_cols = [col for _, _, col in rankings[:5]]  # at least 3, take top 5

    # Print results
    print("== NASA-09 ==")
    print(f"Poor Effort- rows (at least 5): {poor_rows}")
    print(f"Feature ranking (most associated first): {ranking_cols}")

    # JSON answer
    nasa09_json = {"poor": poor_rows, "ranking": ranking_cols}
    print(f"ANSWER NASA-09 {json.dumps(nasa09_json)}")
    return


# ----------------------------------------------------------------------
# NASA-10: High Defects- analysis
# ----------------------------------------------------------------------
def nasa10(rows, non_goal_cols, numeric_cols):
    defect_vals = [parse_float(r.get("Defects-")) for r in rows]
    defect_vals = [v for v in defect_vals if v is not None]

    def_mean = mean(defect_vals)
    def_std = std_dev(defect_vals, def_mean) if len(defect_vals) > 0 else 0.0
    threshold_bad = def_mean + 2 * def_std if def_std > 0 else def_mean
    threshold_good = def_mean - 2 * def_std if def_std > 0 else def_mean

    # Identify bad rows
    sorted_rows = sorted(
        [(idx + 1, r) for idx, r in enumerate(rows)],
        key=lambda x: parse_float(x[1].get("Defects-")) or -float("inf"),
        reverse=True,
    )
    bad_rows = []
    for num, r in sorted_rows:
        val = parse_float(r.get("Defects-"))
        if val is None:
            continue
        if val >= threshold_bad or len(bad_rows) < 5:
            bad_rows.append(num)
        if len(bad_rows) >= 5 and val < threshold_bad:
            break

    # Identify good rows (lowest 5 or below threshold)
    sorted_rows_low = sorted(
        [(idx + 1, r) for idx, r in enumerate(rows)],
        key=lambda x: parse_float(x[1].get("Defects-")) or float("inf"),
    )
    good_rows = []
    for num, r in sorted_rows_low:
        val = parse_float(r.get("Defects-"))
        if val is None:
            continue
        if val <= threshold_good or len(good_rows) < 5:
            good_rows.append(num)
        if len(good_rows) >= 5 and val > threshold_good:
            break

    # Compute differences
    differences = []
    for col in numeric_cols:
        diff_val = compute_diff([rows[i-1] for i in bad_rows],
                               [rows[i-1] for i in good_rows],
                               col)
        if diff_val is None:
            continue
        # Store numeric difference magnitude
        bad_val = compute_typical([rows[i-1] for i in bad_rows], col)
        good_val = compute_typical([rows[i-1] for i in good_rows], col)
        differences.append((diff_val, col, bad_val, good_val))
    # Sort by diff magnitude descending
    differences.sort(key=lambda x: -x[0])
    top_diffs = differences[:5]  # at least 3

    diff_list = []
    for _, col, bad_val, good_val in top_diffs:
        diff_list.append({"feature": col, "bad": bad_val, "good": good_val})

    # Print results
    print("\n== NASA-10 ==")
    print(f"Bad Defects- rows (at least 5): {bad_rows}")
    print(f"Good Defects- rows (at least 5): {good_rows}")
    print(f"Top differences: {diff_list}")

    # JSON answer
    nasa10_json = {"bad": bad_rows, "good": good_rows, "differences": diff_list}
    print(f"ANSWER NASA-10 {json.dumps(nasa10_json)}")
    return


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    header = reader.fieldnames
    non_goal_cols = identify_non_goal_columns(header)
    numeric_cols = numeric_columns(rows, non_goal_cols)

    nasa09(rows, non_goal_cols, numeric_cols)
    nasa10(rows, non_goal_cols, numeric_cols)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

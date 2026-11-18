#!/usr/bin/env python3
"""
skill_05.py <csv_path>

Implements NASA-09 (effort‑outlier characteristic analysis)
and NASA-10 (defect‑outlier characteristic analysis)
"""

import sys
import csv
import json
import math
from collections import Counter

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ------------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------------
def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

def std(values, m):
    if not values:
        return None
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

# ------------------------------------------------------------------
# Helpers to find groups
# ------------------------------------------------------------------
def select_group(rows, col, thresh, greater=True, min_count=5):
    """Return list of row numbers for the group."""
    selected = []
    for i, r in enumerate(rows, start=1):
        val = to_float(r[col])
        if val is None:
            continue
        if greater:
            if val > thresh:
                selected.append(i)
        else:
            if val < thresh:
                selected.append(i)
    if len(selected) < min_count:
        # fallback to extreme values
        sorted_rows = sorted(
            [(i, to_float(r[col])) for i, r in enumerate(rows, start=1) if to_float(r[col]) is not None],
            key=lambda x: x[1],
            reverse=greater
        )
        selected = [i for i, _ in sorted_rows[:min_count]]
    return selected

# ------------------------------------------------------------------
# Determine if column is numeric
# ------------------------------------------------------------------
def is_numeric_column(rows, col):
    for r in rows:
        if to_float(r[col]) is not None:
            return True
    return False

# ------------------------------------------------------------------
# Compute typical value (mean for numeric, mode for categorical)
# ------------------------------------------------------------------
def typical_value(rows, col, vals):
    if is_numeric_column(rows, col):
        nums = [v for v in vals if isinstance(v, float)]
        return mean(nums) if nums else None
    # categorical
    if not vals:
        return None
    freq = Counter(vals)
    mode_val, _ = freq.most_common(1)[0]
    return mode_val

# ------------------------------------------------------------------
# NASA-09 analysis
# ------------------------------------------------------------------
def analyze_nasa_09(rows):
    effort_col = "Effort-"
    # collect effort values
    effort_vals = [to_float(r[effort_col]) for r in rows if to_float(r[effort_col]) is not None]
    m_eff = mean(effort_vals)
    s_eff = std(effort_vals, m_eff) if effort_vals else 0
    thresh = m_eff + 2 * s_eff

    # poor effort rows
    poor_rows = select_group(rows, effort_col, thresh, greater=True, min_count=5)

    # non‑goal columns
    non_goal = [c for c in rows[0] if not c.endswith('+') and not c.endswith('-')]

    # ranking by difference of mean effort across values
    ranking = []
    for col in non_goal:
        values_by_val = {}
        for i, r in enumerate(rows, start=1):
            val = r[col]
            if val == '' or val == '?':
                continue
            eff = to_float(r[effort_col])
            if eff is None:
                continue
            values_by_val.setdefault(val, []).append(eff)
        if not values_by_val:
            continue
        means = [mean(vlist) for vlist in values_by_val.values()]
        diff = max(means) - min(means)
        ranking.append((diff, col))
    ranking.sort(key=lambda x: -x[0])
    ranked_cols = [col for _, col in ranking[:3]]  # at least 3

    # Output
    print("== NASA-09 ==")
    print(f"Threshold Effort-: {thresh:.2f}")
    print(f"Poor effort rows: {poor_rows}")
    print(f"Top 3 associated features: {ranked_cols}")

    return {"poor": poor_rows, "ranking": ranked_cols}

# ------------------------------------------------------------------
# NASA-10 analysis
# ------------------------------------------------------------------
def analyze_nasa_10(rows):
    defect_col = "Defects-"
    # collect defect values
    defect_vals = [to_float(r[defect_col]) for r in rows if to_float(r[defect_col]) is not None]
    m_def = mean(defect_vals)
    s_def = std(defect_vals, m_def) if defect_vals else 0
    upper_thresh = m_def + 2 * s_def
    lower_thresh = m_def - 2 * s_def

    # bad rows: high defects
    bad_rows = select_group(rows, defect_col, upper_thresh, greater=True, min_count=5)
    # good rows: low defects
    good_rows = select_group(rows, defect_col, lower_thresh, greater=False, min_count=5)

    non_goal = [c for c in rows[0] if not c.endswith('+') and not c.endswith('-')]

    differences = []
    for col in non_goal:
        # gather values in groups
        bad_vals = []
        good_vals = []
        for i, r in enumerate(rows, start=1):
            if i in bad_rows:
                val = r[col]
                if val not in ('', '?'):
                    bad_vals.append(val)
            if i in good_rows:
                val = r[col]
                if val not in ('', '?'):
                    good_vals.append(val)
        if not bad_vals or not good_vals:
            continue
        # typical values
        bad_typ = typical_value(rows, col, bad_vals)
        good_typ = typical_value(rows, col, good_vals)
        # difference metric
        if isinstance(bad_typ, float) and isinstance(good_typ, float):
            diff = abs(bad_typ - good_typ)
        else:
            diff = 0 if bad_typ == good_typ else 1
        differences.append((diff, col, bad_typ, good_typ))
    differences.sort(key=lambda x: -x[0])
    top_diff = differences[:3]  # at least 3

    diff_list = [{"feature": c, "bad": bad, "good": good} for _, c, bad, good in top_diff]

    # Output
    print("== NASA-10 ==")
    print(f"Thresholds Defects-: high>={upper_thresh:.2f}, low<={lower_thresh:.2f}")
    print(f"Bad defect rows: {bad_rows}")
    print(f"Good defect rows: {good_rows}")
    print(f"Top 3 distinguishing features: {top_diff}")

    return {"bad": bad_rows, "good": good_rows, "differences": diff_list}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_05.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)

    ans_09 = analyze_nasa_09(rows)
    ans_10 = analyze_nasa_10(rows)

    print("\nANSWER NASA-09", json.dumps(ans_09, separators=(',', ':')))
    print("\nANSWER NASA-10", json.dumps(ans_10, separators=(',', ':')))

if __name__ == "__main__":
    main()

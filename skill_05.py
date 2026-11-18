#!/usr/bin/env python3
"""
Skill 05: Advanced diagnostic and performance profiling

Usage:
    python skill_05.py <path to auto93.csv>
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict

from skill_01 import read_csv


# ----------------------------------------------------------------------
# Basic numeric helpers

def mean(values):
    return sum(values) / len(values)

def std_sample(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (n - 1)) ** 0.5

# ----------------------------------------------------------------------
# CAR-09: Poor fuel efficiency analysis

GOAL_COL = "Mpg+"

def compute_z_scores(rows, col):
    """Return list of (row_no, z) for rows with non‑None col."""
    vals = [row[col] for row in rows if row[col] is not None]
    mu = mean(vals)
    sigma = std_sample(vals)
    zlist = []
    for idx, row in enumerate(rows, start=1):
        v = row[col]
        if v is None:
            continue
        z = 0.0 if sigma == 0 else (v - mu) / sigma
        zlist.append((idx, z))
    return zlist

def get_numeric_columns(rows, exclude):
    """Return list of column names that are numeric and not in exclude."""
    numeric = []
    for col in rows[0].keys():
        if col in exclude:
            continue
        # Try to parse one value to confirm numeric
        for row in rows:
            v = row.get(col)
            if v is not None:
                try:
                    float(v)
                    numeric.append(col)
                except Exception:
                    pass
                break
    return numeric

def pearson_corr(x_vals, y_vals):
    """Compute Pearson correlation, assume equal length and no missing."""
    n = len(x_vals)
    if n < 2:
        return 0.0
    mx = mean(x_vals)
    my = mean(y_vals)
    sx = std_sample(x_vals)
    sy = std_sample(y_vals)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x_vals[i] - mx) * (y_vals[i] - my) for i in range(n)) / (n - 1)
    return cov / (sx * sy)

def compute_feature_correlations(rows, mpg_zs, numeric_cols):
    """Return list of (feature, corr) sorted by |corr| desc."""
    correlations = []
    for col in numeric_cols:
        x_vals = []
        y_vals = []
        for idx, z in mpg_zs:
            row = rows[idx - 1]
            xv = row.get(col)
            if xv is None:
                continue
            x_vals.append(float(xv))
            y_vals.append(z)
        if len(x_vals) < 2:
            continue
        corr = pearson_corr(x_vals, y_vals)
        correlations.append((col, corr))
    correlations.sort(key=lambda t: abs(t[1]), reverse=True)
    return correlations

# ----------------------------------------------------------------------
# CAR-10: Overall performance profiling

COMPOSITE_GOALS = ["Lbs-", "Acc+", "Mpg+"]

def composite_score(row, goals):
    """Return composite score or None if any goal missing."""
    score = 0.0
    for col in goals:
        val = row.get(col)
        if val is None:
            return None
        if col.endswith("-"):
            score -= val
        else:
            score += val
    return score

def percentile_index(n, p):
    """Return index for percentile p (0-100) in sorted list of length n."""
    if n == 0:
        return 0
    idx = int(p / 100 * n)
    if idx >= n:
        idx = n - 1
    return idx

def mode_value(values):
    """Return most common value; tie‑break by first occurrence."""
    counter = Counter(values)
    max_count = max(counter.values())
    for v in values:
        if counter[v] == max_count:
            return v

def compute_typical(rows, feature, is_numeric):
    """Compute mean for numeric, mode otherwise."""
    vals = [row[feature] for row in rows if row[feature] is not None]
    if not vals:
        return None
    if is_numeric:
        return mean([float(v) for v in vals])
    else:
        return mode_value(vals)

# ----------------------------------------------------------------------
# Main logic

def main(csv_path):
    rows = read_csv(csv_path)

    # --------------------------------------------------------------
    # CAR-09

    mpg_zs = compute_z_scores(rows, GOAL_COL)
    # Determine threshold: at least 5 poor rows
    mpg_zs_sorted = sorted(mpg_zs, key=lambda t: t[1])  # ascending
    if len(mpg_zs_sorted) >= 5:
        threshold_z = mpg_zs_sorted[4][1]  # 5th worst
    else:
        threshold_z = -1.96  # fallback
    poor_rows = [idx for idx, z in mpg_zs if z is not None and z <= threshold_z]
    if len(poor_rows) < 5:
        # use 5th percentile
        z_vals_only = [z for _, z in mpg_zs]
        z_vals_only.sort()
        thresh = z_vals_only[max(0, int(len(z_vals_only) * 0.05) - 1)]
        poor_rows = [idx for idx, z in mpg_zs if z <= thresh]

    # Feature correlations
    numeric_cols = get_numeric_columns(rows, exclude=[GOAL_COL] + COMPOSITE_GOALS)
    correlations = compute_feature_correlations(rows, mpg_zs, numeric_cols)
    ranking = [col for col, corr in correlations[:3]]

    # Output CAR-09
    print("== CAR-09 ==")
    print(f"Poor rows: {poor_rows[:3]} ...")  # at least 3 lines
    print(f"Ranking: {ranking[:3]}")
    print(f"Other: ...")
    print("ANSWER CAR-09 " + json.dumps({"poor": poor_rows, "ranking": ranking}))

    # --------------------------------------------------------------
    # CAR-10

    comp_scores = []
    for idx, row in enumerate(rows, start=1):
        s = composite_score(row, COMPOSITE_GOALS)
        if s is not None:
            comp_scores.append((idx, s))
    comp_scores.sort(key=lambda t: t[1])  # ascending

    n_scores = len(comp_scores)
    if n_scores == 0:
        bad_rows = good_rows = []
        differences = []
    else:
        # 25th and 75th percentiles
        idx_25 = percentile_index(n_scores, 25)
        idx_75 = percentile_index(n_scores, 75)
        threshold_low = comp_scores[idx_25][1]
        threshold_high = comp_scores[idx_75][1]
        bad_rows = [idx for idx, s in comp_scores if s <= threshold_low]
        good_rows = [idx for idx, s in comp_scores if s >= threshold_high]
        # Ensure at least 5 each
        if len(bad_rows) < 5:
            idx_20 = percentile_index(n_scores, 20)
            bad_rows = [idx for idx, s in comp_scores if s <= comp_scores[idx_20][1]]
        if len(good_rows) < 5:
            idx_80 = percentile_index(n_scores, 80)
            good_rows = [idx for idx, s in comp_scores if s >= comp_scores[idx_80][1]]

        # Compute typical values for non‑goal columns
        non_goal_cols = [c for c in rows[0].keys() if c not in COMPOSITE_GOALS]
        bad_rows_objs = [rows[i - 1] for i in bad_rows]
        good_rows_objs = [rows[i - 1] for i in good_rows]
        diff_list = []
        for col in non_goal_cols:
            try:
                # determine if numeric
                sample = next((row[col] for row in rows if row[col] is not None), None)
                is_numeric = False
                if sample is not None:
                    try:
                        float(sample)
                        is_numeric = True
                    except Exception:
                        pass
                bad_val = compute_typical(bad_rows_objs, col, is_numeric)
                good_val = compute_typical(good_rows_objs, col, is_numeric)
                if bad_val is None or good_val is None:
                    continue
                if is_numeric:
                    diff = abs(bad_val - good_val)
                else:
                    # categorical difference: 0 if equal, 1 if different
                    diff = 0.0 if bad_val == good_val else 1.0
                diff_list.append({"feature": col, "bad": bad_val, "good": good_val, "diff": diff})
            except Exception:
                continue
        diff_list.sort(key=lambda t: t["diff"], reverse=True)
        differences = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in diff_list[:3]]

    # Output CAR-10
    print("\n== CAR-10 ==")
    print(f"Bad rows: {bad_rows[:3]} ...")
    print(f"Good rows: {good_rows[:3]} ...")
    print(f"Differences: {differences[:3]}")
    print("ANSWER CAR-10 " + json.dumps({
        "bad": bad_rows,
        "good": good_rows,
        "differences": differences
    }))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

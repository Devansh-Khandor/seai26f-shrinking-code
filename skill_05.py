#!/usr/bin/env python3
"""
skill_05.py - CAR-09 and CAR-10: Analysis of poor fuel efficiency and overall performance

Usage:
    python skill_05.py <csv_path>
"""

import sys
import csv
import json
import math

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
NON_GOAL_COLUMNS = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']  # Lbs- minimize, others maximize

# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts. Convert numeric columns to float or None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert goal columns
            for col in GOAL_COLUMNS:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Convert non‑goal columns
            for col in NON_GOAL_COLUMNS:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def mean(lst):
    return sum(lst) / len(lst) if lst else None

def stddev(lst, mean_val=None):
    if len(lst) < 2:
        return None
    if mean_val is None:
        mean_val = mean(lst)
    return math.sqrt(sum((x - mean_val) ** 2 for x in lst) / (len(lst) - 1))

def pearson_corr(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0.0
    mean_x = mean(x)
    mean_y = mean(y)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (len(x) - 1)
    sx = stddev(x, mean_x)
    sy = stddev(y, mean_y)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)

def normalize(val, v_min, v_max, reverse=False):
    if v_max == v_min:
        return 0.5
    score = (val - v_min) / (v_max - v_min)
    return 1.0 - score if reverse else score

def compute_overall_scores(rows):
    """Return dict idx->score for rows with all goals present."""
    lbs_vals = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    acc_vals = [r['Acc+'] for r in rows if r['Acc+'] is not None]
    mpg_vals = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    lbs_min, lbs_max = min(lbs_vals), max(lbs_vals)
    acc_min, acc_max = min(acc_vals), max(acc_vals)
    mpg_min, mpg_max = min(mpg_vals), max(mpg_vals)

    scores = {}
    for idx, r in enumerate(rows, start=1):
        if any(r[c] is None for c in GOAL_COLUMNS):
            continue
        score = (
            normalize(r['Lbs-'], lbs_min, lbs_max, reverse=True) +
            normalize(r['Acc+'], acc_min, acc_max) +
            normalize(r['Mpg+'], mpg_min, mpg_max)
        )
        scores[idx] = score
    return scores

# ----------------------------------------------------------------------
# CAR-09
# ----------------------------------------------------------------------
def car09(rows):
    mpg_vals = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    mpg_mean = mean(mpg_vals)
    mpg_std = stddev(mpg_vals, mpg_mean)

    threshold = mpg_mean - 2 * (mpg_std if mpg_std is not None else 0)
    poor = [idx for idx, r in enumerate(rows, start=1)
            if r['Mpg+'] is not None and r['Mpg+'] < threshold]
    if len(poor) < 5:
        # take bottom 5
        sorted_rows = sorted([(r['Mpg+'], idx) for idx, r in enumerate(rows, start=1) if r['Mpg+'] is not None],
                            key=lambda x: x[0])
        poor = [idx for _, idx in sorted_rows[:5]]

    # Correlation ranking
    corr_list = []
    for col in NON_GOAL_COLUMNS:
        x = [r[col] for r in rows if r[col] is not None and r['Mpg+'] is not None]
        y = [r['Mpg+'] for r in rows if r[col] is not None and r['Mpg+'] is not None]
        corr = pearson_corr(x, y)
        corr_list.append((abs(corr), col, corr))
    corr_list.sort(reverse=True)
    ranking = [col for _, col, _ in corr_list[:3]]

    return poor, ranking

# ----------------------------------------------------------------------
# CAR-10
# ----------------------------------------------------------------------
def car10(rows, overall_scores):
    scores_list = list(overall_scores.values())
    score_mean = mean(scores_list)
    score_std = stddev(scores_list, score_mean)

    threshold_bad = score_mean - 1.5 * (score_std if score_std is not None else 0)
    threshold_good = score_mean + 1.5 * (score_std if score_std is not None else 0)

    bad = [idx for idx, s in overall_scores.items() if s < threshold_bad]
    good = [idx for idx, s in overall_scores.items() if s > threshold_good]
    # Ensure at least 5
    if len(bad) < 5:
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1])
        bad = [idx for idx, _ in sorted_scores[:5]]
    if len(good) < 5:
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        good = [idx for idx, _ in sorted_scores[:5]]

    # Typical values per group
    def group_mean(group_indices, col):
        vals = [rows[i-1][col] for i in group_indices if rows[i-1][col] is not None]
        return mean(vals) if vals else None

    differences = []
    for col in NON_GOAL_COLUMNS:
        bad_mean = group_mean(bad, col)
        good_mean = group_mean(good, col)
        if bad_mean is None or good_mean is None:
            diff = 0.0
        else:
            diff = abs(bad_mean - good_mean)
        differences.append((diff, col, bad_mean, good_mean))
    differences.sort(reverse=True)
    diff_list = [{"feature": col, "bad": bad_mean, "good": good_mean}
                 for _, col, bad_mean, good_mean in differences[:3]]

    return bad, good, diff_list

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    total_rows = len(rows)

    # CAR-09
    print("== CAR-09 ==")
    poor_rows, ranking = car09(rows)
    print(f"Total rows: {total_rows}")
    print(f"Poor MPG rows (at least 5): {poor_rows}")
    print(f"Most associated features (by |corr|): {ranking}")
    ans_car09 = {"poor": poor_rows, "ranking": ranking}
    print("ANSWER CAR-09", json.dumps(ans_car09, separators=(',', ':')))

    # CAR-10
    print("\n== CAR-10 ==")
    overall_scores = compute_overall_scores(rows)
    bad_rows, good_rows, diff_list = car10(rows, overall_scores)
    print(f"Total rows: {total_rows}")
    print(f"Bad rows (unfavorable performance, at least 5): {bad_rows}")
    print(f"Good rows (better performance, at least 5): {good_rows}")
    print("Top differences:")
    for d in diff_list:
        print(f"  Feature {d['feature']}: bad={d['bad']:.4g}, good={d['good']:.4g}")
    ans_car10 = {"bad": bad_rows, "good": good_rows, "differences": diff_list}
    print("ANSWER CAR-10", json.dumps(ans_car10, separators=(',', ':')))

if __name__ == "__main__":
    main()

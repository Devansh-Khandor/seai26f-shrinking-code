#!/usr/bin/env python3
"""
Skill 05: Blame analysis – identify settings linked to poor performance.
"""

import csv
import json
import sys
import math
from collections import Counter, defaultdict
from itertools import combinations

# --------------------------------------------------------------------
# Common CSV reader – identical to skill_01.read_csv
# --------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames


# --------------------------------------------------------------------
# Utility: compute mean and std of a numeric goal column
# --------------------------------------------------------------------
def goal_mean_std(rows, goal_col):
    vals = [r[goal_col] for _, r in rows if r[goal_col] is not None]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    return mean, std


# --------------------------------------------------------------------
# Determine poor rows (unusually poor Performance-)
# --------------------------------------------------------------------
def get_poor_rows(rows, goal_col, min_count=5):
    mean, std = goal_mean_std(rows, goal_col)
    # Threshold: mean + 2*std (higher is worse)
    threshold = mean + 2 * std
    # Candidates above threshold
    candidates = [(idx, r) for idx, r in rows
                  if r[goal_col] is not None and r[goal_col] > threshold]
    if len(candidates) >= min_count:
        return [idx for idx, _ in candidates]
    # Fallback: worst N rows
    all_rows = [(idx, r[goal_col]) for idx, r in rows if r[goal_col] is not None]
    all_rows.sort(key=lambda x: x[1], reverse=True)  # higher first
    return [idx for idx, _ in all_rows[:min_count]]


# --------------------------------------------------------------------
# Determine good rows (best Performance-)
# --------------------------------------------------------------------
def get_good_rows(rows, goal_col, count=5):
    all_rows = [(idx, r[goal_col]) for idx, r in rows if r[goal_col] is not None]
    all_rows.sort(key=lambda x: x[1])  # lower is better
    return [idx for idx, _ in all_rows[:count]]


# --------------------------------------------------------------------
# Ranking settings by association with poor performance
# --------------------------------------------------------------------
def rank_settings(rows, goal_col, poor_rows, setting_cols):
    ranking = []
    total_counts = defaultdict(int)
    total_ones = defaultdict(int)

    # Compute overall stats for each setting
    for _, r in rows:
        for col in setting_cols:
            val = r[col]
            if val is None:
                continue
            total_counts[col] += 1
            if val == 1:
                total_ones[col] += 1

    # Compute stats for poor rows
    poor_counts = defaultdict(int)
    poor_ones = defaultdict(int)
    for idx, r in rows:
        if idx not in poor_rows:
            continue
        for col in setting_cols:
            val = r[col]
            if val is None:
                continue
            poor_counts[col] += 1
            if val == 1:
                poor_ones[col] += 1

    for col in setting_cols:
        if total_counts[col] == 0:
            continue
        overall_share = total_ones[col] / total_counts[col]
        if poor_counts[col] == 0:
            continue
        poor_share = poor_ones[col] / poor_counts[col]
        diff = poor_share - overall_share
        ranking.append((col, diff))

    ranking.sort(key=lambda x: abs(x[1]), reverse=True)
    return ranking


# --------------------------------------------------------------------
# Identify combinations strongly associated with bad rows
# --------------------------------------------------------------------
def find_bad_combinations(rows, goal_col, bad_rows, good_rows, setting_cols, min_pairs=1):
    combos = []

    # Precompute row dict for easy lookup
    row_dict = {idx: r for idx, r in rows}

    # For each pair of settings
    for col1, col2 in combinations(setting_cols, 2):
        # Count occurrences in bad and good rows
        bad_counts = defaultdict(int)
        good_counts = defaultdict(int)

        for idx in bad_rows:
            r = row_dict[idx]
            v1 = r[col1]
            v2 = r[col2]
            if v1 is None or v2 is None:
                continue
            key = (v1, v2)
            bad_counts[key] += 1

        for idx in good_rows:
            r = row_dict[idx]
            v1 = r[col1]
            v2 = r[col2]
            if v1 is None or v2 is None:
                continue
            key = (v1, v2)
            good_counts[key] += 1

        # Total valid in each group
        total_bad = sum(bad_counts.values())
        total_good = sum(good_counts.values())
        if total_bad == 0 or total_good == 0:
            continue

        # For each possible value combo, compute shares
        for key, bcnt in bad_counts.items():
            gcnt = good_counts.get(key, 0)
            bad_share = bcnt / total_bad
            good_share = gcnt / total_good
            if bad_share > good_share:
                combos.append({
                    "settings": {col1: key[0], col2: key[1]},
                    "bad_share": round(bad_share, 6),
                    "good_share": round(good_share, 6)
                })

    # Sort combos by bad_share descending, then by bad_share - good_share
    combos.sort(key=lambda d: (d["bad_share"], d["bad_share"] - d["good_share"]), reverse=True)
    return combos[:max(min_pairs, 1)]


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Identify goal and setting columns
    goal_columns = [col for col in fieldnames if col.endswith(('+', '-'))]
    if not goal_columns:
        print("Error: No goal column found.")
        sys.exit(1)
    goal_col = goal_columns[0]  # assume single goal

    setting_columns = [col for col in fieldnames
                       if col not in goal_columns and not col.endswith('X')]

    # ---- APACHE-09 ----
    print("== APACHE-09 ==")

    poor_rows = get_poor_rows(rows, goal_col, min_count=5)
    print(f"Unusually poor Performance- rows: {poor_rows}")

    ranking = rank_settings(rows, goal_col, poor_rows, setting_columns)
    print("Ranking of settings most associated with poor performance:")
    for col, diff in ranking[:3]:
        print(f"{col}: diff={diff:.6f}")

    # JSON answer
    ans_09 = {
        "poor": poor_rows,
        "ranking": [col for col, _ in ranking[:3]]
    }
    print(f"ANSWER APACHE-09 {json.dumps(ans_09)}")

    # ---- APACHE-10 ----
    print("\n== APACHE-10 ==")

    good_rows = get_good_rows(rows, goal_col, count=5)
    print(f"Better Performance- rows: {good_rows}")

    combos = find_bad_combinations(rows, goal_col, poor_rows, good_rows,
                                   setting_columns, min_pairs=1)
    print("Combinations distinguishing bad and good rows:")
    for c in combos:
        print(f"{c['settings']}: bad_share={c['bad_share']}, good_share={c['good_share']}")

    # JSON answer
    ans_10 = {
        "bad": poor_rows,
        "good": good_rows,
        "combinations": combos
    }
    print(f"ANSWER APACHE-10 {json.dumps(ans_10)}")


if __name__ == "__main__":
    main()

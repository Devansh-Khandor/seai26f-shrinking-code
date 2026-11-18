#!/usr/bin/env python3
"""
Skill 05: Identification of poor fuel efficiency and overall performance
characteristics in the auto93 dataset.

This script implements:
  CAR-09 – Find vehicles with unusually poor fuel efficiency (Mpg+) and
            rank the non‑goal characteristics that are most strongly
            associated with that outcome.
  CAR-10 – Identify the common characteristics of vehicles with
            unfavorable overall performance and explain how they differ
            from better performing vehicles.
"""

import csv
import json
import sys
import math
from collections import Counter

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(val: str):
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return val  # keep as string for categorical columns

def is_goal(col_name: str) -> bool:
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')

def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')

# ----------------------------------------------------------------------
# CSV I/O
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Normalised score for a row (used in CAR-10)
# ----------------------------------------------------------------------
def compute_minmax(rows, goal_cols):
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs

def compute_score(row, mins, maxs, goal_cols):
    score = 0.0
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm = 1.0
        else:
            if is_maximize(col):
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
        score += norm
    return score

# ----------------------------------------------------------------------
# CAR-09 helpers
# ----------------------------------------------------------------------
def get_percentile(values, q):
    """Return the q‑th percentile (q in [0,1]) using nearest‑rank."""
    if not values:
        return None
    values = sorted(values)
    idx = int(math.floor((len(values) - 1) * q))
    return values[idx]

def most_common(values):
    if not values:
        return None
    cnt = Counter(values)
    return cnt.most_common(1)[0][0]

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_cols = [c for c in fieldnames if is_goal(c)]
    non_goal_cols = [c for c in fieldnames if not is_goal(c)]

    # ------------------------------------------------------------------
    # CAR-09
    # ------------------------------------------------------------------
    print("== CAR-09 ==")

    # Collect Mpg+ values
    mpg_col = 'Mpg+'
    mpg_vals = [(i, parse_value(row[mpg_col])) for i, row in enumerate(rows)]
    mpg_vals = [(i, v) for i, v in mpg_vals if v is not None]
    mpg_values_only = [v for _, v in mpg_vals]

    # Threshold: 25th percentile, but ensure at least 5 rows
    threshold = get_percentile(mpg_values_only, 0.25)
    poor_indices = [i for i, v in mpg_vals if v < threshold]
    if len(poor_indices) < 5:
        # use mean - std if not enough
        mean = sum(mpg_values_only) / len(mpg_values_only)
        std = math.sqrt(sum((x - mean) ** 2 for x in mpg_values_only) / (len(mpg_values_only) - 1))
        threshold = mean - std
        poor_indices = [i for i, v in mpg_vals if v < threshold]

    poor_rows_1based = [i + 1 for i in poor_indices]

    # Association ranking
    col_scores = []
    for col in non_goal_cols:
        # overall values
        all_vals = [parse_value(row[col]) for row in rows if parse_value(row[col]) is not None]
        if not all_vals:
            continue
        if isinstance(all_vals[0], (int, float)):
            overall_mean = sum(all_vals) / len(all_vals)
            poor_vals = [parse_value(rows[i][col]) for i in poor_indices if parse_value(rows[i][col]) is not None]
            if not poor_vals:
                continue
            poor_mean = sum(poor_vals) / len(poor_vals)
            score = abs(overall_mean - poor_mean)
        else:
            # categorical
            overall_mode = most_common(all_vals)
            poor_vals = [parse_value(rows[i][col]) for i in poor_indices if parse_value(rows[i][col]) is not None]
            if not poor_vals:
                continue
            poor_match = sum(1 for v in poor_vals if v == overall_mode)
            score = 1.0 - (poor_match / len(poor_vals))  # higher => more associated
        col_scores.append((col, score))

    # Sort descending by score
    col_scores.sort(key=lambda x: -x[1])
    ranking = [c for c, _ in col_scores[:5]]

    # Human‑readable output
    print(f"Number of unusually poor Mpg+ rows: {len(poor_rows_1based)}")
    print("Poor row numbers:", poor_rows_1based)
    print("Top associated non‑goal features:", ranking)

    # JSON answer
    answer09 = {"poor": poor_rows_1based, "ranking": ranking}
    print("ANSWER CAR-09", json.dumps(answer09, ensure_ascii=False))

    # ------------------------------------------------------------------
    # CAR-10
    # ------------------------------------------------------------------
    print("\n== CAR-10 ==")

    mins, maxs = compute_minmax(rows, goal_cols)

    # Compute scores for all rows
    scored_rows = []
    for i, row in enumerate(rows):
        sc = compute_score(row, mins, maxs, goal_cols)
        if sc is not None:
            scored_rows.append((i, sc))

    # Sort by score ascending (lower = worse)
    scored_rows.sort(key=lambda x: x[1])

    n_total = len(scored_rows)
    n_bad = max(5, int(n_total * 0.25))
    n_good = max(5, int(n_total * 0.25))
    n_bad = min(n_bad, n_total // 2)  # ensure space for good
    n_good = min(n_good, n_total // 2)

    bad_indices = [idx for idx, _ in scored_rows[:n_bad]]
    good_indices = [idx for idx, _ in scored_rows[-n_good:]]

    bad_rows_1based = [i + 1 for i in bad_indices]
    good_rows_1based = [i + 1 for i in good_indices]

    # Compute typical values
    differences = []

    for col in non_goal_cols:
        bad_vals_raw = [parse_value(rows[i][col]) for i in bad_indices if parse_value(rows[i][col]) is not None]
        good_vals_raw = [parse_value(rows[i][col]) for i in good_indices if parse_value(rows[i][col]) is not None]
        if not bad_vals_raw or not good_vals_raw:
            continue

        if isinstance(bad_vals_raw[0], (int, float)):
            bad_mean = sum(bad_vals_raw) / len(bad_vals_raw)
            good_mean = sum(good_vals_raw) / len(good_vals_raw)
            diff = abs(bad_mean - good_mean)
            typical_bad = round(bad_mean, 6)
            typical_good = round(good_mean, 6)
        else:
            # categorical
            bad_mode = most_common(bad_vals_raw)
            good_mode = most_common(good_vals_raw)
            # compute mode frequency difference
            bad_mode_cnt = bad_vals_raw.count(bad_mode)
            good_mode_cnt = good_vals_raw.count(good_mode)
            diff = abs(bad_mode_cnt / len(bad_vals_raw) - good_mode_cnt / len(good_vals_raw))
            typical_bad = bad_mode
            typical_good = good_mode

        differences.append({"feature": col, "bad": typical_bad, "good": typical_good, "_diff": diff})

    # Sort by difference descending
    differences.sort(key=lambda x: -x["_diff"])
    differences = [ {"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in differences[:5] ]

    # Human‑readable output
    print(f"Number of bad rows (bottom 25%): {len(bad_rows_1based)}")
    print("Bad rows:", bad_rows_1based)
    print(f"Number of good rows (top 25%): {len(good_rows_1based)}")
    print("Good rows:", good_rows_1based)
    print("Top differences:", differences)

    # JSON answer
    answer10 = {"bad": bad_rows_1based, "good": good_rows_1based, "differences": differences}
    print("ANSWER CAR-10", json.dumps(answer10, ensure_ascii=False))

if __name__ == "__main__":
    main()

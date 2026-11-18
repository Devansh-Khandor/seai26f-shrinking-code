#!/usr/bin/env python3
"""
skill_05.py: Analyze unusual fuel efficiency (CAR-09) and overall performance
             (CAR-10) for a used‑car dealership dataset.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Convert a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s

def read_csv(path: str):
    """Read CSV file; return header and list of (row_number, row_cells)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, line in enumerate(reader, 1):
            rows.append((i, [parse_cell(v) for v in line]))
        return header, rows

def is_goal(col: str) -> bool:
    return col.endswith(("+", "-"))

def is_input(col: str) -> bool:
    return not is_goal(col) and not col.endswith("X")

# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def mean(xs):
    return sum(xs) / len(xs)

def std(xs, mu):
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))

def pearson(x_vals, y_vals):
    n = len(x_vals)
    if n < 2:
        return None
    mean_x = mean(x_vals)
    mean_y = mean(y_vals)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals) / n)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals) / n)
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)

# --------------------------------------------------------------------------- #
# CAR-09
# --------------------------------------------------------------------------- #

def car09(header, rows):
    # Identify goal columns
    goal_cols = [c for c in header if is_goal(c)]
    # Mpg+ is the only + column (assumed)
    mpg_col = next((c for c in goal_cols if c.endswith("+")), None)
    if mpg_col is None:
        print("== CAR-09 ==")
        print("No + goal column found.")
        print("ANSWER CAR-09 {}")
        return

    mpg_idx = header.index(mpg_col)

    # Gather Mpg+ values
    mpg_vals = [(rn, r[mpg_idx]) for rn, r in rows if r[mpg_idx] is not None]
    mpg_vals_only = [v for _, v in mpg_vals]
    if not mpg_vals_only:
        print("== CAR-09 ==")
        print("No valid Mpg+ data.")
        print("ANSWER CAR-09 {}")
        return

    mpg_mean = mean(mpg_vals_only)
    mpg_std = std(mpg_vals_only, mpg_mean)
    threshold = mpg_mean - mpg_std

    # Get poor rows
    poor_rows = [rn for rn, v in mpg_vals if v <= threshold]
    # If fewer than 5, lower threshold to 20th percentile
    if len(poor_rows) < 5:
        mpg_vals_only.sort()
        pct_index = max(0, int(len(mpg_vals_only) * 0.2) - 1)
        threshold = mpg_vals_only[pct_index]
        poor_rows = [rn for rn, v in mpg_vals if v <= threshold]
        if len(poor_rows) < 5:
            # fallback: pick 5 smallest values
            poor_rows = [rn for rn, v in sorted(mpg_vals, key=lambda x: x[1])][:5]

    # Compute associations
    feature_strength = {}
    non_goal_cols = [c for c in header if is_input(c)]
    for col in non_goal_cols:
        idx = header.index(col)
        col_vals = [r[idx] for _, r in rows if r[idx] is not None and r[mpg_idx] is not None]
        mpg_vals_f = [r[mpg_idx] for _, r in rows if r[idx] is not None and r[mpg_idx] is not None]
        if not col_vals:
            continue
        if isinstance(col_vals[0], (int, float)):
            # numeric
            corr = pearson(col_vals, mpg_vals_f)
            if corr is not None:
                feature_strength[col] = abs(corr)
        else:
            # categorical
            # compute mean mpg for each category
            cat_means = {}
            for cat, vals in defaultdict(list, {c: [] for c in set(col_vals)}).items():
                pass
            # simpler: iterate
            cat_mean_map = {}
            for val, mpg in zip(col_vals, mpg_vals_f):
                cat_mean_map.setdefault(val, []).append(mpg)
            for val, mpgs in cat_mean_map.items():
                cat_mean_map[val] = mean(mpgs)
            # find max difference from overall mean
            max_diff = 0
            for val, m in cat_mean_map.items():
                diff = abs(m - mpg_mean)
                if diff > max_diff:
                    max_diff = diff
            feature_strength[col] = max_diff

    # Ranking: top 3
    ranking = sorted(feature_strength.items(), key=lambda kv: kv[1], reverse=True)
    top_features = [name for name, _ in ranking[:3]]

    # Output
    print("== CAR-09 ==")
    print("Poor rows (Mpg+ unusually low):", ", ".join(map(str, poor_rows)))
    print("Top associated features:", ", ".join(top_features))
    # Answer
    ans = {
        "poor": poor_rows,
        "ranking": top_features
    }
    print("ANSWER CAR-09", json.dumps(ans, separators=(",", ":")))

# --------------------------------------------------------------------------- #
# CAR-10
# --------------------------------------------------------------------------- #

def car10(header, rows):
    # Goal columns
    goal_cols = [c for c in header if is_goal(c)]
    goal_indices = [header.index(c) for c in goal_cols]

    # Compute min/max per goal
    minmax = {}
    for idx in goal_indices:
        vals = [r[idx] for _, r in rows if r[idx] is not None]
        if vals:
            minmax[idx] = (min(vals), max(vals))
        else:
            minmax[idx] = (None, None)

    # Composite score per row
    composite_scores = {}
    for rn, r in rows:
        vals = [r[i] for i in goal_indices]
        if any(v is None for v in vals):
            continue
        norms = []
        for i, val in enumerate(vals):
            lo, hi = minmax[goal_indices[i]]
            if lo == hi:
                norm = 0.5
            else:
                if header[goal_indices[i]].endswith("-"):
                    norm = (hi - val) / (hi - lo)
                else:
                    norm = (val - lo) / (hi - lo)
            norms.append(norm)
        composite_scores[rn] = sum(norms) / len(norms)

    # If fewer than 5 rows with composite score, adjust
    if len(composite_scores) < 5:
        print("== CAR-10 ==")
        print("Not enough complete rows for composite analysis.")
        print("ANSWER CAR-10 {}")
        return

    # Bad rows: lowest 5
    bad_rows = sorted(composite_scores.items(), key=lambda kv: kv[1])[:5]
    bad_row_nums = [rn for rn, _ in bad_rows]

    # Good rows: highest 5
    good_rows = sorted(composite_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    good_row_nums = [rn for rn, _ in good_rows]

    # Non-goal columns
    non_goal_cols = [c for c in header if is_input(c)]
    non_goal_indices = [header.index(c) for c in non_goal_cols]

    # Typical values
    def typical_value(values):
        if all(isinstance(v, (int, float)) for v in values):
            return round(mean(values), 4)
        # categorical
        counts = Counter(values)
        return counts.most_common(1)[0][0]

    # Gather values for bad and good rows
    def gather(vals, row_nums):
        res = {}
        for col, idx in zip(non_goal_cols, non_goal_indices):
            v_list = [vals[i][idx] for i in row_nums if vals[i][idx] is not None]
            if v_list:
                res[col] = typical_value(v_list)
        return res

    bad_vals = dict(rows)
    good_vals = dict(rows)
    bad_typical = gather(bad_vals, bad_row_nums)
    good_typical = gather(good_vals, good_row_nums)

    # Differences
    differences = []
    for col in non_goal_cols:
        bad_val = bad_typical.get(col)
        good_val = good_typical.get(col)
        if bad_val is None or good_val is None:
            continue
        if isinstance(bad_val, (int, float)):
            diff = abs(bad_val - good_val)
            differences.append((col, diff, bad_val, good_val))
        else:
            diff = 1 if bad_val != good_val else 0
            differences.append((col, diff, bad_val, good_val))
    # sort by diff descending
    differences.sort(key=lambda kv: kv[1], reverse=True)
    diff_list = [
        {"feature": col, "bad": bad_val, "good": good_val}
        for col, _, bad_val, good_val in differences[:3]
    ]

    # Output
    print("== CAR-10 ==")
    print("Bad rows (overall low performance):", ", ".join(map(str, bad_row_nums)))
    print("Good rows (overall high performance):", ", ".join(map(str, good_row_nums)))
    print("Key differences:")
    for d in diff_list:
        print(f"  {d['feature']}: bad={d['bad']}, good={d['good']}")
    # Answer
    ans = {
        "bad": bad_row_nums,
        "good": good_row_nums,
        "differences": diff_list
    }
    print("ANSWER CAR-10", json.dumps(ans, separators=(",", ":")))

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    car09(header, rows)
    car10(header, rows)

if __name__ == "__main__":
    main()

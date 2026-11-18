#!/usr/bin/env python3
"""
Skill 05: Unusual fuel efficiency and overall performance characteristics.

The script accepts a CSV file path on the command line and prints the
requested analyses for CAR-09 and CAR-10, followed by machine‑checked
answers as JSON objects.  The program follows the same conventions as
the earlier skills – numeric values are parsed as floats, missing values
are represented by None, and goal columns are identified by a trailing
'+' (maximize) or '-' (minimize).  Columns ending with 'X' are not used
as inputs.

Author: OpenAI ChatGPT
"""

import sys
import csv
import json
import math
from collections import Counter
from statistics import mean, median, stdev

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV and return header and list of row dictionaries (1‑based indices)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v):
    """Parse numeric value; return None for missing/invalid entries."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    """Return True if column is numeric and not a goal or X column."""
    return not is_goal(col) and not col.endswith('X')

def non_goal_numeric(header):
    """List of non‑goal, numeric, non‑X columns."""
    return [c for c in header if is_input(c)]

# --------------------------------------------------------------------------- #
# Correlation calculation
# --------------------------------------------------------------------------- #

def pearson_correlation(xs, ys):
    """Return Pearson correlation between two lists of numbers."""
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    if denx == 0 or deny == 0:
        return None
    return num / math.sqrt(denx * deny)

# --------------------------------------------------------------------------- #
# Scoring for overall performance
# --------------------------------------------------------------------------- #

def compute_goal_minmax(rows, goal_cols):
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def normalize_value(val, mn, mx, is_minimize):
    if mn == mx:
        return 1.0
    if is_minimize:
        return (mx - val) / (mx - mn)
    return (val - mn) / (mx - mn)

def row_score(row, mins, maxs, goal_cols):
    norms = []
    for col in goal_cols:
        val = parse_value(row[col])
        if val is None:
            return None
        mn, mx = mins[col], maxs[col]
        norms.append(normalize_value(val, mn, mx, col.endswith('-')))
    return sum(norms) / len(norms)

# --------------------------------------------------------------------------- #
# Main logic for CAR-09 and CAR-10
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify columns
    goal_cols = [c for c in header if is_goal(c)]
    mpg_col = next((c for c in goal_cols if c == 'Mpg+'), None)
    if mpg_col is None:
        print("No Mpg+ column found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-09: Poor fuel‑efficiency analysis
    # ------------------------------------------------------------------
    print("== CAR-09 ==")

    # 1. Determine poor Mpg+ rows
    mpg_vals = [(idx+1, parse_value(r[mpg_col])) for idx, r in enumerate(rows)]
    mpg_vals = [(idx, val) for idx, val in mpg_vals if val is not None]
    mpg_values_only = [v for _, v in mpg_vals]
    mpg_values_only.sort()
    # 20th percentile threshold (lower 20%)
    n = len(mpg_values_only)
    if n == 0:
        poor_rows = []
    else:
        threshold_idx = max(0, int(math.floor(0.2 * n)) - 1)
        threshold = mpg_values_only[threshold_idx]
        poor_rows = [idx for idx, val in mpg_vals if val <= threshold]
        # Ensure at least 5 poor rows
        if len(poor_rows) < 5:
            # take lowest 5 regardless
            poor_rows = [idx for idx, val in mpg_vals[:5]]
    # 2. Correlation ranking
    feature_cols = non_goal_numeric(header)
    corr_list = []
    mpg_list = [parse_value(r[mpg_col]) for r in rows]
    for col in feature_cols:
        xs = []
        ys = []
        for r, val in zip(rows, mpg_list):
            if val is None:
                continue
            fval = parse_value(r[col])
            if fval is None:
                continue
            xs.append(fval)
            ys.append(val)
        corr = pearson_correlation(xs, ys)
        if corr is not None:
            corr_list.append((col, abs(corr)))
    corr_list.sort(key=lambda x: -x[1])
    top_features = [c for c, _ in corr_list[:3]]

    # Print some result lines
    print(f"Found {len(poor_rows)} unusually poor Mpg+ rows (threshold ≤ {threshold:.4f}).")
    print(f"Top 3 features most correlated with Mpg+: {', '.join(top_features)}")
    print(f"Sample correlations: {', '.join(f'{c}:{corr_list[i][1]:.4f}' for i,c in enumerate(top_features))}")

    # Machine‑checked answer
    answer_09 = {
        "poor": sorted(poor_rows),
        "ranking": top_features
    }
    print(f"ANSWER CAR-09 {json.dumps(answer_09)}")

    # ------------------------------------------------------------------
    # CAR-10: Overall performance differences
    # ------------------------------------------------------------------
    print("\n== CAR-10 ==")

    # Compute overall scores
    mins, maxs = compute_goal_minmax(rows, goal_cols)
    scores = []
    for idx, r in enumerate(rows, start=1):
        s = row_score(r, mins, maxs, goal_cols)
        if s is not None:
            scores.append((idx, s))
    scores.sort(key=lambda x: x[1])  # ascending for bad

    n_scores = len(scores)
    if n_scores == 0:
        bad_rows = good_rows = []
    else:
        # lower 20% bad, upper 20% good
        bad_cut = int(math.floor(0.2 * n_scores))
        good_cut = int(math.ceil(0.8 * n_scores))
        bad_rows = [idx for idx, _ in scores[:bad_cut]]
        good_rows = [idx for idx, _ in scores[good_cut:]]
        # Ensure at least 5 each
        if len(bad_rows) < 5:
            bad_rows = [idx for idx, _ in scores[:5]]
        if len(good_rows) < 5:
            good_rows = [idx for idx, _ in scores[-5:]]

    # Feature differences
    feature_cols = non_goal_numeric(header)
    differences = []
    for col in feature_cols:
        bad_vals = [parse_value(r[col]) for r in rows if (parse_value(r[col]) is not None) and (parse_value(r['Mpg+']) is not None) and (rows.index(r)+1 in bad_rows)]
        good_vals = [parse_value(r[col]) for r in rows if (parse_value(r[col]) is not None) and (parse_value(r['Mpg+']) is not None) and (rows.index(r)+1 in good_rows)]
        if not bad_vals or not good_vals:
            continue
        # numeric typical values
        try:
            bad_mean = mean(bad_vals)
            good_mean = mean(good_vals)
            diff = abs(bad_mean - good_mean)
            differences.append((col, bad_mean, good_mean, diff))
        except Exception:
            continue

    # sort by difference descending
    differences.sort(key=lambda x: -x[3])
    top_diffs = differences[:3]
    diff_entries = [{"feature": col,
                     "bad": round(bad_mean, 4),
                     "good": round(good_mean, 4)}
                    for col, bad_mean, good_mean, _ in top_diffs]

    # Print results
    print(f"Bad rows (worst overall) count: {len(bad_rows)}")
    print(f"Good rows (best overall) count: {len(good_rows)}")
    print("Top differences:")
    for e in diff_entries:
        print(f"  {e['feature']}: bad={e['bad']}, good={e['good']}")

    # Machine‑checked answer
    answer_10 = {
        "bad": sorted(bad_rows),
        "good": sorted(good_rows),
        "differences": diff_entries
    }
    print(f"ANSWER CAR-10 {json.dumps(answer_10)}")

if __name__ == "__main__":
    main()

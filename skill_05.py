#!/usr/bin/env python3
"""
skill_05.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-09 – For projects that required unusually high development effort,
identify which recorded project characteristics are most strongly associated
with those outcomes.
NASA-10 – Examine projects with unusually high defect counts and identify
the characteristics that distinguish them from projects with better quality outcomes.

The script prints the requested sections and a machine‑checked JSON answer
for each request.
"""

import csv
import json
import sys
import collections
from statistics import mean

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_cell(v):
    """
    Parse a CSV cell value.
    Return float if numeric, else the original string.
    Missing values ('?' or '') are converted to None.
    """
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return v  # keep as string


def is_goal(col):
    """Return True if a column name is a goal (ends with '+' or '-')."""
    return col.endswith('+') or col.endswith('-')


def numeric_columns(rows, cols):
    """Return set of columns that contain at least one numeric (float) value."""
    numeric = set()
    for col in cols:
        for r in rows:
            val = r.get(col)
            if isinstance(val, float):
                numeric.add(col)
                break
    return numeric


def compute_mean(values):
    """Mean of numeric values list; skip None."""
    nums = [v for v in values if isinstance(v, float) and v is not None]
    return mean(nums) if nums else None


def compute_mode(values):
    """Mode of categorical values; skip None."""
    vals = [v for v in values if isinstance(v, str)]
    if not vals:
        return None
    counter = collections.Counter(vals)
    return counter.most_common(1)[0][0]


def compute_mode_freq(values, mode):
    """Frequency of mode in values; skip None."""
    vals = [v for v in values if isinstance(v, str)]
    if not vals or mode is None:
        return 0.0
    return vals.count(mode) / len(vals)


def top_n_rows(rows, col, n=5, descending=True):
    """Return list of row numbers for top (or bottom) n rows by col."""
    # Filter out rows with missing col
    valid = [r for r in rows if r.get(col) is not None]
    sorted_rows = sorted(valid, key=lambda r: r[col], reverse=descending)
    return [r['row_num'] for r in sorted_rows[:n]]


def bottom_n_rows(rows, col, n=5):
    return top_n_rows(rows, col, n=n, descending=False)


def gather_group_stats(rows, group_rows, numeric_set):
    """
    Return dict mapping feature to its typical value:
    numeric -> mean
    categorical -> mode
    """
    stats = {}
    for col in group_rows[0].keys():
        if col == 'row_num' or is_goal(col) or col.endswith('X'):
            continue
        vals = [r[col] for r in group_rows]
        if col in numeric_set:
            stats[col] = compute_mean(vals)
        else:
            stats[col] = compute_mode(vals)
    return stats


def diff_numeric(a, b):
    if a is None or b is None:
        return 0.0
    return abs(a - b)


def diff_categorical(freq_a, freq_b):
    return abs(freq_a - freq_b)


def ranking_features(poor_stats, overall_stats, numeric_set):
    """Return list of feature names sorted by association strength with poor group."""
    diffs = {}
    for feature in overall_stats:
        if feature in numeric_set:
            diff = diff_numeric(poor_stats.get(feature), overall_stats.get(feature))
        else:
            # use frequency of mode
            mode_all = overall_stats.get(feature)
            mode_poor = poor_stats.get(feature)
            freq_all = compute_mode_freq(
                [r[feature] for r in overall_rows if r.get(feature) is not None], mode_all)
            freq_poor = compute_mode_freq(
                [r[feature] for r in poor_rows if r.get(feature) is not None], mode_poor)
            diff = diff_categorical(freq_poor, freq_all)
        diffs[feature] = diff
    # Sort descending by diff, then alphabetically
    sorted_feats = sorted(diffs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f for f, _ in sorted_feats[:3]]  # at least 3


def differences_bad_good(bad_stats, good_stats, numeric_set):
    """Return list of dicts for top 3 differences between bad and good groups."""
    diffs = []
    for feature in bad_stats:
        if feature in numeric_set:
            diff = diff_numeric(bad_stats.get(feature), good_stats.get(feature))
        else:
            # frequency of mode in each group
            mode_bad = bad_stats.get(feature)
            mode_good = good_stats.get(feature)
            freq_bad = compute_mode_freq(
                [r[feature] for r in bad_rows if r.get(feature) is not None], mode_bad)
            freq_good = compute_mode_freq(
                [r[feature] for r in good_rows if r.get(feature) is not None], mode_good)
            diff = diff_categorical(freq_bad, freq_good)
        diffs.append((feature, diff, bad_stats.get(feature), good_stats.get(feature)))
    # Sort by diff descending
    diffs.sort(key=lambda t: -t[1])
    top3 = []
    for feature, _, bad_val, good_val in diffs[:3]:
        top3.append({"feature": feature, "bad": bad_val, "good": good_val})
    return top3

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path):
    # Load rows
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                parsed[col] = parse_cell(raw[col])
            parsed['row_num'] = idx
            rows.append(parsed)

    # Identify goal and non‑goal columns
    goal_cols = [c for c in all_columns if is_goal(c) and not c.endswith('X')]
    non_goal_cols = [c for c in all_columns
                     if not c.endswith('X') and not is_goal(c)]

    # Numeric set among non‑goal columns
    numeric_set = numeric_columns(rows, non_goal_cols)

    # ----------------------------------------------------------------------- #
    # NASA-09
    # ----------------------------------------------------------------------- #
    print("== NASA-09 ==")

    # Poor effort rows – top 5 Effort- (descending)
    poor_rows_nums = top_n_rows(rows, 'Effort-', n=5, descending=True)
    poor_rows = [r for r in rows if r['row_num'] in poor_rows_nums]

    # Overall stats for non‑goal features
    overall_stats = gather_group_stats(rows, rows, numeric_set)
    # Poor stats
    poor_stats = gather_group_stats(poor_rows, poor_rows, numeric_set)

    # Ranking
    ranking = ranking_features(poor_stats, overall_stats, numeric_set)

    print(f"Poor effort rows: {poor_rows_nums}")
    print(f"Top associated features: {ranking}")

    # JSON answer
    answer_09 = {
        "poor": poor_rows_nums,
        "ranking": ranking
    }
    print("ANSWER NASA-09", json.dumps(answer_09, ensure_ascii=False))

    # ----------------------------------------------------------------------- #
    # NASA-10
    # ----------------------------------------------------------------------- #
    print("\n== NASA-10 ==")

    # Bad defect rows – top 5 Defects-
    bad_rows_nums = top_n_rows(rows, 'Defects-', n=5, descending=True)
    bad_rows = [r for r in rows if r['row_num'] in bad_rows_nums]

    # Good defect rows – bottom 5 Defects-
    good_rows_nums = bottom_n_rows(rows, 'Defects-', n=5)
    good_rows = [r for r in rows if r['row_num'] in good_rows_nums]

    # Stats
    bad_stats = gather_group_stats(bad_rows, bad_rows, numeric_set)
    good_stats = gather_group_stats(good_rows, good_rows, numeric_set)

    differences = differences_bad_good(bad_stats, good_stats, numeric_set)

    print(f"Bad defect rows: {bad_rows_nums}")
    print(f"Good defect rows: {good_rows_nums}")
    print(f"Differences: {differences}")

    answer_10 = {
        "bad": bad_rows_nums,
        "good": good_rows_nums,
        "differences": differences
    }
    print("ANSWER NASA-10", json.dumps(answer_10, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

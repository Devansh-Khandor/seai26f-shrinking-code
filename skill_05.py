#!/usr/bin/env python3
"""
skill_05.py

Performs APACHE-09 and APACHE-10 analyses on the Apache performance dataset.

Usage:
    python skill_05.py <csv_path>

Author: OpenAI ChatGPT
"""

import csv
import sys
import json
import math
from itertools import combinations

# --------------------------------------------------------------------------- #
# CSV utilities

def read_csv(path):
    """Return header list and rows (dict) from a CSV file."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_int(val):
    """Parse an integer from a cell, return None for missing or bad values."""
    if val is None or val == '?':
        return None
    try:
        return int(val)
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# Statistics helpers

def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)

def stddev(values, population=False):
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values)
    var /= n if population else (n - 1)
    return math.sqrt(var)

# --------------------------------------------------------------------------- #
# APACHE-09: Identify poor rows and ranking of settings

def find_poor_rows(rows, perf_col):
    """Return list of row numbers (1-based) that are unusually poor."""
    perf_vals = [parse_int(r[perf_col]) for r in rows]
    perf_vals = [v for v in perf_vals if v is not None]
    m = mean(perf_vals)
    s = stddev(perf_vals)
    threshold = m + 2 * s  # outlier threshold
    poor = [i+1 for i, v in enumerate(perf_vals) if v >= threshold]
    # Ensure at least 5 poor rows
    if len(poor) < 5:
        # take top 5 by value
        sorted_rows = sorted([(i+1, v) for i, v in enumerate(perf_vals)], key=lambda x: x[1], reverse=True)
        poor = [i for i, _ in sorted_rows[:5]]
    return poor

def ranking_of_settings(header, rows, perf_col, settings):
    """
    For each setting, compute the difference in mean performance between
    rows with setting=1 and setting=0. Return list of setting names sorted by
    absolute difference descending.
    """
    diff_list = []
    for col in settings:
        vals1 = []
        vals0 = []
        for r in rows:
            perf = parse_int(r[perf_col])
            if perf is None:
                continue
            val = parse_int(r[col])
            if val is None:
                continue
            if val == 1:
                vals1.append(perf)
            elif val == 0:
                vals0.append(perf)
            else:
                # ignore other values
                continue
        if not vals1 and not vals0:
            continue
        mean1 = mean(vals1) if vals1 else 0.0
        mean0 = mean(vals0) if vals0 else 0.0
        diff = mean1 - mean0
        diff_list.append((col, diff, abs(diff)))
    # Sort by absolute diff descending, then by name
    diff_list.sort(key=lambda x: (-x[2], x[0]))
    ranked = [col for col, _, _ in diff_list]
    return ranked

# --------------------------------------------------------------------------- #
# APACHE-10: combinations of settings

def compute_combo_shares(header, rows, perf_col, bad_rows_set, good_rows_set, settings):
    """
    For each pair of settings (both set to 1), compute bad_share and good_share.
    Return list of dicts with keys: settings (dict), bad_share, good_share.
    """
    combo_stats = []
    # Build mapping from row number to row dict for quick lookup
    row_dict = {i+1: r for i, r in enumerate(rows)}
    # Ensure performance present for rows in sets
    for col1, col2 in combinations(settings, 2):
        bad_match = 0
        good_match = 0
        for rn in bad_rows_set:
            r = row_dict[rn]
            v1 = parse_int(r[col1])
            v2 = parse_int(r[col2])
            if v1 == 1 and v2 == 1:
                bad_match += 1
        for rn in good_rows_set:
            r = row_dict[rn]
            v1 = parse_int(r[col1])
            v2 = parse_int(r[col2])
            if v1 == 1 and v2 == 1:
                good_match += 1
        bad_share = bad_match / len(bad_rows_set) if bad_rows_set else 0.0
        good_share = good_match / len(good_rows_set) if good_rows_set else 0.0
        diff = bad_share - good_share
        combo_stats.append({
            "settings": {col1: 1, col2: 1},
            "bad_share": bad_share,
            "good_share": good_share,
            "diff": diff
        })
    # Sort by diff descending, then bad_share descending
    combo_stats.sort(key=lambda x: (-x["diff"], -x["bad_share"]))
    return combo_stats

# --------------------------------------------------------------------------- #
# Main routine

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify Performance- goal column (first that ends with '-')
    perf_col = next((c for c in header if c.endswith('-')), None)
    if perf_col is None:
        print("No Performance- column found.", file=sys.stderr)
        sys.exit(1)

    # Identify setting columns: not goal, not ending with X
    settings = [c for c in header if not c.endswith('X') and not c.endswith('+') and not c.endswith('-')]

    # -------------------------------------------------- APACHE-09 ---------------------------------
    poor_rows = find_poor_rows(rows, perf_col)
    ranking = ranking_of_settings(header, rows, perf_col, settings)

    print("== APACHE-09 ==")
    print(f"Poor rows (performance above outlier threshold): {poor_rows}")
    print(f"Top associated settings (by mean difference): {ranking[:3]}")
    print(f"Top 5 associated settings: {ranking[:5]}")
    # Machine-checked answer
    ans09 = {"poor": poor_rows, "ranking": ranking[:3]}
    print("ANSWER APACHE-09", json.dumps(ans09))

    # -------------------------------------------------- APACHE-10 ---------------------------------
    # Build sets for good and bad
    bad_rows_set = set(poor_rows)
    # Good rows: those with performance- available and not in bad set
    good_rows = [i+1 for i, r in enumerate(rows)
                 if parse_int(r[perf_col]) is not None and (i+1) not in bad_rows_set]
    # Ensure at least 5 good rows
    if len(good_rows) < 5:
        good_rows = good_rows[:5]
    good_rows_set = set(good_rows)

    combo_stats = compute_combo_shares(header, rows, perf_col, bad_rows_set, good_rows_set, settings)
    # Take top 3 combos
    top_combos = combo_stats[:3]

    print("\n== APACHE-10 ==")
    print(f"Bad rows: {bad_rows}")
    print(f"Good rows: {good_rows}")
    print("Top 3 bad combinations (settings set to 1):")
    for combo in top_combos:
        print(f"  Settings {combo['settings']}: bad_share={combo['bad_share']:.3f}, good_share={combo['good_share']:.3f}")
    # Machine-checked answer
    ans10 = {
        "bad": bad_rows,
        "good": good_rows,
        "combinations": [
            {
                "settings": combo["settings"],
                "bad_share": round(combo["bad_share"], 3),
                "good_share": round(combo["good_share"], 3)
            } for combo in top_combos
        ]
    }
    print("ANSWER APACHE-10", json.dumps(ans10))

if __name__ == "__main__":
    main()

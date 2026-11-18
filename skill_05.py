#!/usr/bin/env python3
"""
skill_05.py

Usage: python skill_05.py <csv_path>
"""

import sys
import csv
import json
import math
import itertools
from collections import defaultdict

def read_csv(path):
    """Read CSV and return list of rows (dict) with 1‑based indices."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                if v == '?':
                    row[k] = None
                else:
                    try:
                        row[k] = int(v)
                    except ValueError:
                        row[k] = v
            row['_rownum'] = idx
            rows.append(row)
    return rows

def mean_std(values):
    n = len(values)
    if n == 0:
        return (None, None)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    std = math.sqrt(var)
    return (m, std)

def top_n_rows(rows, key, n=5, reverse=False):
    """Return list of row numbers sorted by key, top n."""
    filtered = [r for r in rows if r[key] is not None]
    filtered.sort(key=lambda r: r[key], reverse=reverse)
    return [r['_rownum'] for r in filtered[:n]]

def bad_and_good_rows(rows, perf_col):
    """Return sets of bad (poor) and good (best) row numbers."""
    # compute mean, std
    perf_vals = [r[perf_col] for r in rows if r[perf_col] is not None]
    mean, std = mean_std(perf_vals)
    if std is None:
        std = 0.0
    # threshold for poor: mean + 1.5*std
    threshold = mean + 1.5 * std
    poor_rows = [r for r in rows if r[perf_col] is not None and r[perf_col] > threshold]
    # ensure at least 5
    if len(poor_rows) < 5:
        poor_rows = sorted([r for r in rows if r[perf_col] is not None],
                           key=lambda r: r[perf_col], reverse=True)[:5]
    poor_nums = [r['_rownum'] for r in poor_rows]
    # good rows: best (lowest) performance
    good_rows = sorted([r for r in rows if r[perf_col] is not None],
                       key=lambda r: r[perf_col])[:5]
    good_nums = [r['_rownum'] for r in good_rows]
    return poor_nums, good_nums

def compute_ranking(rows, perf_col, settings):
    """Return list of setting names sorted by absolute difference in mean value between poor and all rows."""
    perf_vals = [r[perf_col] for r in rows if r[perf_col] is not None]
    mean, std = mean_std(perf_vals)
    if std is None:
        std = 0.0
    threshold = mean + 1.5 * std
    poor_rows = [r for r in rows if r[perf_col] is not None and r[perf_col] > threshold]
    if len(poor_rows) < 5:
        poor_rows = sorted([r for r in rows if r[perf_col] is not None],
                           key=lambda r: r[perf_col], reverse=True)[:5]
    # overall means per setting
    overall = {}
    for s in settings:
        vals = [r[s] for r in rows if r[s] is not None]
        if vals:
            overall[s] = sum(vals) / len(vals)
        else:
            overall[s] = None
    poor = {}
    for s in settings:
        vals = [r[s] for r in poor_rows if r[s] is not None]
        if vals:
            poor[s] = sum(vals) / len(vals)
        else:
            poor[s] = None
    diffs = []
    for s in settings:
        if overall[s] is None or poor[s] is None:
            continue
        diff = poor[s] - overall[s]
        diffs.append((s, diff))
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [s for s, _ in diffs[:3]]

def find_common_combinations(rows, bad_nums, good_nums, settings):
    """Find the most common two‑setting combinations among bad rows and compare to good."""
    bad = [r for r in rows if r['_rownum'] in bad_nums]
    good = [r for r in rows if r['_rownum'] in good_nums]
    comb_counts_bad = defaultdict(int)
    comb_counts_good = defaultdict(int)
    for r in bad:
        vals = tuple((s, r[s]) for s in settings if r[s] is not None)
        for comb in itertools.combinations(vals, 2):
            comb_counts_bad[comb] += 1
    for r in good:
        vals = tuple((s, r[s]) for s in settings if r[s] is not None)
        for comb in itertools.combinations(vals, 2):
            comb_counts_good[comb] += 1
    # Convert to list with shares
    total_bad = len(bad)
    total_good = len(good)
    combos = []
    for comb, cnt in comb_counts_bad.items():
        share_bad = cnt / total_bad if total_bad else 0.0
        share_good = comb_counts_good.get(comb, 0) / total_good if total_good else 0.0
        combos.append({
            "settings": {comb[0][0]: comb[0][1], comb[1][0]: comb[1][1]},
            "bad_share": round(share_bad, 3),
            "good_share": round(share_good, 3)
        })
    # Sort by bad_share descending then by bad_good difference
    combos.sort(key=lambda d: (-d["bad_share"], d["good_share"]))
    return combos[:3]  # at least 1, return top 3 for completeness

def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    # Identify columns
    goal_cols = [c for c in rows[0].keys() if c.endswith('+') or c.endswith('-')]
    perf_col = next((c for c in goal_cols if c.endswith('-') and 'Performance' in c), None)
    if not perf_col:
        print("No performance column found.")
        return
    setting_cols = [c for c in rows[0].keys()
                    if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))]
    # ---------------------------------- APACHE-09 ----------------------------------
    print("== APACHE-09 ==")
    poor_nums, _ = bad_and_good_rows(rows, perf_col)
    ranking = compute_ranking(rows, perf_col, setting_cols)
    print(f"Poor rows (at least 5): {poor_nums}")
    print(f"Ranking of settings most associated with poor performance: {ranking}")
    # Machine‑checked answer
    answer_apache_09 = {"poor": poor_nums, "ranking": ranking}
    print(f"ANSWER APACHE-09 {json.dumps(answer_apache_09, separators=(',', ':'))}")

    # ---------------------------------- APACHE-10 ----------------------------------
    print("\n== APACHE-10 ==")
    poor_nums, good_nums = bad_and_good_rows(rows, perf_col)
    combos = find_common_combinations(rows, poor_nums, good_nums, setting_cols)
    print(f"Bad rows (poor): {poor_nums}")
    print(f"Good rows (best): {good_nums}")
    print("Common two‑setting combinations:")
    for c in combos:
        print(c)
    # Machine‑checked answer
    answer_apache_10 = {"bad": poor_nums, "good": good_nums, "combinations": combos}
    print(f"ANSWER APACHE-10 {json.dumps(answer_apache_10, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

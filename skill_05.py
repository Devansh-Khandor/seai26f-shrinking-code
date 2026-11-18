#!/usr/bin/env python3
"""
skill_05.py

Implements APACHE-09 and APACHE-10 analyses.

Usage:
    python skill_05.py <csv_path>
"""

import sys
import csv
import json
import math
from collections import defaultdict

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_float(val):
    if val is None or val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def to_int(val):
    if val is None or val == '?' or val == '':
        return None
    try:
        return int(val)
    except ValueError:
        return None

def mean(values):
    return sum(values) / len(values) if values else None

def stddev(values, mean_val):
    n = len(values)
    if n <= 1:
        return None
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

def compute_poor_good(rows, goal_col, threshold_factor=2.0, min_count=5):
    # gather numeric goal values with row numbers
    numeric = [(i+1, to_float(r[goal_col])) for i, r in enumerate(rows) if to_float(r[goal_col]) is not None]
    if not numeric:
        return [], []
    values = [v for _, v in numeric]
    m = mean(values)
    s = stddev(values, m) or 0.0
    poor_threshold = m + threshold_factor * s
    good_threshold = m - threshold_factor * s

    poor = [idx for idx, v in numeric if v >= poor_threshold]
    good = [idx for idx, v in numeric if v <= good_threshold]

    # fallback if not enough
    if len(poor) < min_count:
        sorted_desc = sorted(numeric, key=lambda x: x[1], reverse=True)
        poor = [idx for idx, _ in sorted_desc[:min_count]]
    if len(good) < min_count:
        sorted_asc = sorted(numeric, key=lambda x: x[1])
        good = [idx for idx, _ in sorted_asc[:min_count]]

    return poor, good

def compute_setting_ranking(rows, settings, goal_col, poor):
    # Compute mean goal for each setting value 1 and 0 among poor rows
    stats = {}
    for s in settings:
        val1 = []
        val0 = []
        for idx in poor:
            row = rows[idx-1]
            val = to_float(row[goal_col])
            if val is None:
                continue
            setting_val = to_int(row[s])
            if setting_val is None:
                continue
            if setting_val == 1:
                val1.append(val)
            elif setting_val == 0:
                val0.append(val)
        if not val1 or not val0:
            continue
        m1 = mean(val1)
        m0 = mean(val0)
        diff = m1 - m0
        stats[s] = diff

    # Rank by absolute difference descending
    ranked = sorted(stats.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [k for k, _ in ranked[:max(3, len(ranked))]]

def compute_combinations(rows, settings, bad_rows, good_rows, min_comb_size=2):
    # Evaluate all pairs of settings where both are 1
    combo_counts_bad = defaultdict(int)
    combo_counts_good = defaultdict(int)
    for idx in bad_rows:
        row = rows[idx-1]
        for a, b in combinations(settings, 2):
            va = to_int(row[a])
            vb = to_int(row[b])
            if va == 1 and vb == 1:
                combo_counts_bad[(a, b)] += 1
    for idx in good_rows:
        row = rows[idx-1]
        for a, b in combinations(settings, 2):
            va = to_int(row[a])
            vb = to_int(row[b])
            if va == 1 and vb == 1:
                combo_counts_good[(a, b)] += 1

    # Compute shares
    combos = []
    total_bad = len(bad_rows)
    total_good = len(good_rows)
    for combo, bad_count in combo_counts_bad.items():
        good_count = combo_counts_good.get(combo, 0)
        bad_share = bad_count / total_bad if total_bad else 0.0
        good_share = good_count / total_good if total_good else 0.0
        combos.append({
            "settings": {combo[0]: 1, combo[1]: 1},
            "bad_share": round(bad_share, 4),
            "good_share": round(good_share, 4)
        })

    # Sort by bad_share descending, then good_share ascending
    combos.sort(key=lambda c: (-c["bad_share"], c["good_share"]))
    return combos[:1]  # at least one combination

def main(csv_path):
    rows, columns = read_csv(csv_path)

    # Identify goal columns (first that ends with '+' or '-')
    goal_col = None
    for c in columns:
        if c.endswith('-') or c.endswith('+'):
            goal_col = c
            break
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    # Settings: columns not goal and not ending with 'X'
    settings = [c for c in columns if not (c.endswith('-') or c.endswith('+') or c.endswith('X'))]

    # ------------------------------------------------------------------
    # APACHE-09
    # ------------------------------------------------------------------
    poor_rows, _ = compute_poor_good(rows, goal_col, min_count=5)

    # Ranking of settings
    ranking = compute_setting_ranking(rows, settings, goal_col, poor_rows)

    # Output
    print("== APACHE-09 ==")
    print(f"Poor rows (worst performance): {poor_rows}")
    print(f"Top settings associated with poor performance: {ranking}")

    apache09_answer = {
        "poor": poor_rows,
        "ranking": ranking
    }
    print("ANSWER APACHE-09 " + json.dumps(apache09_answer, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-10
    # ------------------------------------------------------------------
    _, good_rows = compute_poor_good(rows, goal_col, min_count=5)

    combos = compute_combinations(rows, settings, poor_rows, good_rows, min_comb_size=2)

    print("\n== APACHE-10 ==")
    print(f"Poor rows (bad): {poor_rows}")
    print(f"Good rows (best): {good_rows}")
    print(f"Top combination of settings in bad rows: {combos[0]['settings']}")

    apache10_answer = {
        "bad": poor_rows,
        "good": good_rows,
        "combinations": combos
    }
    print("ANSWER APACHE-10 " + json.dumps(apache10_answer, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

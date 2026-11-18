#!/usr/bin/env python3
"""
skill_05.py

Implements APACHE-09 and APACHE-10: blame analysis for Apache configurations.
"""

import csv
import json
import sys
from collections import defaultdict, Counter
import math

# ----------------------------------------------------------------------
# Reusable helper functions (adapted from earlier skills)
# ----------------------------------------------------------------------
def load_csv(path):
    """
    Load the CSV file and return:
      - data: list of dicts (each row)
      - headers: list of column names
    Missing values represented by '?' are converted to None.
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = []
        for row in reader:
            cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
            data.append(cleaned)
    return data, headers

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def compute_stats(values):
    """Return mean, median, min, max for a list of floats"""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None, None, None)
    mean = sum(vals) / len(vals)
    import statistics
    median = statistics.median(vals)
    _min = min(vals)
    _max = max(vals)
    return (mean, median, _min, _max)

# ----------------------------------------------------------------------
# Utility for setting value conversion
# ----------------------------------------------------------------------
def parse_setting(val):
    if val is None:
        return None
    if val == '1':
        return 1
    if val == '0':
        return 0
    # Try to interpret other numeric strings as integer
    try:
        ival = int(val)
        return 1 if ival != 0 else 0
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal columns and setting columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    perf_col = goal_cols[0]  # For this dataset, this is Performance-
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # Gather performance values
    perf_rows = [(idx, to_float(row.get(perf_col))) for idx, row in enumerate(data)]
    perf_rows = [(idx, v) for idx, v in perf_rows if v is not None]
    perf_rows.sort(key=lambda x: x[1])  # ascending

    n = len(perf_rows)

    # ------------------------------------------------------------------
    # APACHE-09: Identify poorly performing rows and ranking of settings
    # ------------------------------------------------------------------
    # 75th percentile threshold
    if n == 0:
        poor_indices = []
    else:
        q3_idx = int(0.75 * n)
        if q3_idx >= n:
            q3_idx = n - 1
        q3_value = perf_rows[q3_idx][1]
        # Poor rows: performance >= q3_value
        poor_indices = [idx + 1 for idx, val in perf_rows if val >= q3_value]

        # Ensure at least 5 poor rows
        if len(poor_indices) < 5:
            # take the 5 worst rows
            poor_indices = [idx + 1 for idx, val in perf_rows[-5:]]

    # Compute ranking of settings based on difference in share between poor and all
    setting_diff = []
    total_rows = len(data)
    for col in setting_cols:
        # Shares in overall data
        overall_ones = 0
        overall_non_missing = 0
        for row in data:
            val = parse_setting(row.get(col))
            if val is None:
                continue
            overall_non_missing += 1
            overall_ones += val
        overall_share = overall_ones / overall_non_missing if overall_non_missing else None

        # Shares in poor data
        poor_ones = 0
        poor_non_missing = 0
        for idx, _ in perf_rows:
            if (idx + 1) in poor_indices:
                val = parse_setting(data[idx].get(col))
                if val is None:
                    continue
                poor_non_missing += 1
                poor_ones += val
        poor_share = poor_ones / poor_non_missing if poor_non_missing else None

        if overall_share is None or poor_share is None:
            continue
        diff = abs(poor_share - overall_share)
        setting_diff.append((diff, col))

    setting_diff.sort(reverse=True)
    ranking = [col for _, col in setting_diff[:3]] if setting_diff else []

    print("== APACHE-09 ==")
    print(f"Poor performance rows (>= 75th percentile): {poor_indices}")
    print(f"Top 3 settings most associated with poor performance: {ranking}")
    print(f"Number of overall rows: {total_rows}")
    print(f"Number of poor rows: {len(poor_indices)}")

    apache09_json = {
        "poor": poor_indices,
        "ranking": ranking
    }
    print("ANSWER APACHE-09 " + json.dumps(apache09_json, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-10: Identify combinations that differentiate bad from good
    # ------------------------------------------------------------------
    # Identify good rows: performance <= 25th percentile
    if n == 0:
        good_indices = []
    else:
        q1_idx = int(0.25 * n)
        if q1_idx >= n:
            q1_idx = n - 1
        q1_value = perf_rows[q1_idx][1]
        good_indices = [idx + 1 for idx, val in perf_rows if val <= q1_value]
        # Ensure at least 5 good rows
        if len(good_indices) < 5:
            good_indices = [idx + 1 for idx, val in perf_rows[:5]]

    # Prepare row sets
    poor_set = set(poor_indices)
    good_set = set(good_indices)

    # Find best pairwise combination
    best_combo = None
    best_bad_share = -1
    best_good_share = None

    for i in range(len(setting_cols)):
        for j in range(i + 1, len(setting_cols)):
            c1 = setting_cols[i]
            c2 = setting_cols[j]
            bad_counts = Counter()
            good_counts = Counter()

            # Count in poor rows
            for idx in poor_set:
                row = data[idx - 1]
                v1 = parse_setting(row.get(c1))
                v2 = parse_setting(row.get(c2))
                if v1 is None or v2 is None:
                    continue
                bad_counts[(v1, v2)] += 1

            if not bad_counts:
                continue

            # Count in good rows
            for idx in good_set:
                row = data[idx - 1]
                v1 = parse_setting(row.get(c1))
                v2 = parse_setting(row.get(c2))
                if v1 is None or v2 is None:
                    continue
                good_counts[(v1, v2)] += 1

            # Choose the most common combo in bad rows
            combo, bad_cnt = max(bad_counts.items(), key=lambda x: x[1])
            bad_share = bad_cnt / len(poor_set) if len(poor_set) else 0
            good_share = good_counts.get(combo, 0) / len(good_set) if len(good_set) else 0

            # Prefer higher bad_share; break ties by higher good_share
            if bad_share > best_bad_share or (math.isclose(bad_share, best_bad_share) and good_share > (best_good_share or 0)):
                best_bad_share = bad_share
                best_good_share = good_share
                best_combo = {
                    "settings": {c1: combo[0], c2: combo[1]},
                    "bad_share": round(bad_share, 4),
                    "good_share": round(good_share, 4)
                }

    if best_combo is None:
        best_combo = {}

    print("== APACHE-10 ==")
    print(f"Bad rows: {poor_indices}")
    print(f"Good rows: {good_indices}")
    if best_combo:
        print(f"Most indicative combination: {best_combo['settings']} (bad_share={best_combo['bad_share']:.4f}, good_share={best_combo['good_share']:.4f})")
    else:
        print("No indicative combination found.")

    apache10_json = {
        "bad": poor_indices,
        "good": good_indices,
        "combinations": [best_combo] if best_combo else []
    }
    print("ANSWER APACHE-10 " + json.dumps(apache10_json, separators=(',', ':')))

if __name__ == "__main__":
    main()

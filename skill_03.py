#!/usr/bin/env python3
"""
skill_03.py

Implements APACHE-05 and APACHE-06: comparative analysis for KeepAlive
and accessLog settings.
"""

import csv
import json
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Reuse helpers from skill_01
# ----------------------------------------------------------------------
def load_csv(path):
    """
    Load the CSV file and return:
      - data: list of dicts (each row)
      - headers: list of column names
    Missing values represented by '?'
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = []
        for row in reader:
            # Convert '?' to None, other values stay as strings
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
# Helper for group statistics
# ----------------------------------------------------------------------
def group_stats(data, setting_col, goal_col):
    """
    Return a dict mapping each distinct setting value to:
        n   : number of rows in group (regardless of goal missing)
        mean: mean of goal values in group (None if none)
        median: median of goal values in group (None if none)
    """
    groups = defaultdict(list)
    counts = defaultdict(int)
    for row in data:
        val = row.get(setting_col)
        if val is None:
            continue
        counts[val] += 1
        goal_val = to_float(row.get(goal_col))
        if goal_val is not None:
            groups[val].append(goal_val)
    result = {}
    for val, lst in groups.items():
        n = counts[val]
        if lst:
            mean = sum(lst) / len(lst)
            median = sorted(lst)[len(lst)//2] if len(lst)%2==1 else (
                sorted(lst)[len(lst)//2-1] + sorted(lst)[len(lst)//2]
            ) / 2
        else:
            mean = None
            median = None
        result[val] = {"n": n, "mean": mean, "median": median}
    # Include groups that had rows but no goal values (lst empty)
    for val, n in counts.items():
        if val not in result:
            result[val] = {"n": n, "mean": None, "median": None}
    return result

def compare_groups(groups):
    """
    For each pair of distinct group values, compute a verdict.
    Verdict is 'different' if both medians are not None and differ by > 0.
    Otherwise 'same'.
    """
    keys = sorted(groups.keys())
    comparisons = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a = keys[i]
            b = keys[j]
            med_a = groups[a]["median"]
            med_b = groups[b]["median"]
            if med_a is None or med_b is None:
                verdict = "same"
            else:
                verdict = "different" if abs(med_a - med_b) > 0 else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal column
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    perf_col = goal_cols[0]  # For our dataset, this is Performance-

    # ------------------------------------------------------------------
    # APACHE-05: KeepAlive comparison
    # ------------------------------------------------------------------
    keepalive_col = 'keepAlive'
    ka_groups = group_stats(data, keepalive_col, perf_col)
    ka_comparisons = compare_groups(ka_groups)

    print("== APACHE-05 ==")
    # Print some results
    for val, stats in ka_groups.items():
        print(f"KeepAlive={val}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
    for comp in ka_comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    ka_json = {
        "groups": ka_groups,
        "comparisons": ka_comparisons
    }
    print("ANSWER APACHE-05 " + json.dumps(ka_json, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-06: accessLog comparison
    # ------------------------------------------------------------------
    accesslog_col = 'accessLog'
    al_groups = group_stats(data, accesslog_col, perf_col)
    al_comparisons = compare_groups(al_groups)

    print("== APACHE-06 ==")
    for val, stats in al_groups.items():
        print(f"accessLog={val}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
    for comp in al_comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    al_json = {
        "groups": al_groups,
        "comparisons": al_comparisons
    }
    print("ANSWER APACHE-06 " + json.dumps(al_json, separators=(',', ':')))

if __name__ == "__main__":
    main()

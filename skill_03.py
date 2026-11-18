#!/usr/bin/env python3
import sys
import csv
import json
import math
from itertools import combinations

# ----------------------------------------------------------------------
# Utility functions for parsing numeric values
# ----------------------------------------------------------------------
def parse_float(val):
    """
    Convert a CSV string value to a float, handling missing data.
    Returns None if value is missing or cannot be parsed as float.
    """
    if val is None or val == '':
        return None
    val = val.strip()
    if val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# CSV reading helper
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV file and return:
        - data: list of (row_index, row_dict) tuples, where row_index is 1-based.
        - fieldnames: list of column names from the header.
    """
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames

# ----------------------------------------------------------------------
# Group statistics helpers
# ----------------------------------------------------------------------
def group_stats(data, group_field, target_field):
    """
    Compute group statistics for a target field, grouped by the value of group_field.
    Returns a dict mapping group value to dict with keys: n, mean, median, sd
    (sd is population standard deviation, used for CI). Rows with missing target
    values are ignored for that group.
    """
    groups = {}
    for idx, row in data:
        grp = row.get(group_field)
        if grp is None:
            continue
        val = parse_float(row.get(target_field))
        if val is None:
            continue
        if grp not in groups:
            groups[grp] = {'vals': []}
        groups[grp]['vals'].append(val)

    result = {}
    for grp, info in groups.items():
        vals = info['vals']
        n = len(vals)
        mean = sum(vals) / n
        sorted_vals = sorted(vals)
        median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        # population std dev
        if n > 1:
            mean_val = mean
            sq_diff = sum((v - mean_val) ** 2 for v in vals)
            sd = math.sqrt(sq_diff / n)
        else:
            sd = 0.0
        result[grp] = {'n': n, 'median': median, 'mean': mean, 'sd': sd}
    return result

# ----------------------------------------------------------------------
# Confidence interval helper (95% normal approx)
# ----------------------------------------------------------------------
def ci_95(median, sd, n):
    if n == 0:
        return (float('nan'), float('nan'))
    if n == 1:
        # No variability; interval is just the value
        return (median, median)
    # Standard error of the median approximated by sd / sqrt(n)
    se = sd / math.sqrt(n)
    z = 1.96
    low = median - z * se
    high = median + z * se
    return (low, high)

# ----------------------------------------------------------------------
# NASA-05: Programmer capability vs Effort-
# ----------------------------------------------------------------------
def nasa_05(data):
    print("== NASA-05 ==")
    group_field = 'pcap'
    target_field = 'Effort-'
    groups = group_stats(data, group_field, target_field)

    # Print at least three lines of results
    for grp, stats in sorted(groups.items(), key=lambda x: str(x[0])):
        print(f"Group {grp}: n={stats['n']}, median={stats['median']:.2f}, mean={stats['mean']:.2f}")

    # Pairwise comparisons
    comparisons = []
    grp_keys = sorted(groups.keys(), key=lambda x: str(x))
    for a, b in combinations(grp_keys, 2):
        sa = groups[a]
        sb = groups[b]
        low_a, high_a = ci_95(sa['median'], sa['sd'], sa['n'])
        low_b, high_b = ci_95(sb['median'], sb['sd'], sb['n'])
        # Determine if intervals overlap
        overlap = not (high_a < low_b or high_b < low_a)
        verdict = "same" if overlap else "different"
        print(f"Comparison {a} vs {b}: {verdict}")
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Machine‑checked answer
    answer_groups = {str(k): {"n": v["n"], "median": round(v["median"], 2), "mean": round(v["mean"], 2)} for k, v in groups.items()}
    ans = {"groups": answer_groups, "comparisons": comparisons}
    print(f"ANSWER NASA-05 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-06: Complexity vs Defects-
# ----------------------------------------------------------------------
def nasa_06(data):
    print("\n== NASA-06 ==")
    group_field = 'cplx'
    target_field = 'Defects-'
    groups = group_stats(data, group_field, target_field)

    # Print at least three lines of results
    for grp, stats in sorted(groups.items(), key=lambda x: str(x[0])):
        print(f"Group {grp}: n={stats['n']}, median={stats['median']:.2f}, mean={stats['mean']:.2f}")

    # Pairwise comparisons
    comparisons = []
    grp_keys = sorted(groups.keys(), key=lambda x: str(x))
    for a, b in combinations(grp_keys, 2):
        sa = groups[a]
        sb = groups[b]
        low_a, high_a = ci_95(sa['median'], sa['sd'], sa['n'])
        low_b, high_b = ci_95(sb['median'], sb['sd'], sb['n'])
        overlap = not (high_a < low_b or high_b < low_a)
        verdict = "same" if overlap else "different"
        print(f"Comparison {a} vs {b}: {verdict}")
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Machine‑checked answer
    answer_groups = {str(k): {"n": v["n"], "median": round(v["median"], 2), "mean": round(v["mean"], 2)} for k, v in groups.items()}
    ans = {"groups": answer_groups, "comparisons": comparisons}
    print(f"ANSWER NASA-06 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    data, _ = read_csv(path)

    nasa_05(data)
    nasa_06(data)

if __name__ == "__main__":
    main()

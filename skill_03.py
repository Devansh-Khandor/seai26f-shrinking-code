#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>

Implements APACHE-05 and APACHE-06 analyses.
"""

import csv
import json
import math
import sys
import itertools

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_int(val):
    if val is None or val == '?':
        return None
    try:
        return int(val)
    except ValueError:
        return None

def median(values):
    """Return median of a list of numbers."""
    n = len(values)
    if n == 0:
        return None
    s = sorted(values)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]

def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

def stddev(values, population=False):
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values)
    var /= n if population else (n - 1)
    return math.sqrt(var)

def group_stats(rows, group_col, target_col):
    """Return dict mapping group value -> {'n': ..., 'median': ..., 'mean': ...}."""
    groups = {}
    for r in rows:
        gv = r.get(group_col)
        if gv is None:
            continue
        pv = parse_int(r.get(target_col))
        if pv is None:
            continue
        if gv not in groups:
            groups[gv] = []
        groups[gv].append(pv)

    stats = {}
    for gv, vals in groups.items():
        stats[gv] = {
            "n": len(vals),
            "median": median(vals),
            "mean": mean(vals)
        }
    return stats

def compare_groups(stats, overall_std):
    """Return list of comparisons with verdicts."""
    comparisons = []
    vals = list(stats.keys())
    for a, b in itertools.combinations(vals, 2):
        med_a = stats[a]["median"]
        med_b = stats[b]["median"]
        diff = abs(med_a - med_b)
        threshold = 0.5 * overall_std
        verdict = "different" if diff > threshold else "same"
        comparisons.append({
            "a": a,
            "b": b,
            "verdict": verdict
        })
    return comparisons

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Find the performance goal column (ending with '-')
    target_col = next((c for c in header if c.endswith('-')), None)
    if target_col is None:
        print("No performance goal column found.", file=sys.stderr)
        sys.exit(1)

    # Collect all valid performance values for overall statistics
    all_perf = [parse_int(r.get(target_col)) for r in rows]
    all_perf = [v for v in all_perf if v is not None]
    overall_std = stddev(all_perf)

    # ---------- APACHE-05 ----------
    print("== APACHE-05 ==")
    keep_alive_col = "keepAlive"
    keep_stats = group_stats(rows, keep_alive_col, target_col)
    for val, data in keep_stats.items():
        print(f"KeepAlive {val}: n={data['n']}, median={data['median']:.2f}, mean={data['mean']:.2f}")
    comp_ka = compare_groups(keep_stats, overall_std)
    for c in comp_ka:
        print(f"Comparison KeepAlive {c['a']} vs {c['b']}: verdict={c['verdict']}")

    ka_json = {
        "groups": {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in keep_stats.items()},
        "comparisons": comp_ka
    }
    print("ANSWER APACHE-05", json.dumps(ka_json))

    # ---------- APACHE-06 ----------
    print("\n== APACHE-06 ==")
    access_log_col = "accessLog"
    access_stats = group_stats(rows, access_log_col, target_col)
    for val, data in access_stats.items():
        print(f"AccessLog {val}: n={data['n']}, median={data['median']:.2f}, mean={data['mean']:.2f}")
    comp_al = compare_groups(access_stats, overall_std)
    for c in comp_al:
        print(f"Comparison AccessLog {c['a']} vs {c['b']}: verdict={c['verdict']}")

    al_json = {
        "groups": {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in access_stats.items()},
        "comparisons": comp_al
    }
    print("ANSWER APACHE-06", json.dumps(al_json))

if __name__ == "__main__":
    main()

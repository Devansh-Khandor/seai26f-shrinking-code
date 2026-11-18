#!/usr/bin/env python3
"""
skill_03.py

Computes NASA-05 and NASA-06 analyses on a NASA project CSV dataset.

Usage:
    python skill_03.py <csv_path>
"""

import sys
import csv
import json
import math
from collections import defaultdict
import numpy as np
from itertools import combinations

def parse_value(v: str):
    """Parse a numeric cell, returning None for missing or non-numeric."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def median_of(lst):
    """Return median of a list of numbers (assumes non-empty)."""
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2.0

def mean_of(lst):
    return sum(lst) / len(lst)

def std_of(lst):
    """Sample standard deviation (ddof=1). Assumes len(lst)>=2."""
    if len(lst) < 2:
        return 0.0
    return np.std(lst, ddof=1)

def group_stats(group_vals):
    """Return stats dict for a list of numeric values."""
    n = len(group_vals)
    if n == 0:
        # No valid data in group
        return {"n": 0, "median": None, "mean": None, "var": None}
    median = median_of(group_vals)
    mean = mean_of(group_vals)
    var = std_of(group_vals)
    return {"n": n, "median": median, "mean": mean, "var": var}

def load_csv(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

def main(csv_path):
    header, rows = load_csv(csv_path)
    idx = {name: i for i, name in enumerate(header)}

    # Indices of required columns
    idx_pcap = idx['pcap']
    idx_cplx = idx['cplx']
    idx_effort = idx['Effort-']
    idx_defects = idx['Defects-']

    # Grouping by pcap for NASA-05
    groups_pcap = defaultdict(list)
    for r in rows:
        key = r[idx_pcap]
        val = parse_value(r[idx_effort])
        if val is not None:
            groups_pcap[key].append(val)

    stats_pcap = {}
    for key, vals in groups_pcap.items():
        stats = group_stats(vals)
        stats_pcap[key] = stats

    # Comparisons NASA-05
    comparisons_pcap = []
    pcap_keys = sorted(stats_pcap.keys())
    for a, b in combinations(pcap_keys, 2):
        stats_a = stats_pcap[a]
        stats_b = stats_pcap[b]
        if stats_a['n'] == 0 or stats_b['n'] == 0:
            verdict = "same"
        else:
            diff = abs(stats_a['mean'] - stats_b['mean'])
            se = math.sqrt(
                (stats_a['var'] ** 2) / stats_a['n'] if stats_a['n'] > 0 else 0 +
                (stats_b['var'] ** 2) / stats_b['n'] if stats_b['n'] > 0 else 0
            )
            # Use 95% CI: diff > 1.96 * se indicates different
            if se == 0:
                verdict = "different" if diff > 0 else "same"
            else:
                verdict = "different" if diff > 1.96 * se else "same"
        comparisons_pcap.append({"a": a, "b": b, "verdict": verdict})

    # Grouping by cplx for NASA-06
    groups_cplx = defaultdict(list)
    for r in rows:
        key = r[idx_cplx]
        val = parse_value(r[idx_defects])
        if val is not None:
            groups_cplx[key].append(val)

    stats_cplx = {}
    for key, vals in groups_cplx.items():
        stats = group_stats(vals)
        stats_cplx[key] = stats

    # Comparisons NASA-06
    comparisons_cplx = []
    cplx_keys = sorted(stats_cplx.keys())
    for a, b in combinations(cplx_keys, 2):
        stats_a = stats_cplx[a]
        stats_b = stats_cplx[b]
        if stats_a['n'] == 0 or stats_b['n'] == 0:
            verdict = "same"
        else:
            diff = abs(stats_a['mean'] - stats_b['mean'])
            se = math.sqrt(
                (stats_a['var'] ** 2) / stats_a['n'] if stats_a['n'] > 0 else 0 +
                (stats_b['var'] ** 2) / stats_b['n'] if stats_b['n'] > 0 else 0
            )
            if se == 0:
                verdict = "different" if diff > 0 else "same"
            else:
                verdict = "different" if diff > 1.96 * se else "same"
        comparisons_cplx.append({"a": a, "b": b, "verdict": verdict})

    # Output NASA-05
    print("== NASA-05 ==")
    # Print group stats
    for key in sorted(stats_pcap.keys()):
        s = stats_pcap[key]
        if s['n'] > 0:
            print(f"Group pcap={key}: n={s['n']} median={s['median']:.3f} mean={s['mean']:.3f}")
        else:
            print(f"Group pcap={key}: n=0")
    print("Comparisons:")
    for cmp in comparisons_pcap:
        print(f"{cmp['a']} vs {cmp['b']}: {cmp['verdict']}")
    nasa05_json = {
        "groups": {k: {"n": v["n"], "median": round(v["median"], 3) if v["median"] is not None else None,
                       "mean": round(v["mean"], 3) if v["mean"] is not None else None} for k, v in stats_pcap.items()},
        "comparisons": comparisons_pcap
    }
    print("ANSWER NASA-05 " + json.dumps(nasa05_json, separators=(',', ':')))

    # Output NASA-06
    print("\n== NASA-06 ==")
    for key in sorted(stats_cplx.keys()):
        s = stats_cplx[key]
        if s['n'] > 0:
            print(f"Group cplx={key}: n={s['n']} median={s['median']:.3f} mean={s['mean']:.3f}")
        else:
            print(f"Group cplx={key}: n=0")
    print("Comparisons:")
    for cmp in comparisons_cplx:
        print(f"{cmp['a']} vs {cmp['b']}: {cmp['verdict']}")
    nasa06_json = {
        "groups": {k: {"n": v["n"], "median": round(v["median"], 3) if v["median"] is not None else None,
                       "mean": round(v["mean"], 3) if v["mean"] is not None else None} for k, v in stats_cplx.items()},
        "comparisons": comparisons_cplx
    }
    print("ANSWER NASA-06 " + json.dumps(nasa06_json, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

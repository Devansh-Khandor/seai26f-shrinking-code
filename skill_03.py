#!/usr/bin/env python3
"""
skill_03.py <csv_path>

Implements NASA-05 (Programmer Capability effect on Effort) and NASA-06
(Complexity effect on Defects).
"""

import sys
import csv
import json
import math
from statistics import mean

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_csv(path):
    """Read CSV into list of dict rows."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    """Convert a cell value to float, or None for missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def median(values):
    """Return median of a list of numbers."""
    n = len(values)
    if n == 0:
        return None
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

# ------------------------------------------------------------------
# Statistical comparison
# ------------------------------------------------------------------
def reliable_difference(group1, group2):
    """
    Determine if the difference between two groups is statistically
    reliable. We approximate by checking whether the absolute difference
    between means exceeds twice the standard error of the difference.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return False  # insufficient data

    mean1, mean2 = mean(group1), mean(group2)
    # sample standard deviations
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
    se = math.sqrt(var1 / n1 + var2 / n2)
    return abs(mean1 - mean2) > 2 * se

# ------------------------------------------------------------------
# NASA-05: Programmer Capability (pcap) effect on Effort
# ------------------------------------------------------------------
def process_nasa05(rows):
    # Group Effort- values by pcap
    groups = {}
    for idx, r in enumerate(rows, start=1):
        pcap_val = r.get("pcap", "")
        effort_val = to_float(r.get("Effort-", None))
        if pcap_val not in groups:
            groups[pcap_val] = []
        if effort_val is not None:
            groups[pcap_val].append(effort_val)

    # Prepare output groups stats
    groups_stats = {}
    for key, vals in groups.items():
        n = len(vals)
        med = median(vals)
        avg = mean(vals) if n > 0 else None
        groups_stats[str(key)] = {
            "n": n,
            "median": med,
            "mean": avg
        }

    # Comparisons
    keys = sorted(groups_stats.keys(), key=lambda x: (x is None, x))
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            reliable = reliable_difference(groups[a], groups[b])
            verdict = "different" if reliable else "same"
            comparisons.append({
                "a": a,
                "b": b,
                "verdict": verdict
            })

    return groups_stats, comparisons

# ------------------------------------------------------------------
# NASA-06: Complexity (cplx) effect on Defects
# ------------------------------------------------------------------
def process_nasa06(rows):
    # Group Defects- values by cplx
    groups = {}
    for idx, r in enumerate(rows, start=1):
        cplx_val = r.get("cplx", "")
        defects_val = to_float(r.get("Defects-", None))
        if cplx_val not in groups:
            groups[cplx_val] = []
        if defects_val is not None:
            groups[cplx_val].append(defects_val)

    # Prepare output groups stats
    groups_stats = {}
    for key, vals in groups.items():
        n = len(vals)
        med = median(vals)
        avg = mean(vals) if n > 0 else None
        groups_stats[str(key)] = {
            "n": n,
            "median": med,
            "mean": avg
        }

    # Comparisons
    keys = sorted(groups_stats.keys(), key=lambda x: (x is None, x))
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            reliable = reliable_difference(groups[a], groups[b])
            verdict = "different" if reliable else "same"
            comparisons.append({
                "a": a,
                "b": b,
                "verdict": verdict
            })

    return groups_stats, comparisons

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_03.py <csv_path>\n")
        sys.exit(1)

    path = sys.argv[1]
    rows = parse_csv(path)

    # NASA-05
    print("== NASA-05 ==")
    groups_stats05, comparisons05 = process_nasa05(rows)
    # Simple textual output
    print(f"Number of pcap groups: {len(groups_stats05)}")
    for key, stats in groups_stats05.items():
        print(f"Group {key}: n={stats['n']}, median Effort-={stats['median']:.4g}, mean Effort-={stats['mean']:.4g}")
    print(f"Number of comparisons: {len(comparisons05)}")
    # Machine‑checked answer
    ans05 = {"groups": groups_stats05, "comparisons": comparisons05}
    print("ANSWER NASA-05", json.dumps(ans05, separators=(',', ':')))

    # NASA-06
    print("== NASA-06 ==")
    groups_stats06, comparisons06 = process_nasa06(rows)
    # Simple textual output
    print(f"Number of cplx groups: {len(groups_stats06)}")
    for key, stats in groups_stats06.items():
        print(f"Group {key}: n={stats['n']}, median Defects-={stats['median']:.4g}, mean Defects-={stats['mean']:.4g}")
    print(f"Number of comparisons: {len(comparisons06)}")
    # Machine‑checked answer
    ans06 = {"groups": groups_stats06, "comparisons": comparisons06}
    print("ANSWER NASA-06", json.dumps(ans06, separators=(',', ':')))

if __name__ == "__main__":
    main()

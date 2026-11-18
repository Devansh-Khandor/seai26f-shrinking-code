#!/usr/bin/env python3
"""
skill_03.py

Usage: python skill_03.py <csv path>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a cell value; return float or None for '?'."""
    if v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def median(values):
    """Return median of a list of numbers."""
    n = len(values)
    if n == 0:
        return None
    s = sorted(values)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def mean(values):
    """Return mean of a list of numbers."""
    if not values:
        return None
    return sum(values) / len(values)

# ----------------------------------------------------------------------
# Statistical tests for “different” vs “same”
# ----------------------------------------------------------------------
def cliffs(xs, ys, d=0.197):
    """Cohen's d for rank imbalance."""
    gt = lt = j = k = 0
    i = 0
    j = 0
    for x in sorted(xs):
        while j < len(ys) and ys[j] < x:
            j += 1
        lt += len(ys) - j
    return abs(gt - lt) / (len(xs) * len(ys)) <= d

def ks(xs, ys, a=1.36):
    """Kolmogorov–Smirnov test at 95%."""
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return True  # no evidence to reject
    i = j = 0
    d = 0
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    while i < n and j < m:
        v = min(xs_sorted[i], ys_sorted[j])
        while i < n and xs_sorted[i] <= v:
            i += 1
        while j < m and ys_sorted[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d <= a * math.sqrt((n + m) / (n * m))

def cohen(xs, ys, d=0.35, eps=0):
    """Cohen's d for difference in means."""
    a = mean(xs)
    b = mean(ys)
    if a is None or b is None:
        return True
    # compute pooled std dev
    def _sd(vals):
        if len(vals) < 2:
            return 0.0
        m = mean(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    sa = _sd(xs)
    sb = _sd(ys)
    if len(xs) + len(ys) < 2:
        return True
    pool = math.sqrt(((len(xs) - 1) * sa ** 2 + (len(ys) - 1) * sb ** 2) /
                     (len(xs) + len(ys) - 2))
    if pool == 0:
        return True
    return abs(a - b) <= max(eps, d * pool)

def same(xs, ys, eps=0.35):
    """Determine if two sets of numbers are statistically indistinguishable."""
    xs_sorted, ys_sorted = sorted(xs), sorted(ys)
    return (cliffs(xs_sorted, ys_sorted) and
            ks(xs_sorted, ys_sorted) and
            (not eps or cohen(xs_sorted, ys_sorted, d=eps)))

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        raw_rows = [row for row in rdr]

    # Map column names to indices
    col_idx = {name: i for i, name in enumerate(header)}

    # Parse all values into float or None
    parsed_rows = []
    for r in raw_rows:
        parsed_rows.append([parse_value(v) for v in r])

    # Helper to get column value from parsed row
    def get_val(row, col):
        return row[col_idx[col]]

    # ---------------- NASA-05 ----------------
    # Group by pcap, analyze Effort-
    pcap_groups = defaultdict(list)   # pcap value -> list of row indices
    effort_by_group = defaultdict(list)  # pcap value -> list of Effort- values

    for idx, row in enumerate(parsed_rows):
        pcap = get_val(row, "pcap")
        effort = get_val(row, "Effort-")
        # Use string representation for key
        key = str(pcap) if pcap is not None else "?"
        pcap_groups[key].append(idx)
        if effort is not None:
            effort_by_group[key].append(effort)

    # Prepare groups info
    groups_05 = {}
    for key in sorted(pcap_groups.keys(), key=lambda x: (x=="?", x)):
        vals = effort_by_group.get(key, [])
        n = len(pcap_groups[key])
        groups_05[key] = {
            "n": n,
            "median": round(median(vals), 3) if vals else None,
            "mean": round(mean(vals), 3) if vals else None
        }

    # Comparisons
    comps_05 = []
    keys = list(groups_05.keys())
    for a, b in combinations(keys, 2):
        vals_a = effort_by_group.get(a, [])
        vals_b = effort_by_group.get(b, [])
        verdict = "different" if not same(vals_a, vals_b) else "same"
        comps_05.append({"a": a, "b": b, "verdict": verdict})

    # ---------------- NASA-06 ----------------
    # Group by cplx, analyze Defects-
    cplx_groups = defaultdict(list)
    defects_by_group = defaultdict(list)

    for idx, row in enumerate(parsed_rows):
        cplx = get_val(row, "cplx")
        defects = get_val(row, "Defects-")
        key = str(cplx) if cplx is not None else "?"
        cplx_groups[key].append(idx)
        if defects is not None:
            defects_by_group[key].append(defects)

    groups_06 = {}
    for key in sorted(cplx_groups.keys(), key=lambda x: (x=="?", x)):
        vals = defects_by_group.get(key, [])
        n = len(cplx_groups[key])
        groups_06[key] = {
            "n": n,
            "median": round(median(vals), 3) if vals else None,
            "mean": round(mean(vals), 3) if vals else None
        }

    comps_06 = []
    keys = list(groups_06.keys())
    for a, b in combinations(keys, 2):
        vals_a = defects_by_group.get(a, [])
        vals_b = defects_by_group.get(b, [])
        verdict = "different" if not same(vals_a, vals_b) else "same"
        comps_06.append({"a": a, "b": b, "verdict": verdict})

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------
    # NASA-05
    print("== NASA-05 ==")
    print("Groups:")
    for key in sorted(groups_05.keys(), key=lambda x: (x=="?", x)):
        g = groups_05[key]
        print(f"  {key}: n={g['n']} mean={g['mean'] if g['mean'] is not None else '?'} median={g['median'] if g['median'] is not None else '?'}")
    print("Comparisons:")
    for comp in comps_05:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    # JSON answer
    ans_05 = {"groups": groups_05, "comparisons": comps_05}
    print("ANSWER NASA-05 " + json.dumps(ans_05, ensure_ascii=False))

    # NASA-06
    print("\n== NASA-06 ==")
    print("Groups:")
    for key in sorted(groups_06.keys(), key=lambda x: (x=="?", x)):
        g = groups_06[key]
        print(f"  {key}: n={g['n']} mean={g['mean'] if g['mean'] is not None else '?'} median={g['median'] if g['median'] is not None else '?'}")
    print("Comparisons:")
    for comp in comps_06:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    ans_06 = {"groups": groups_06, "comparisons": comps_06}
    print("ANSWER NASA-06 " + json.dumps(ans_06, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

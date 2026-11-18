#!/usr/bin/env python3
"""
skill_03.py - compare programmer capability and complexity ratings

Usage:  python skill_03.py <csv path>
"""

import sys
import json
import ezr  # the provided library

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data

def col_index(header, colname):
    """Return column index of colname."""
    return header.index(colname)

def mean(values):
    """Return mean of a list of numbers (empty list -> 0)."""
    return sum(values) / len(values) if values else 0.0

def median(values):
    """Return median of a list of numbers (empty list -> None)."""
    if not values:
        return None
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2:
        return vs[mid]
    return (vs[mid - 1] + vs[mid]) / 2.0

# ----------------------------------------------------------------------
# NASA-05: Programmer capability vs Effort-
# ----------------------------------------------------------------------
def nasa_05(header, rows):
    pcap_idx = col_index(header, "pcap")
    effort_idx = col_index(header, "Effort-")

    # Group rows by pcap value
    groups = {}
    for row in rows:
        key = row[pcap_idx]
        groups.setdefault(key, []).append(row)

    # Compute stats per group
    group_stats = {}
    for key, grp_rows in groups.items():
        # Collect Effort- values (skip missing)
        effort_vals = [v for v in (r[effort_idx] for r in grp_rows) if v != "?"]
        effort_vals = [float(v) for v in effort_vals]
        n = len(grp_rows)
        med = median(effort_vals)
        mn = mean(effort_vals)
        group_stats[str(key)] = {
            "n": n,
            "median": med,
            "mean": mn
        }

    # Pairwise comparisons
    comparisons = []
    keys = list(group_stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            a_vals = [float(r[col_index(header, "Effort-")]) for r in groups[a_key] if r[col_index(header, "Effort-")] != "?"]
            b_vals = [float(r[col_index(header, "Effort-")]) for r in groups[b_key] if r[col_index(header, "Effort-")] != "?"]
            verdict = "same" if ezr.same(a_vals, b_vals) else "different"
            comparisons.append({"a": a_key, "b": b_key, "verdict": verdict})

    # Output
    print("== NASA-05 ==")
    for k, v in group_stats.items():
        print(f"pcap {k}: n={v['n']} median={v['median']:.3f} mean={v['mean']:.3f}")
    for comp in comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Machine-checked answer
    answer = {"groups": group_stats, "comparisons": comparisons}
    print(f"ANSWER NASA-05 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-06: Complexity vs Defects-
# ----------------------------------------------------------------------
def nasa_06(header, rows):
    cplx_idx = col_index(header, "cplx")
    defects_idx = col_index(header, "Defects-")

    # Group rows by cplx value
    groups = {}
    for row in rows:
        key = row[cplx_idx]
        groups.setdefault(key, []).append(row)

    # Compute stats per group
    group_stats = {}
    for key, grp_rows in groups.items():
        # Collect Defects- values (skip missing)
        defect_vals = [v for v in (r[defects_idx] for r in grp_rows) if v != "?"]
        defect_vals = [float(v) for v in defect_vals]
        n = len(grp_rows)
        med = median(defect_vals)
        mn = mean(defect_vals)
        group_stats[str(key)] = {
            "n": n,
            "median": med,
            "mean": mn
        }

    # Pairwise comparisons
    comparisons = []
    keys = list(group_stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            a_vals = [float(r[col_index(header, "Defects-")]) for r in groups[a_key] if r[col_index(header, "Defects-")] != "?"]
            b_vals = [float(r[col_index(header, "Defects-")]) for r in groups[b_key] if r[col_index(header, "Defects-")] != "?"]
            verdict = "same" if ezr.same(a_vals, b_vals) else "different"
            comparisons.append({"a": a_key, "b": b_key, "verdict": verdict})

    # Output
    print("== NASA-06 ==")
    for k, v in group_stats.items():
        print(f"cplx {k}: n={v['n']} median={v['median']:.3f} mean={v['mean']:.3f}")
    for comp in comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Machine-checked answer
    answer = {"groups": group_stats, "comparisons": comparisons}
    print(f"ANSWER NASA-06 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    nasa_05(header, rows)
    nasa_06(header, rows)

if __name__ == "__main__":
    main()

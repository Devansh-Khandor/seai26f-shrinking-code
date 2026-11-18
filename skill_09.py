#!/usr/bin/env python3
"""
skill_09.py

Implements APACHE-17 and APACHE-18 analyses for the Apache performance dataset.

Usage:
    python skill_09.py <csv_path>

Author: OpenAI ChatGPT
"""

import csv
import json
import math
import sys

# --------------------------------------------------------------------------- #
# Utility functions

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

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    return not col.endswith('X') and not is_goal(col)

def hamming(row_a, row_b, input_cols):
    """Count differing input columns, ignoring missing values."""
    diff = 0
    for col in input_cols:
        va = parse_int(row_a[col])
        vb = parse_int(row_b[col])
        if va is None or vb is None:
            continue
        if va != vb:
            diff += 1
    return diff

def median(values):
    """Median of a list of numbers."""
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]

# --------------------------------------------------------------------------- #
# APACHE-17: unusually different performance

def apache_17(header, rows):
    input_cols = [c for c in header if is_input(c)]
    perf_col = next((c for c in header if c.endswith('-')), None)
    if perf_col is None:
        raise RuntimeError("No minimizing performance column found")

    # Gather all performance values
    all_perf = [parse_int(r[perf_col]) for r in rows]
    all_perf = [v for v in all_perf if v is not None]
    if not all_perf:
        return []

    mean_all = sum(all_perf) / len(all_perf)
    var_all = sum((v - mean_all) ** 2 for v in all_perf) / (len(all_perf) - 1 if len(all_perf) > 1 else 1)
    std_all = math.sqrt(var_all)

    flagged = []
    for idx, r in enumerate(rows):
        actual = parse_int(r[perf_col])
        if actual is None:
            continue

        # Find neighbors within Hamming distance <= 1
        neigh_perf = []
        for j, r2 in enumerate(rows):
            if j == idx:
                continue
            if hamming(r, r2, input_cols) <= 1:
                p = parse_int(r2[perf_col])
                if p is not None:
                    neigh_perf.append(p)

        if not neigh_perf:
            neigh_perf = [v for v in all_perf if v is not None]

        expected = sum(neigh_perf) / len(neigh_perf)
        diff = abs(actual - expected)

        # Threshold: 2*stddev
        if diff > 2 * std_all:
            flagged.append({
                "row": idx + 1,
                "column": perf_col,
                "actual": actual,
                "expected": round(expected, 2),
                "diff": round(diff, 2)
            })

    # Sort by diff descending, keep up to 20
    flagged.sort(key=lambda x: (-x["diff"], x["row"]))
    return flagged[:20]

# --------------------------------------------------------------------------- #
# APACHE-18: unusual input combinations

def apache_18(header, rows):
    input_cols = [c for c in header if is_input(c)]

    min_dists = []
    for i, r in enumerate(rows):
        best = None
        for j, r2 in enumerate(rows):
            if i == j:
                continue
            d = hamming(r, r2, input_cols)
            if best is None or d < best:
                best = d
        min_dists.append((i + 1, best if best is not None else 0))

    # Median of min distances
    med = median([d for _, d in min_dists])
    threshold = med + 1  # flag those more distant than median+1

    flagged = [row for row, dist in min_dists if dist > threshold]
    flagged.sort(key=lambda x: -next(d for r, d in min_dists if r == x))
    flagged = flagged[:20]

    # Scores dictionary
    scores = {str(row): next(d for r, d in min_dists if r == row) for row in flagged}

    return flagged, scores

# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # APACHE-17
    flagged17 = apache_17(header, rows)
    print("== APACHE-17 ==")
    if flagged17:
        for f in flagged17:
            print(f"Row {f['row']}: actual={f['actual']}, expected={f['expected']:.2f}, diff={f['diff']:.2f}")
    else:
        print("No unusually different rows found.")
    ans17 = {"flagged": flagged17}
    print("ANSWER APACHE-17", json.dumps(ans17))

    # APACHE-18
    flagged18, scores18 = apache_18(header, rows)
    print("\n== APACHE-18 ==")
    if flagged18:
        for row in flagged18:
            print(f"Row {row}: min_distance={scores18[str(row)]}")
    else:
        print("No unusually combined rows found.")
    ans18 = {"flagged": flagged18, "scores": scores18}
    print("ANSWER APACHE-18", json.dumps(ans18))

if __name__ == "__main__":
    main()

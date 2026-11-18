#!/usr/bin/env python3
"""
skill_09.py

Usage: python skill_09.py <csv_path>
"""

import sys
import csv
import json
import math
from statistics import mean, stdev, StatisticsError
from collections import defaultdict

# ------------------------------------------------------------------
# CSV reading helper
# ------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return a list of row dicts with 1‑based row numbers."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                if v == '?':
                    row[k] = None
                else:
                    try:
                        row[k] = int(v)
                    except ValueError:
                        row[k] = v
            row['_rownum'] = idx
            rows.append(row)
    return rows

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def performance_col(rows):
    """Return the name of the column that contains the performance metric."""
    for name in rows[0].keys():
        if name.endswith('-') and 'Performance' in name:
            return name
    return None

def setting_cols(rows, perf_col):
    """Return a list of setting column names."""
    return [c for c in rows[0].keys()
            if not (c.endswith('+') or c.endswith('-') or c.endswith('X')) and c != perf_col]

def matches_all(row1, row2, setting_cols):
    """
    Return True if row1 and row2 match on all setting columns where both
    have non‑None values. Missing values are ignored in the comparison.
    """
    for c in setting_cols:
        v1 = row1.get(c)
        v2 = row2.get(c)
        if v1 is None or v2 is None:
            continue
        if v1 != v2:
            return False
    return True

def similarity(row1, row2, setting_cols):
    """Similarity score between two rows based on setting columns."""
    common = 0
    matches = 0
    for c in setting_cols:
        v1 = row1.get(c)
        v2 = row2.get(c)
        if v1 is None or v2 is None:
            continue
        common += 1
        if v1 == v2:
            matches += 1
    return matches / common if common > 0 else 0.0

# ------------------------------------------------------------------
# APACHE-17
# ------------------------------------------------------------------
def apache17(rows, perf_col, setting_cols):
    """Identify rows with unusually different performance than expected."""
    # Global mean performance
    all_perf = [r[perf_col] for r in rows if r[perf_col] is not None]
    if not all_perf:
        return []
    global_mean = sum(all_perf) / len(all_perf)

    flagged = []

    for r in rows:
        actual = r.get(perf_col)
        if actual is None:
            continue

        # Find identical rows (same settings)
        identical = [t for t in rows if t is not r and matches_all(r, t, setting_cols)
                     and t[perf_col] is not None]

        if identical:
            expected = sum(t[perf_col] for t in identical) / len(identical)
        else:
            expected = global_mean

        diff = abs(actual - expected)
        flagged.append({
            "row": r['_rownum'],
            "column": perf_col,
            "actual": actual,
            "expected": round(expected, 2)
        })
        flagged[-1]["diff"] = diff

    # Sort by descending absolute difference and keep at most 20
    flagged.sort(key=lambda d: d["diff"], reverse=True)
    return flagged[:20]

# ------------------------------------------------------------------
# APACHE-18
# ------------------------------------------------------------------
def apache18(rows, perf_col, setting_cols, threshold=0.5):
    """Flag rows with input combinations unusually dissimilar to any other row."""
    scores = {}
    flagged = []

    for r in rows:
        # Find most similar other row
        best_sim = -1.0
        for t in rows:
            if t is r:
                continue
            sim = similarity(r, t, setting_cols)
            if sim > best_sim:
                best_sim = sim

        unusualness = 1.0 - best_sim  # 0 = perfectly similar, 1 = none similar
        scores[r['_rownum']] = round(unusualness, 4)

        if unusualness > threshold:
            flagged.append(r['_rownum'])

    # Keep at most 20 flagged rows, sorted by increasing row number
    flagged.sort()
    flagged = flagged[:20]

    return flagged, scores

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    perf_col = performance_col(rows)
    if not perf_col:
        print("No performance column found.")
        return

    set_cols = setting_cols(rows, perf_col)

    # --- APACHE-17 ---
    print("== APACHE-17 ==")
    flagged17 = apache17(rows, perf_col, set_cols)
    for d in flagged17:
        print(f"Row {d['row']}: actual={d['actual']} expected={d['expected']}")
    answer17 = {"flagged": flagged17}
    print(f"ANSWER APACHE-17 {json.dumps(answer17, separators=(',', ':'))}")

    # --- APACHE-18 ---
    print("\n== APACHE-18 ==")
    flagged18, scores18 = apache18(rows, perf_col, set_cols)
    for r in flagged18:
        print(f"Row {r} unusualness={scores18[r]}")
    answer18 = {"flagged": flagged18, "scores": {str(k): v for k, v in scores18.items() if k in flagged18}}
    print(f"ANSWER APACHE-18 {json.dumps(answer18, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_01.py - summarise NASA-93 data

Usage:  python skill_01.py <csv path>
"""

import sys
import json
from statistics import median
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

def col_values(header, data, colname):
    """Return list of numeric values for a column, skipping missing."""
    idx = header.index(colname)
    return [row[idx] for row in data if row[idx] != "?"]

def compute_stats(values):
    """Return dict with mean, median, min, max."""
    n = len(values)
    if n == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    s = sum(values) / n
    med = median(values)
    mi = min(values)
    ma = max(values)
    return {"mean": s, "median": med, "min": mi, "max": ma}

def best_worst(colname, values, direction):
    """Return best and worst values for the column given direction."""
    if not values:
        return {"best": None, "worst": None}
    if direction == "+":  # maximize
        return {"best": max(values), "worst": min(values)}
    else:  # minimize
        return {"best": min(values), "worst": max(values)}

# ----------------------------------------------------------------------
# NASA-01: summary statistics
# ----------------------------------------------------------------------
def nasa_01(header, data):
    goal_cols = [c for c in header if c.endswith(("+", "-"))]
    stats = {}
    for col in goal_cols:
        vals = col_values(header, data, col)
        cs = compute_stats(vals)
        bw = best_worst(col, vals, col[-1])
        stats[col] = {**cs, **bw}
    # Print human‑readable section
    print("== NASA-01 ==")
    for col, s in stats.items():
        print(f"{col}: mean={s['mean']:.3f}, median={s['median']:.3f}, "
              f"min={s['min']:.3f}, max={s['max']:.3f}, "
              f"best={s['best']:.3f}, worst={s['worst']:.3f}")
    print(f"ANSWER NASA-01 {json.dumps(stats, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-02: top 10 projects with best composite score
# ----------------------------------------------------------------------
def nasa_02(header, data):
    # Columns of interest
    kcol, ecol, dcol = "Kloc+", "Effort-", "Defects-"

    # Gather values (ignore missing)
    k_vals = col_values(header, data, kcol)
    e_vals = col_values(header, data, ecol)
    d_vals = col_values(header, data, dcol)

    # Compute ranges for normalization
    k_min, k_max = min(k_vals), max(k_vals)
    e_min, e_max = min(e_vals), max(e_vals)
    d_min, d_max = min(d_vals), max(d_vals)

    ranges = {
        kcol: (k_min, k_max),
        ecol: (e_min, e_max),
        dcol: (d_min, d_max),
    }

    # Compute composite score for each row (skip rows with any missing)
    scores = []
    for idx, row in enumerate(data, start=1):
        vals = {c: row[header.index(c)] for c in (kcol, ecol, dcol)}
        if any(v == "?" for v in vals.values()):
            continue
        # Normalize: Kloc+ (higher better)
        k = (vals[kcol] - k_min) / (k_max - k_min) if k_max != k_min else 0.0
        # Effort- and Defects- (lower better)
        e = (e_max - vals[ecol]) / (e_max - e_min) if e_max != e_min else 0.0
        d = (d_max - vals[dcol]) / (d_max - d_min) if d_max != d_min else 0.0
        score = k + e + d  # equal weight
        scores.append((score, idx))

    # Top 10 rows, best first
    top = sorted(scores, key=lambda x: x[0], reverse=True)[:10]
    top_rows = [idx for _, idx in top]

    # Print human‑readable section
    print("== NASA-02 ==")
    print(f"Top 10 project rows: {top_rows}")
    print(f"ANSWER NASA-02 {json.dumps({'top': top_rows}, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    nasa_01(header, data)
    nasa_02(header, data)

if __name__ == "__main__":
    main()

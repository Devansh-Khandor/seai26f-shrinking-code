#!/usr/bin/env python3
"""
skill_01.py <csv_path>
Generates the requested summaries for NASA-01 and NASA-02.
"""

import sys
import csv
import json
import math
from statistics import mean, median

def parse_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(rows, col, direction):
    # direction: '+' for maximize, '-' for minimize
    vals = [to_float(r[col]) for r in rows if to_float(r[col]) is not None]
    if not vals:
        return None
    n = len(vals)
    m = mean(vals)
    med = median(vals)
    mn = min(vals)
    mx = max(vals)
    if direction == '+':
        best = mx
        worst = mn
    else:
        best = mn
        worst = mx
    return {
        "mean": m,
        "median": med,
        "min": mn,
        "max": mx,
        "best": best,
        "worst": worst
    }

def nasa_01(rows):
    print("== NASA-01 ==")
    stats = {}
    for col, dirc in [("Kloc+", "+"), ("Effort-", "-"), ("Defects-", "-")]:
        s = compute_stats(rows, col, dirc)
        stats[col] = s
    # Simple textual summary (at least three lines)
    print(f"Total rows processed: {len(rows)}")
    print(f"Kloc+ stats: mean={stats['Kloc+']['mean']:.6g}, median={stats['Kloc+']['median']:.6g}, min={stats['Kloc+']['min']:.6g}, max={stats['Kloc+']['max']:.6g}, best={stats['Kloc+']['best']:.6g}, worst={stats['Kloc+']['worst']:.6g}")
    print(f"Effort- stats: mean={stats['Effort-']['mean']:.6g}, median={stats['Effort-']['median']:.6g}, min={stats['Effort-']['min']:.6g}, max={stats['Effort-']['max']:.6g}, best={stats['Effort-']['best']:.6g}, worst={stats['Effort-']['worst']:.6g}")
    # Machine-checked answer
    ans = {k: {
            "mean": v["mean"],
            "median": v["median"],
            "min": v["min"],
            "max": v["max"],
            "best": v["best"],
            "worst": v["worst"]
          } for k, v in stats.items()}
    print("ANSWER NASA-01", json.dumps(ans, separators=(',', ':')))

def nasa_02(rows):
    # Compute composite score: Kloc+ - Effort- - Defects-
    scores = []
    for idx, r in enumerate(rows, start=1):
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        d = to_float(r["Defects-"])
        if None in (k, e, d):
            continue
        score = k - e - d
        scores.append((score, idx))
    # Sort by score descending, then row number ascending
    scores.sort(key=lambda x: (-x[0], x[1]))
    top_rows = [idx for _, idx in scores[:10]]
    print("== NASA-02 ==")
    print(f"Top 10 rows with best combination: {top_rows}")
    # Machine-checked answer
    print("ANSWER NASA-02", json.dumps({"top": top_rows}, separators=(',', ':')))

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_01.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    nasa_01(rows)
    nasa_02(rows)

if __name__ == "__main__":
    main()

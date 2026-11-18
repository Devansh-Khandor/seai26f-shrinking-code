#!/usr/bin/env python3
import sys
import csv
import math
import json
from collections import defaultdict

def read_csv(path):
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):  # row numbers start at 1
            data.append((idx, row))
    return data, reader.fieldnames

def parse_float(val):
    if val is None or val == '':
        return None
    val = val.strip()
    if val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def compute_stats(data, field):
    vals = [parse_float(row[field]) for _, row in data if parse_float(row[field]) is not None]
    if not vals:
        return {'mean': None, 'median': None, 'min': None, 'max': None, 'best': None, 'worst': None}
    mean = sum(vals) / len(vals)
    median = sorted(vals)[len(vals)//2] if len(vals) % 2 == 1 else (sorted(vals)[len(vals)//2-1] + sorted(vals)[len(vals)//2]) / 2
    mn = min(vals)
    mx = max(vals)
    # determine direction by suffix
    if field.endswith('+'):
        best, worst = mx, mn
    else:
        best, worst = mn, mx
    return {'mean': mean, 'median': median, 'min': mn, 'max': mx, 'best': best, 'worst': worst}

def nasa_01(data):
    print("== NASA-01 ==")
    results = {}
    for field in ['Kloc+', 'Effort-', 'Defects-']:
        stats = compute_stats(data, field)
        results[field] = stats
        # print at least three lines of results
        print(f"{field}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, min={stats['min']:.2f}")
    # machine‑checked answer
    json_ans = json.dumps(results, separators=(',', ':'))
    print(f"ANSWER NASA-01 {json_ans}")

def nasa_02(data):
    print("\n== NASA-02 ==")
    # collect numeric values, skip rows with missing in any of the three fields
    rows = []
    for idx, row in data:
        k = parse_float(row['Kloc+'])
        e = parse_float(row['Effort-'])
        d = parse_float(row['Defects-'])
        if k is None or e is None or d is None:
            continue
        rows.append((idx, k, e, d))
    if not rows:
        print("No complete data available for NASA-02.")
        print("ANSWER NASA-02 {\"top\":[]}")
        return

    # compute ranges
    k_vals = [k for _, k, _, _ in rows]
    e_vals = [e for _, _, e, _ in rows]
    d_vals = [d for _, _, _, d in rows]
    k_min, k_max = min(k_vals), max(k_vals)
    e_min, e_max = min(e_vals), max(e_vals)
    d_min, d_max = min(d_vals), max(d_vals)

    # compute composite score: higher better
    scored = []
    for idx, k, e, d in rows:
        if k_max != k_min:
            k_norm = (k - k_min) / (k_max - k_min)
        else:
            k_norm = 0.0
        if e_max != e_min:
            e_norm = (e_max - e) / (e_max - e_min)
        else:
            e_norm = 0.0
        if d_max != d_min:
            d_norm = (d_max - d) / (d_max - d_min)
        else:
            d_norm = 0.0
        total = k_norm + e_norm + d_norm
        scored.append((total, idx))

    scored.sort(reverse=True)  # highest total first
    top10 = [idx for _, idx in scored[:10]]
    print("Top 10 projects by combined score:", top10)
    # machine‑checked answer
    json_ans = json.dumps({"top": top10}, separators=(',', ':'))
    print(f"ANSWER NASA-02 {json_ans}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    nasa_01(data)
    nasa_02(data)

if __name__ == "__main__":
    main()

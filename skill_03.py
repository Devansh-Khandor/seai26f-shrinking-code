#!/usr/bin/env python3
"""
skill_03.py

Usage: python skill_03.py <csv_path>
"""

import sys
import csv
import json
from statistics import median, mean as _mean, stdev, StatisticsError

def read_csv(path):
    """Read CSV and return list of rows (dict) with 1‑based indices."""
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

def group_stats(rows, key, goal):
    """Return dict of group -> stats dict with n, median, mean, std."""
    groups = {}
    for r in rows:
        val = r.get(key)
        if val is None:
            continue
        val = str(val)
        groups.setdefault(val, []).append(r[goal])
    stats = {}
    for g, values in groups.items():
        # remove None values from goal
        vals = [v for v in values if v is not None]
        n = len(vals)
        if n == 0:
            med = None
            m = None
            std = None
        else:
            med = round(median(vals), 2)
            m = round(_mean(vals), 2)
            try:
                std = round(stdev(vals), 2)
            except StatisticsError:
                std = 0.0
        stats[g] = {
            "n": n,
            "median": med,
            "mean": m,
            "std": std
        }
    return stats

def confidence_interval(mean, std, n):
    """Return (low, high) 95% CI for mean."""
    if n == 0:
        return (float('-inf'), float('inf'))
    se = std / (n ** 0.5)
    margin = 1.96 * se
    return (mean - margin, mean + margin)

def compare_groups(stats):
    """Return list of comparison dicts."""
    comps = []
    vals = sorted(stats.keys(), key=lambda x: (isinstance(x, str), x))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a = vals[i]
            b = vals[j]
            ma = stats[a]["mean"]
            mb = stats[b]["mean"]
            sa = stats[a]["std"]
            sb = stats[b]["std"]
            na = stats[a]["n"]
            nb = stats[b]["n"]
            ca_low, ca_high = confidence_interval(ma, sa, na)
            cb_low, cb_high = confidence_interval(mb, sb, nb)
            # CIs disjoint if one high < other low
            if ca_high < cb_low or cb_high < ca_low:
                verdict = "different"
            else:
                verdict = "same"
            comps.append({
                "a": a,
                "b": b,
                "verdict": verdict
            })
    return comps

def print_section(id, header, rows, key, goal):
    print(f"== {id} ==")
    stats = group_stats(rows, key, goal)
    for g in sorted(stats.keys(), key=lambda x: (isinstance(x, str), x)):
        s = stats[g]
        print(f"{g}: n={s['n']} median={s['median']} mean={s['mean']}")
    comps = compare_groups(stats)
    for c in comps:
        print(f"{c['a']} vs {c['b']}: verdict={c['verdict']}")
    answer = {
        "groups": {g: {"n": stats[g]["n"], "median": stats[g]["median"], "mean": stats[g]["mean"]}
                   for g in stats},
        "comparisons": comps
    }
    print(f"ANSWER {id} {json.dumps(answer, separators=(',', ':'))}")

def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        return

    # Goal column: Performance- (minimize)
    perf_col = "Performance-"

    print_section("APACHE-05", "KeepAlive", rows, "keepAlive", perf_col)
    print_section("APACHE-06", "AccessLog", rows, "accessLog", perf_col)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

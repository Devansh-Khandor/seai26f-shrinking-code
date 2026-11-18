#!/usr/bin/env python3
"""
skill_03.py: Compare fuel efficiency by cylinders (CAR-05) and
              performance by model year ranges (CAR-06).
"""

import csv
import json
import sys
from itertools import combinations
from math import sqrt, log2

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Convert a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def read_csv(path: str):
    """Return header list and rows list (each row is a list of parsed cells)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(parse_cell, row)) for row in reader]
    return header, rows

# --------------------------------------------------------------------------- #
# Statistical tests (taken from ezr but re‑implemented here)
# --------------------------------------------------------------------------- #

def cliffs(xs, ys, d=0.197):
    """Cliff's delta.  Returns True if distributions are similar."""
    gt = lt = j = k = 0
    xs = sorted(xs)
    ys = sorted(ys)
    for x in xs:
        while j < len(ys) and ys[j] < x: j += 1; k = j
        while k < len(ys) and ys[k] <= x: k += 1
        gt += j
        lt += len(ys) - k
    return abs(gt - lt) / (len(xs) * len(ys)) <= d

def ks(xs, ys, a=1.36):
    """Kolmogorov–Smirnov test.  Returns True if distributions are similar."""
    xs = sorted(xs)
    ys = sorted(ys)
    n, m = len(xs), len(ys)
    i = j = d = 0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] <= v: i += 1
        while j < m and ys[j] <= v: j += 1
        d = max(d, abs(i/n - j/m))
    return d <= a * sqrt((n + m) / (n * m))

def same(xs, ys):
    """Return True if two numerical samples are statistically indistinguishable."""
    return cliffs(xs, ys) and ks(xs, ys)

# --------------------------------------------------------------------------- #
# CAR-05: fuel efficiency by cylinders
# --------------------------------------------------------------------------- #

def car05(header, rows):
    """Compute CAR-05 results."""
    # Identify indices
    cyl_col = "Clndrs"
    mpg_col = "Mpg+"
    cyl_idx = header.index(cyl_col)
    mpg_idx = header.index(mpg_col)

    # Build groups
    groups = {}
    for r in rows:
        cyl = r[cyl_idx]
        mpg = r[mpg_idx]
        if cyl is None: continue
        if cyl not in groups: groups[cyl] = []
        if mpg is not None:
            groups[cyl].append(mpg)

    # Compute statistics per group
    group_stats = {}
    for cyl, mpgs in groups.items():
        n = len(mpgs)
        if n == 0:
            mean = median = 0.0
        else:
            mean = sum(mpgs) / n
            mpgs_sorted = sorted(mpgs)
            mid = n // 2
            median = (mpgs_sorted[mid] if n % 2 else
                      (mpgs_sorted[mid-1] + mpgs_sorted[mid]) / 2)
        group_stats[str(cyl)] = {"n": n, "mean": round(mean, 4), "median": round(median, 4)}

    # Comparisons
    comparisons = []
    cyl_values = sorted(groups.keys())
    for a, b in combinations(cyl_values, 2):
        mpg_a = groups[a]
        mpg_b = groups[b]
        verdict = "different" if not same(mpg_a, mpg_b) else "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Output
    print("== CAR-05 ==")
    for cyl in sorted(groups.keys()):
        stats = group_stats[str(cyl)]
        print(f"Cylinder {cyl}: n={stats['n']}, mean={stats['mean']}, median={stats['median']}")
    if comparisons:
        print("Comparisons:")
        for comp in comparisons:
            print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    # JSON answer
    answer = {"groups": group_stats, "comparisons": comparisons}
    print(f"ANSWER CAR-05 {json.dumps(answer, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# CAR-06: performance by model year ranges
# --------------------------------------------------------------------------- #

def car06(header, rows):
    """Compute CAR-06 results."""
    goal_cols = [c for c in header if c.endswith(("+", "-"))]
    goal_indices = [header.index(c) for c in goal_cols]
    model_col = "Model"
    model_idx = header.index(model_col)

    # Sort rows by Model value
    sorted_rows = sorted(rows, key=lambda r: r[model_idx] if r[model_idx] is not None else float("inf"))
    n_total = len(sorted_rows)
    split = n_total // 2

    group1 = sorted_rows[:split]
    group2 = sorted_rows[split:]

    def group_stats(g):
        n = len(g)
        mins = min(r[model_idx] for r in g if r[model_idx] is not None)
        maxs = max(r[model_idx] for r in g if r[model_idx] is not None)
        return {"from": mins, "to": maxs, "n": n}

    grp1_stats = group_stats(group1)
    grp2_stats = group_stats(group2)

    # Overall performance test: compare each goal column distribution
    verdict = "same"
    for gi in goal_indices:
        vals1 = [r[gi] for r in group1 if r[gi] is not None]
        vals2 = [r[gi] for r in group2 if r[gi] is not None]
        if not same(vals1, vals2):
            verdict = "different"
            break

    # Output
    print("== CAR-06 ==")
    print(f"Group 1: from={grp1_stats['from']} to={grp1_stats['to']} n={grp1_stats['n']}")
    print(f"Group 2: from={grp2_stats['from']} to={grp2_stats['to']} n={grp2_stats['n']}")
    print(f"Verdict: {verdict}")

    # JSON answer
    answer = {"groups": [grp1_stats, grp2_stats], "verdict": verdict}
    print(f"ANSWER CAR-06 {json.dumps(answer, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    car05(header, rows)
    car06(header, rows)

if __name__ == "__main__":
    main()

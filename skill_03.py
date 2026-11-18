#!/usr/bin/env python3
"""
skill_03.py: Implements CAR-05 and CAR-06 for used‑car dealership data.

Usage:
    python skill_03.py <path_to_csv>

Outputs:
    == CAR-05 ==
    ... results ...
    ANSWER CAR-05 <json>

    == CAR-06 ==
    ... results ...
    ANSWER CAR-06 <json>
"""

import csv
import json
import sys
import math
from typing import List, Tuple, Any, Dict, Iterable, Optional

# ------------------------------------------------------------------
# CSV reading utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for row in reader:
            conv_row: List[Any] = []
            for v in row:
                if v == "?":
                    conv_row.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv_row.append(float(v))
                        else:
                            conv_row.append(int(v))
                    except ValueError:
                        conv_row.append(v)
            rows.append(conv_row)
    return header, rows

# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def median(xs: List[float]) -> float:
    """Return median of a non‑empty list."""
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    if n % 2 == 1:
        return xs_sorted[mid]
    else:
        return (xs_sorted[mid - 1] + xs_sorted[mid]) / 2.0

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

# ------------------------------------------------------------------
# Statistical tests (copied from ezr)
# ------------------------------------------------------------------
def cliffs(xs: List[float], ys: List[float], d: float = 0.197) -> bool:
    """Sorted in.  Is the rank imbalance small enough?"""
    gt = lt = j = k = 0
    for x in xs:
        while j < len(ys) and ys[j] < x: j += 1; k = j
        while k < len(ys) and ys[k] <= x: k += 1
        gt += j; lt += len(ys) - k
    return abs(gt - lt) / (len(xs) * len(ys)) <= d

def ks(xs: List[float], ys: List[float], a: float = 1.36) -> bool:
    """Sorted in.  95% Kolmogorov‑Smirnov."""
    n, m = len(xs), len(ys)
    i = j = d = 0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] <= v: i += 1
        while j < m and ys[j] <= v: j += 1
        d = max(d, abs(i / n - j / m))
    return d <= a * math.sqrt((n + m) / (n * m))

def cohen(xs: List[float], ys: List[float], d: float = 0.35, eps: float = 0) -> bool:
    """Are the means within D pooled standard deviations?"""
    a = add(xs)
    b = add(ys)
    pool = ((a.n - 1) * div(a) ** 2 + (b.n - 1) * div(b) ** 2) / (a.n + b.n - 2)
    return abs(a.mu - b.mu) <= max(eps, d * math.sqrt(pool))

def same(xs: List[float], ys: List[float], eps: float = 0) -> bool:
    """Indistinguishable.  Always cliffs and ks; EPS adds cohen."""
    xs, ys = sorted(xs), sorted(ys)
    return (cliffs(xs, ys) and ks(xs, ys) and (not eps or cohen(xs, ys, d=eps)))

# Helpers for cohen/div
class Num:
    __slots__ = ("n", "mu", "m2", "sd")
    def __init__(self) -> None:
        self.n = self.mu = self.m2 = self.sd = 0

def add(col: Num, v: float) -> None:
    if v is None:
        return
    col.n += 1
    d = v - col.mu
    col.mu += d / col.n
    col.m2 = max(0, col.m2 + d * (v - col.mu))
    col.sd = 0 if col.n < 2 else math.sqrt(col.m2 / (col.n - 1))

def div(col: Num) -> float:
    return col.sd

# ------------------------------------------------------------------
# CAR-05: Cylinder groups
# ------------------------------------------------------------------
def car05(header: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    # Indices
    cyl_idx = header.index("Clndrs")
    mpg_idx = header.index("Mpg+")

    groups: Dict[Any, List[float]] = {}
    for row in rows:
        cyl = row[cyl_idx]
        mpg = row[mpg_idx]
        if mpg is None:
            continue
        groups.setdefault(cyl, []).append(mpg)

    # Prepare JSON groups
    json_groups: Dict[str, Dict[str, Any]] = {}
    for cyl, mpgs in groups.items():
        json_groups[str(cyl)] = {
            "n": len(mpgs),
            "median": round(median(mpgs), 4),
            "mean": round(mean(mpgs), 4)
        }

    # Comparisons
    comparisons: List[Dict[str, Any]] = []
    cyl_values = sorted(groups.keys(), key=lambda x: (float(x) if isinstance(x, (int, float)) else x))
    for i in range(len(cyl_values)):
        for j in range(i + 1, len(cyl_values)):
            a_val = cyl_values[i]
            b_val = cyl_values[j]
            verdict = "different" if not same(groups[a_val], groups[b_val]) else "same"
            comparisons.append({
                "a": a_val,
                "b": b_val,
                "verdict": verdict
            })

    # Print results
    print("== CAR-05 ==")
    for cyl, stats in json_groups.items():
        print(f"Cylinder {cyl}: n={stats['n']}, mean={stats['mean']:.4f}, median={stats['median']:.4f}")
    print("Comparisons:")
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")
    print()

    # Answer line
    answer = {"groups": json_groups, "comparisons": comparisons}
    print(f"ANSWER CAR-05 {json.dumps(answer, separators=(',', ':'))}")

# ------------------------------------------------------------------
# CAR-06: Model year ranges
# ------------------------------------------------------------------
def car06(header: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    # Identify goal columns
    goal_idxs = [i for i, name in enumerate(header) if name.endswith(('+', '-'))]
    if not goal_idxs:
        raise ValueError("No goal columns found")

    # Compute min/max for each goal (ignoring None)
    goal_mins: Dict[int, float] = {}
    goal_maxs: Dict[int, float] = {}
    for idx in goal_idxs:
        vals = [r[idx] for r in rows if r[idx] is not None]
        if not vals:
            continue
        goal_mins[idx] = min(vals)
        goal_maxs[idx] = max(vals)

    # Normalized performance metric per row
    def normalized_score(row: List[Any]) -> float:
        total = 0.0
        count = 0
        for idx in goal_idxs:
            v = row[idx]
            if v is None:
                continue
            mn = goal_mins[idx]
            mx = goal_maxs[idx]
            name = header[idx]
            if mn == mx:  # avoid division by zero
                norm = 0.5
            else:
                if name.endswith('+'):  # maximize
                    norm = (v - mn) / (mx - mn)
                else:  # minimize
                    norm = (mx - v) / (mx - mn)
            total += norm
            count += 1
        return total  # higher better

    # Build groups based on Model ranges
    model_idx = header.index("Model")
    models = [row[model_idx] for row in rows if row[model_idx] is not None]
    unique_models = sorted(set(models))
    if len(unique_models) < 2:
        # fallback: use all rows as single group
        groups_ranges = [(min(models), max(models))]
    else:
        mid = len(unique_models) // 2 - 1 if len(unique_models) % 2 == 0 else len(unique_models) // 2
        first_max = unique_models[mid]
        second_min = unique_models[mid + 1]
        groups_ranges = [(min(unique_models), first_max), (second_min, max(unique_models))]

    group_metrics: List[List[float]] = []
    group_counts: List[int] = []
    for r_from, r_to in groups_ranges:
        metrics = []
        count = 0
        for row in rows:
            m = row[model_idx]
            if m is None:
                continue
            if r_from <= m <= r_to:
                metrics.append(normalized_score(row))
                count += 1
        group_metrics.append(metrics)
        group_counts.append(count)

    # Determine verdict
    verdict = "different" if not same(group_metrics[0], group_metrics[1]) else "same"

    # Print results
    print("== CAR-06 ==")
    for idx, (r_from, r_to) in enumerate(groups_ranges):
        print(f"Group {idx + 1}: Model {r_from} to {r_to}, n={group_counts[idx]}")
    print(f"Verdict: {verdict}")
    print()

    # Answer line
    json_groups = [
        {"from": r_from, "to": r_to, "n": group_counts[idx]}
        for idx, (r_from, r_to) in enumerate(groups_ranges)
    ]
    answer = {"groups": json_groups, "verdict": verdict}
    print(f"ANSWER CAR-06 {json.dumps(answer, separators=(',', ':'))}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    car05(header, rows)
    car06(header, rows)

if __name__ == "__main__":
    main()

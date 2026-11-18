#!/usr/bin/env python3
"""
CAR-05: Compare fuel efficiency (Mpg+) across cylinder counts (Clndrs).
CAR-06: Compare overall performance across two model‑year ranges.
"""

import csv
import json
import sys
from typing import List, Dict, Tuple, Any, Set

# ----------------------------------------------------------------------
# Helpers for statistics
# ----------------------------------------------------------------------
def median(values: List[float]) -> float:
    """Return median of a non‑empty list."""
    n = len(values)
    v = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return v[mid]
    return (v[mid - 1] + v[mid]) / 2.0


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


# ----------------------------------------------------------------------
# Statistical tests (copy of ezr.same & helpers)
# ----------------------------------------------------------------------
def _cliffs(xs: List[float], ys: List[float], d: float = 0.197) -> bool:
    gt = lt = j = k = 0
    for x in xs:
        while j < len(ys) and ys[j] < x:
            j += 1
            k = j
        while k < len(ys) and ys[k] <= x:
            k += 1
        gt += j
        lt += len(ys) - k
    return abs(gt - lt) / (len(xs) * len(ys)) <= d


def _ks(xs: List[float], ys: List[float], a: float = 1.36) -> bool:
    n, m, i, j, d = len(xs), len(ys), 0, 0, 0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] <= v:
            i += 1
        while j < m and ys[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d <= a * ((n + m) / (n * m)) ** 0.5


def _cohen(xs: List[float], ys: List[float], d: float = 0.35, eps: float = 0) -> bool:
    a, b = _add(xs), _add(ys)
    pool = ((a.n - 1) * _div(a) ** 2 + (b.n - 1) * _div(b) ** 2) / (a.n + b.n - 2)
    return abs(a.mu - b.mu) <= max(eps, d * pool ** 0.5)


def _div(col):
    if isinstance(col, list):
        return (sum(v * v for v in col) / len(col)) ** 0.5
    return 0


def _add(vals):
    class Dummy:
        n = len(vals)
        mu = sum(vals) / n if n else 0

    return Dummy()


def same(xs: List[float], ys: List[float], eps: float = 0) -> bool:
    """Indistinguishable.  Always cliffs and ks; EPS adds cohen."""
    xs_sorted, ys_sorted = sorted(xs), sorted(ys)
    return _cliffs(xs_sorted, ys_sorted) and _ks(xs_sorted, ys_sorted) and (
        not eps or _cohen(xs_sorted, ys_sorted, eps=eps)
    )


# ----------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = None
                elif v == "?":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows, header


# ----------------------------------------------------------------------
# Goal columns
# ----------------------------------------------------------------------
def goal_columns(header: List[str]) -> List[str]:
    return [c for c in header if c.endswith(("+", "-"))]


# ----------------------------------------------------------------------
# Normalisation helpers for ydist
# ----------------------------------------------------------------------
def goal_stats(rows: List[Dict[str, Any]], goals: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for g in goals:
        vals = [r[g] for r in rows if r[g] is not None]
        if not vals:
            stats[g] = {"min": None, "max": None}
        else:
            stats[g] = {"min": min(vals), "max": max(vals)}
    return stats


def normalize_value(row: Dict[str, Any], g: str, st: Dict[str, float]) -> float:
    v = row[g]
    if v is None:
        return 0.0
    g_min = st[g]["min"]
    g_max = st[g]["max"]
    if g_min == g_max:
        return 0.0
    if g.endswith("-"):
        # minimize: closer to min is better
        return (g_max - v) / (g_max - g_min)
    else:
        return (v - g_min) / (g_max - g_min)


def ydist(row: Dict[str, Any], goals: List[str], st: Dict[str, float]) -> float:
    """Root‑mean‑square distance to goal vector."""
    vals = []
    for g in goals:
        if row[g] is None:
            return None
        vals.append(abs(normalize_value(row, g, st) - (0 if g.endswith("-") else 1)))
    return (sum(v * v for v in vals) / len(vals)) ** 0.5


# ----------------------------------------------------------------------
# CAR-05 logic
# ----------------------------------------------------------------------
def car05(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Any, Dict[str, Any]] = {}
    mpg_col = "Mpg+"
    for r in rows:
        cl = r["Clndrs"]
        mpg = r[mpg_col]
        if mpg is None:
            continue
        groups.setdefault(cl, {"values": []})["values"].append(mpg)
    groups_out: Dict[Any, Dict[str, Any]] = {}
    for cl, info in sorted(groups.items()):
        vals = info["values"]
        groups_out[str(cl)] = {
            "n": len(vals),
            "median": round(median(vals), 6),
            "mean": round(mean(vals), 6),
        }

    # Comparisons
    comps = []
    cl_keys = sorted(groups_out.keys(), key=lambda x: int(x))
    for i in range(len(cl_keys)):
        for j in range(i + 1, len(cl_keys)):
            a = int(cl_keys[i])
            b = int(cl_keys[j])
            vals_a = groups[a]["values"]
            vals_b = groups[b]["values"]
            verdict = "different" if not same(vals_a, vals_b) else "same"
            comps.append({"a": a, "b": b, "verdict": verdict})
    return {"groups": groups_out, "comparisons": comps}


# ----------------------------------------------------------------------
# CAR-06 logic
# ----------------------------------------------------------------------
def car06(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Determine model ranges
    model_vals = [r["Model"] for r in rows if r["Model"] is not None]
    sorted_models = sorted(set(model_vals))
    median_model = median(sorted_models)
    # First range: <= median
    r1 = [r for r in rows if r["Model"] is not None and r["Model"] <= median_model]
    # Second range: > median
    r2 = [r for r in rows if r["Model"] is not None and r["Model"] > median_model]

    # Compute overall performance metric (ydist) for each row
    goals = goal_columns([k for k in rows[0].keys()]) if rows else []
    gs = goal_stats(rows, goals)
    yvals: List[Tuple[Any, float]] = []
    for r in rows:
        yd = ydist(r, goals, gs)
        if yd is not None:
            yvals.append((r, yd))
    # group ydist lists
    y_r1 = [yd for r, yd in yvals if r in r1]
    y_r2 = [yd for r, yd in yvals if r in r2]
    verdict = "different" if not same(y_r1, y_r2) else "same"

    groups_out = [
        {"from": min(r["Model"] for r in r1), "to": max(r["Model"] for r in r1), "n": len(r1)},
        {"from": min(r["Model"] for r in r2), "to": max(r["Model"] for r in r2), "n": len(r2)},
    ]
    return {"groups": groups_out, "verdict": verdict}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, header = read_csv(csv_path)

    # CAR‑05
    print("== CAR-05 ==")
    res05 = car05(rows)
    # Pretty print groups
    print("Groups:")
    for cl, info in res05["groups"].items():
        print(f"  Clndrs {cl}: n={info['n']}, median={info['median']}, mean={info['mean']}")
    print("Comparisons:")
    for comp in res05["comparisons"]:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")
    print(f"ANSWER CAR-05 {json.dumps(res05, separators=(',', ':'))}")

    # CAR‑06
    print("\n== CAR-06 ==")
    res06 = car06(rows)
    print("Ranges:")
    for rg in res06["groups"]:
        print(f"  {rg['from']} – {rg['to']}: n={rg['n']}")
    print(f"Verdict: {res06['verdict']}")
    print(f"ANSWER CAR-06 {json.dumps(res06, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

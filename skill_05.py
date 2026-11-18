#!/usr/bin/env python3
"""
CAR-09: Identify rows with unusually poor fuel efficiency (Mpg+) and rank the
non‑goal characteristics most strongly associated with that outcome.

CAR-10: Find the common characteristics of vehicles with unfavorable overall
performance and show how they differ from better‑performing vehicles.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Parsing utilities
# ----------------------------------------------------------------------
def try_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for rnum, row in enumerate(reader, start=1):
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                v = v.strip()
                if v == "" or v == "?":
                    parsed[k] = None
                else:
                    fval = try_float(v)
                    parsed[k] = fval if fval is not None else v
            parsed["_rownum"] = rnum
            rows.append(parsed)
    return rows, headers

# ----------------------------------------------------------------------
# Goal / feature identification
# ----------------------------------------------------------------------
def goal_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith(("+", "-"))]

def feature_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if not c.endswith(("+", "-")) and not c.endswith("X")]

def is_numeric_col(col: str, rows: List[Dict[str, Any]]) -> bool:
    return any(isinstance(row[col], (int, float)) for row in rows if row[col] is not None)

# ----------------------------------------------------------------------
# Normalisation and ydist
# ----------------------------------------------------------------------
def compute_goal_bounds(rows: List[Dict[str, Any]], goals: List[str]) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for g in goals:
        vals = [row[g] for row in rows if isinstance(row.get(g), (int, float))]
        if not vals:
            bounds[g] = (0.0, 1.0)
        else:
            bounds[g] = (min(vals), max(vals))
    return bounds

def norm_val(g: str, val: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if lo == hi:
        return 0.0
    if g.endswith("+"):
        return (val - lo) / (hi - lo)
    else:
        return (hi - val) / (hi - lo)

def ydist(row: Dict[str, Any], goals: List[str], bounds: Dict[str, Tuple[float, float]]) -> Optional[float]:
    parts = []
    for g in goals:
        v = row.get(g)
        if v is None or not isinstance(v, (int, float)):
            return None
        norm = norm_val(g, v, bounds[g])
        ideal = 1.0 if g.endswith("+") else 0.0
        parts.append((norm - ideal) ** 2)
    return math.sqrt(sum(parts) / len(parts))

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def mode(values: List[Any]) -> Any:
    if not values:
        return None
    c = Counter(values)
    return c.most_common(1)[0][0]

def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sx * sy)
    if denom == 0:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / denom

# ----------------------------------------------------------------------
# CAR-09 implementation
# ----------------------------------------------------------------------
def car09(rows: List[Dict[str, Any]], features: List[str]) -> Dict[str, Any]:
    mpg_col = "Mpg+"
    # collect all numeric Mpg+ values
    mpg_vals = [row[mpg_col] for row in rows if isinstance(row.get(mpg_col), (int, float))]
    if not mpg_vals:
        poor_rows = []
    else:
        mpg_vals_sorted = sorted(mpg_vals)
        # 20th percentile
        idx = max(0, int(len(mpg_vals_sorted) * 0.2) - 1)
        thresh = mpg_vals_sorted[idx]
        poor_rows = [row["_rownum"] for row in rows
                     if isinstance(row.get(mpg_col), (int, float)) and row[mpg_col] <= thresh]
        # ensure at least 5
        if len(poor_rows) < 5:
            poor_rows = sorted([row["_rownum"] for row in rows
                                 if isinstance(row.get(mpg_col), (int, float))])[:5]
    # Association ranking
    assoc: List[Tuple[str, float]] = []
    for f in features:
        if is_numeric_col(f, rows):
            xs = [row[f] for row in rows if isinstance(row.get(f), (int, float)) and isinstance(row.get(mpg_col), (int, float))]
            ys = [row[mpg_col] for row in rows if isinstance(row.get(f), (int, float)) and isinstance(row.get(mpg_col), (int, float))]
            corr = abs(pearson(xs, ys))
            assoc.append((f, corr))
        else:
            # categorical: compute mean per category
            groups: Dict[Any, List[float]] = defaultdict(list)
            for row in rows:
                val = row.get(f)
                tgt = row.get(mpg_col)
                if val is not None and tgt is not None and isinstance(tgt, (int, float)):
                    groups[val].append(tgt)
            if len(groups) <= 1:
                diff = 0.0
            else:
                means = [mean(vs) for vs in groups.values()]
                diff = max(means) - min(means)
            assoc.append((f, diff))
    assoc.sort(key=lambda x: -x[1])
    ranking = [f for f, _ in assoc[:max(3, len(assoc))]]
    return {"poor": poor_rows, "ranking": ranking}

# ----------------------------------------------------------------------
# CAR-10 implementation
# ----------------------------------------------------------------------
def car10(rows: List[Dict[str, Any]], features: List[str], goals: List[str], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    # compute ydist for all rows
    ydist_list: List[Tuple[int, float]] = []
    for row in rows:
        yd = ydist(row, goals, bounds)
        if yd is not None:
            ydist_list.append((row["_rownum"], yd))
    ydist_list.sort(key=lambda x: x[1])  # ascending (better first)
    n_total = len(ydist_list)
    n_pick = max(5, int(n_total * 0.3))
    good_rows = [num for num, _ in ydist_list[:n_pick]]
    bad_rows = [num for num, _ in ydist_list[-n_pick:]]
    # typical values
    def typical(vals: List[Any], numeric: bool) -> Any:
        if numeric:
            return round(mean([v for v in vals if isinstance(v, (int, float))]), 6)
        else:
            return mode([v for v in vals if v is not None])
    diff_list: List[Dict[str, Any]] = []
    for f in features:
        bad_vals = [row[f] for row in rows if row["_rownum"] in bad_rows]
        good_vals = [row[f] for row in rows if row["_rownum"] in good_rows]
        numeric = is_numeric_col(f, rows)
        if numeric:
            b = typical(bad_vals, True)
            g = typical(good_vals, True)
            diff = abs(b - g)
        else:
            b = typical(bad_vals, False)
            g = typical(good_vals, False)
            diff = 0 if b == g else 1
        diff_list.append({"feature": f, "bad": b, "good": g, "diff": diff})
    diff_list.sort(key=lambda x: -x["diff"])
    differences = [{"feature": d["feature"], "bad": d["bad"], "good": d["good"]} for d in diff_list[:max(3, len(diff_list))]]
    return {"bad": bad_rows, "good": good_rows, "differences": differences}

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)
    goals = goal_columns(headers)
    features = feature_columns(headers)
    bounds = compute_goal_bounds(rows, goals)

    # CAR-09
    print("== CAR-09 ==")
    res09 = car09(rows, features)
    print(f"Poor Mpg+ rows: {res09['poor']}")
    print(f"Top features: {res09['ranking']}")
    print(f"ANSWER CAR-09 {json.dumps(res09, separators=(',', ':'))}")

    # CAR-10
    print("\n== CAR-10 ==")
    res10 = car10(rows, features, goals, bounds)
    print(f"Bad performance rows: {res10['bad']}")
    print(f"Good performance rows: {res10['good']}")
    print(f"Feature differences: {res10['differences']}")
    print(f"ANSWER CAR-10 {json.dumps(res10, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

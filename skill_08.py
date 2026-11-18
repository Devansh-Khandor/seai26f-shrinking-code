#!/usr/bin/env python3
"""
CAR-15: For a poorly rated vehicle, show me alternative vehicles in our historical inventory
with similar characteristics but better recorded performance, and explain which differences
could account for the improvement.

CAR-16: For a vehicle that does not meet our preferred performance targets,
identify the closest available alternative that meets more of those targets while
changing as few vehicle characteristics as possible.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read CSV, return list of rows (dict) and header list.  Missing cells => None."""
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
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            parsed["_rownum"] = rnum
            rows.append(parsed)
    return rows, headers

# ----------------------------------------------------------------------
# Column classification
# ----------------------------------------------------------------------
def goal_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if h.endswith(("+", "-"))]

def x_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if h.endswith("X")]

def input_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if not h.endswith(("+", "-")) and not h.endswith("X")]

# ----------------------------------------------------------------------
# Bounds helpers
# ----------------------------------------------------------------------
def compute_bounds(rows: List[Dict[str, Any]], cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """Return min and max for each column over rows with non‑missing numeric values."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            bounds[c] = (min(vals), max(vals))
        else:
            bounds[c] = (0.0, 1.0)
    return bounds

# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------
def norm_val(val: Any, lo: float, hi: float) -> float:
    if isinstance(val, (int, float)):
        if hi == lo:
            return 0.0
        return (val - lo) / (hi - lo)
    return 0.0

# ----------------------------------------------------------------------
# ydist calculation
# ----------------------------------------------------------------------
def ydist(row: Dict[str, Any], goals: List[str], goal_bounds: Dict[str, Tuple[float, float]]) -> Optional[float]:
    parts: List[float] = []
    for g in goals:
        v = row.get(g)
        if v is None or not isinstance(v, (int, float)):
            return None
        lo, hi = goal_bounds[g]
        norm = norm_val(v, lo, hi)
        ideal = 1.0 if g.endswith("+") else 0.0
        parts.append((norm - ideal) ** 2)
    return math.sqrt(sum(parts) / len(parts)) if parts else None

# ----------------------------------------------------------------------
# Identify worst row
# ----------------------------------------------------------------------
def worst_row(rows: List[Dict[str, Any]], goals: List[str], goal_bounds: Dict[str, Tuple[float, float]]) -> Tuple[int, Optional[float]]:
    worst_id = None
    worst_dist = -1.0
    for r in rows:
        d = ydist(r, goals, goal_bounds)
        if d is None:
            continue
        if d > worst_dist:
            worst_dist = d
            worst_id = r["_rownum"]
    return worst_id, worst_dist

# ----------------------------------------------------------------------
# Similarity distance over input columns
# ----------------------------------------------------------------------
def input_distance(a: Dict[str, Any], b: Dict[str, Any],
                   input_cols: List[str],
                   num_bounds: Dict[str, Tuple[float, float]],
                   cat_set: Dict[str, set]) -> float:
    diff2 = 0.0
    for c in input_cols:
        va = a.get(c)
        vb = b.get(c)
        if va is None or vb is None:
            d = 1.0
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            lo, hi = num_bounds[c]
            da = norm_val(va, lo, hi)
            db = norm_val(vb, lo, hi)
            d = abs(da - db)
        else:
            d = 0.0 if va == vb else 1.0
        diff2 += d * d
    return math.sqrt(diff2 / len(input_cols)) if input_cols else 0.0

# ----------------------------------------------------------------------
# CAR-15 logic
# ----------------------------------------------------------------------
def car15(rows: List[Dict[str, Any]], goals: List[str], input_cols: List[str],
          num_bounds: Dict[str, Tuple[float, float]], cat_set: Dict[str, set],
          history_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    goal_bounds = compute_bounds(rows, goals)
    worst_id, worst_dist = worst_row(rows, goals, goal_bounds)
    if worst_id is None:
        return {"row": None, "alternatives": []}

    poor_row = next(r for r in rows if r["_rownum"] == worst_id)
    # Find alternatives
    alternatives: List[Tuple[float, Dict[str, Any], float]] = []
    for h in history_rows:
        d = ydist(h, goals, goal_bounds)
        if d is None or d >= worst_dist:
            continue
        dist_inp = input_distance(poor_row, h, input_cols, num_bounds, cat_set)
        alternatives.append((dist_inp, h, d))

    alternatives.sort(key=lambda x: (x[0], x[2]))
    # Pick up to 3 best
    bests = alternatives[:3]
    result_alts = []
    for dist, alt, d in bests:
        changes: Dict[str, List[Any]] = {}
        for c in input_cols:
            pv = poor_row.get(c)
            av = alt.get(c)
            if pv != av:
                changes[c] = [pv, av]
        result_alts.append({"row": alt["_rownum"], "changes": changes})
    return {"row": worst_id, "alternatives": result_alts}

# ----------------------------------------------------------------------
# CAR-16 logic
# ----------------------------------------------------------------------
def car16(rows: List[Dict[str, Any]], goals: List[str], input_cols: List[str],
          history_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # thresholds: best values for each goal
    thresholds: Dict[str, Any] = {}
    for g in goals:
        vals = [r[g] for r in rows if isinstance(r.get(g), (int, float))]
        if not vals:
            thresholds[g] = None
            continue
        thresholds[g] = max(vals) if g.endswith("+") else min(vals)

    def meets(row: Dict[str, Any]) -> int:
        cnt = 0
        for g, thr in thresholds.items():
            if thr is None:
                continue
            val = row.get(g)
            if val is None:
                continue
            if g.endswith("+"):
                if val >= thr:
                    cnt += 1
            else:
                if val <= thr:
                    cnt += 1
        return cnt

    # pick a row that does not meet all targets: use worst row
    goal_bounds = compute_bounds(rows, goals)
    worst_id, _ = worst_row(rows, goals, goal_bounds)
    if worst_id is None:
        return {"row": None, "targets": thresholds, "alternative": None, "changed": []}
    row = next(r for r in rows if r["_rownum"] == worst_id)
    row_cnt = meets(row)

    # candidates from history
    candidates: List[Tuple[int, int, Dict[str, Any]]] = []  # (cnt, changes, row)
    for h in history_rows:
        cnt = meets(h)
        if cnt <= row_cnt:
            continue
        # count changes in input columns
        changes = 0
        for c in input_cols:
            if h.get(c) != row.get(c):
                changes += 1
        candidates.append((cnt, changes, h))

    if not candidates:
        return {"row": worst_id, "targets": thresholds, "alternative": None, "changed": []}

    # best candidate: max cnt, min changes, min rownum
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]["_rownum"]))
    best_cnt, best_chg, best_row_ = candidates[0]
    changed_cols = [c for c in input_cols if best_row_.get(c) != row.get(c)]
    return {"row": worst_id,
            "targets": thresholds,
            "alternative": best_row_["_rownum"],
            "changed": changed_cols}

# ----------------------------------------------------------------------
# Helper to print differences nicely
# ----------------------------------------------------------------------
def format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, headers = parse_csv(csv_path)
    if not rows:
        print("No data")
        return

    goals = goal_columns(headers)
    x_cols = x_columns(headers)
    inputs = input_columns(headers)

    # Determine arrivals and history
    total = len(rows)
    arrivals = round(0.2 * total)
    history = rows[:total - arrivals]
    arrivals_rows = rows[total - arrivals:]

    # Compute numeric bounds for inputs over history
    num_bounds = compute_bounds(history, inputs)
    # Determine categorical columns
    cat_set: Dict[str, set] = {}
    for c in inputs:
        if all(isinstance(r.get(c), (int, float)) or r.get(c) is None for r in rows):
            # if all numeric (or missing) treat numeric
            pass
        else:
            # gather unique non-missing values
            cat_set[c] = {r[c] for r in rows if r.get(c) is not None}

    # CAR-15
    print("== CAR-15 ==")
    car15_res = car15(rows, goals, inputs, num_bounds, cat_set, history)
    print(f"Poor row: {car15_res['row']}")
    if car15_res["alternatives"]:
        print("Alternatives:")
        for alt in car15_res["alternatives"]:
            print(f"  Row {alt['row']} changes:")
            for col, vals in alt["changes"].items():
                print(f"    {col}: {format_value(vals[0])} -> {format_value(vals[1])}")
    else:
        print("  No better alternatives found.")
    print(f"ANSWER CAR-15 {json.dumps(car15_res, separators=(',', ':'))}")

    # CAR-16
    print("\n== CAR-16 ==")
    car16_res = car16(rows, goals, inputs, history)
    print(f"Row not meeting all targets: {car16_res['row']}")
    print("Target thresholds:")
    for g, thr in car16_res["targets"].items():
        print(f"  {g}: {format_value(thr)}")
    if car16_res["alternative"] is not None:
        print(f"Alternative: {car16_res['alternative']}")
        print("Changed columns:")
        for col in car16_res["changed"]:
            print(f"  {col}")
    else:
        print("  No suitable alternative found.")
    print(f"ANSWER CAR-16 {json.dumps(car16_res, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

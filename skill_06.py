#!/usr/bin/env python3
"""
CAR-11: Explain a recommended vehicle using similar vehicles.
CAR-12: Compare two vehicles and explain trade‑offs.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ----------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------
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
    return [c for c in headers if c.endswith(("+", "-"))]

def x_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith("X")]

def feature_columns(headers: List[str]) -> List[str]:
    # Features that are not X and not goals
    return [c for c in headers if not c.endswith("X") and not c.endswith(("+", "-"))]

# ----------------------------------------------------------------------
# Min/Max calculation for normalisation
# ----------------------------------------------------------------------
def compute_bounds(rows: List[Dict[str, Any]], cols: List[str]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
        if not vals:
            bounds[c] = (0.0, 1.0)
        else:
            bounds[c] = (min(vals), max(vals))
    return bounds

# ----------------------------------------------------------------------
# Normalisation helpers
# ----------------------------------------------------------------------
def norm_val(v: Any, lo: float, hi: float) -> float:
    if lo == hi:
        return 0.0
    return (v - lo) / (hi - lo)

# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def score_row(row: Dict[str, Any], goal_cols: List[str],
              goal_bounds: Dict[str, Tuple[float, float]]) -> Optional[float]:
    parts: List[float] = []
    for g in goal_cols:
        val = row.get(g)
        if val is None or not isinstance(val, (int, float)):
            return None
        lo, hi = goal_bounds[g]
        norm = norm_val(val, lo, hi)
        target = 1.0 if g.endswith("+") else 0.0
        parts.append(norm - target)
    # return average normalised score (higher is better)
    return 1.0 - sum(p * p for p in parts) ** 0.5 / math.sqrt(len(parts))

# ----------------------------------------------------------------------
# Distance between two rows over feature columns
# ----------------------------------------------------------------------
def distance(a: Dict[str, Any], b: Dict[str, Any],
             feat_cols: List[str],
             feat_bounds: Dict[str, Tuple[float, float]]) -> float:
    diff2 = 0.0
    for f in feat_cols:
        va = a.get(f)
        vb = b.get(f)
        if va is None or vb is None:
            d = 1.0
        else:
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                lo, hi = feat_bounds[f]
                da = norm_val(va, lo, hi)
                db = norm_val(vb, lo, hi)
                d = abs(da - db)
            else:  # categorical
                d = 0.0 if va == vb else 1.0
        diff2 += d * d
    return math.sqrt(diff2 / len(feat_cols)) if feat_cols else 0.0

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)
    if not rows:
        print("No data")
        return

    goals = goal_columns(headers)
    xs = x_columns(headers)
    feats = feature_columns(headers)

    # bounds for normalisation
    goal_bounds = compute_bounds(rows, goals)
    feat_bounds = compute_bounds(rows, feats)

    # --------------------- CAR-11 ----------------------------------
    print("== CAR-11 ==")

    # Budget: at most min(30, number_of_rows // 5) rows evaluated
    budget = min(30, len(rows) // 5)
    scored: List[Tuple[int, float]] = []  # (rownum, score)
    for r in rows:
        if len(scored) >= budget:
            break
        s = score_row(r, goals, goal_bounds)
        if s is not None:
            scored.append((r["_rownum"], s))

    if not scored:
        print("No evaluable rows for recommendation.")
        rec_rownum = None
        rec_score = None
    else:
        # best row
        rec_rownum, rec_score = max(scored, key=lambda x: (x[1], -x[0]))
        rec_row = next(r for r in rows if r["_rownum"] == rec_rownum)

    # Find at least 3 nearest neighbours (excluding itself)
    neighs: List[Tuple[float, int, Dict[str, Any]]] = []
    for r in rows:
        if r["_rownum"] == rec_rownum:
            continue
        d = distance(rec_row, r, feats, feat_bounds)
        neighs.append((d, r["_rownum"], r))

    neighs.sort(key=lambda x: x[0])
    neighbors_out = []
    for d, rn, r in neighs[:3]:
        values = {g: r.get(g) for g in goals}
        neighbors_out.append({"row": rn, "values": values})

    # JSON answer
    ans_car11 = {
        "recommended": rec_rownum,
        "neighbors": neighbors_out
    }
    print(f"Recommended row: {rec_rownum}")
    print("Nearest neighbours:")
    for nb in neighbors_out:
        print(f"  Row {nb['row']}: {nb['values']}")
    print(f"ANSWER CAR-11 {json.dumps(ans_car11, separators=(',', ':'))}")

    # --------------------- CAR-12 ----------------------------------
    print("\n== CAR-12 ==")

    if len(scored) < 2:
        print("Not enough rows to compare.")
        a_num = b_num = None
        a_score = b_score = None
        diff_list = []
        tradeoffs = []
    else:
        # choose best two
        scored_sorted = sorted(scored, key=lambda x: (-x[1], x[0]))
        a_num, a_score = scored_sorted[0]
        b_num, b_score = scored_sorted[1]
        a_row = next(r for r in rows if r["_rownum"] == a_num)
        b_row = next(r for r in rows if r["_rownum"] == b_num)

        # differences on all columns (excluding X)
        diff_list = []
        for col in headers:
            if col in xs:
                continue
            av = a_row.get(col)
            bv = b_row.get(col)
            if av == bv:
                continue
            diff_list.append({"feature": col, "a": av, "b": bv})
        if not diff_list:
            diff_list = [{"feature": headers[0], "a": a_row.get(headers[0]),
                         "b": b_row.get(headers[0])}]

        # tradeoffs: goal columns where the non‑preferred row (b) is better
        tradeoffs = []
        for g in goals:
            av = a_row.get(g)
            bv = b_row.get(g)
            if av is None or bv is None:
                continue
            better = False
            if g.endswith("-"):
                better = bv < av
            else:
                better = bv > av
            if better:
                tradeoffs.append(g)

    ans_car12 = {
        "a": a_num,
        "b": b_num,
        "preferred": a_num if a_score is not None and a_score >= (b_score or -math.inf) else b_num,
        "scores": {str(a_num): a_score, str(b_num): b_score} if a_num is not None else {},
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }
    print(f"Vehicle A (row {a_num}): score={a_score}")
    print(f"Vehicle B (row {b_num}): score={b_score}")
    print("Differences:")
    for d in diff_list:
        print(f"  {d['feature']}: A={d['a']}, B={d['b']}")
    print("Tradeoffs where B is better:")
    print(f"  {', '.join(tradeoffs)}")
    print(f"ANSWER CAR-12 {json.dumps(ans_car12, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

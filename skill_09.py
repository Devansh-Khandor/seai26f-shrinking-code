#!/usr/bin/env python3
"""
CAR-17: Flag vehicle records whose combinations of characteristics are unusually different from the rest of our inventory so that staff can review them.
CAR-18: Identify vehicles whose reported fuel efficiency or weight differs substantially from what similar vehicles would lead us to expect, so we can check for possible recording mistakes.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ---------- CSV parsing -------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return list of rows (dict of column->value) and header list. Missing cells become None."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for rnum, row in enumerate(reader, start=1):
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v.strip() == "" or v.strip() == "?":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            parsed["_rownum"] = rnum
            rows.append(parsed)
    return rows, headers


# ---------- column helpers ---------------------------------------------
def goal_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith(("+", "-"))]


def x_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith("X")]


def input_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if not c.endswith(("+", "-")) and not c.endswith("X")]


# ---------- normalization ---------------------------------------------
def compute_numeric_bounds(rows: List[Dict[str, Any]], nums: List[str]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for n in nums:
        vals = [r[n] for r in rows if isinstance(r.get(n), (int, float))]
        if vals:
            bounds[n] = (min(vals), max(vals))
        else:
            bounds[n] = (0.0, 1.0)
    return bounds


def normalize_value(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (v - lo) / (hi - lo)


# ---------- distance -----------------------------------------------
def pairwise_distance(r1: Dict[str, Any], r2: Dict[str, Any],
                      norm_vals1: Dict[str, float], norm_vals2: Dict[str, float],
                      num_cols: List[str], cat_cols: List[str]) -> float:
    sq = 0.0
    cnt = 0
    # numeric
    for c in num_cols:
        v1 = norm_vals1.get(c)
        v2 = norm_vals2.get(c)
        if v1 is None or v2 is None:
            continue
        d = v1 - v2
        sq += d * d
        cnt += 1
    # categorical
    for c in cat_cols:
        v1 = r1.get(c)
        v2 = r2.get(c)
        if v1 is None or v2 is None:
            continue
        d = 0.0 if v1 == v2 else 1.0
        sq += d * d
        cnt += 1
    if cnt == 0:
        return 0.0
    return math.sqrt(sq / cnt)


# ---------- CAR-17 -----------------------------------------------
def car17(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    inp_cols = input_columns([r for r in rows[0].keys() if r != "_rownum"])
    # separate numeric and categorical input columns
    num_inp = [c for c in inp_cols if isinstance(rows[0].get(c), (int, float))]
    cat_inp = [c for c in inp_cols if not isinstance(rows[0].get(c), (int, float))]

    bounds = compute_numeric_bounds(rows, num_inp)

    # precompute normalized numeric values per row
    norm_vals_list: List[Dict[str, float]] = []
    for r in rows:
        nv = {}
        for c in num_inp:
            v = r.get(c)
            if isinstance(v, (int, float)):
                nv[c] = normalize_value(v, *bounds[c])
        norm_vals_list.append(nv)

    # compute average nearest‑neighbor distance for each row
    k = 5
    scores: Dict[int, float] = {}
    for i, r in enumerate(rows):
        dists = []
        for j, other in enumerate(rows):
            if i == j:
                continue
            d = pairwise_distance(r, other, norm_vals_list[i], norm_vals_list[j], num_inp, cat_inp)
            dists.append(d)
        dists.sort()
        avg = sum(dists[:k]) / k
        scores[r["_rownum"]] = avg

    # Determine threshold: mean + 2*std; if none flagged, fallback to top 10
    vals = list(scores.values())
    mean_s = sum(vals) / len(vals) if vals else 0.0
    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in vals) / len(vals)) if vals else 0.0
    thresh = mean_s + 2 * std_s
    flagged = [rn for rn, sc in scores.items() if sc > thresh]
    if not flagged:
        flagged = sorted(scores.keys(), key=lambda r: -scores[r])[:10]
    # limit to 40
    flagged = sorted(flagged)[:40]

    return {"flagged": flagged, "scores": {str(rn): round(scores[rn], 6) for rn in flagged}}


# ---------- CAR-18 -----------------------------------------------
def car18(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    goal_cols = goal_columns([r for r in rows[0].keys() if r != "_rownum"])
    inp_cols = input_columns([r for r in rows[0].keys() if r != "_rownum"])

    # numeric vs categorical input
    num_inp = [c for c in inp_cols if isinstance(rows[0].get(c), (int, float))]
    cat_inp = [c for c in inp_cols if not isinstance(rows[0].get(c), (int, float))]

    bounds = compute_numeric_bounds(rows, num_inp)

    norm_vals_list: List[Dict[str, float]] = []
    for r in rows:
        nv = {}
        for c in num_inp:
            v = r.get(c)
            if isinstance(v, (int, float)):
                nv[c] = normalize_value(v, *bounds[c])
        norm_vals_list.append(nv)

    # For each target, collect differences
    diff_by_target: Dict[str, List[float]] = defaultdict(list)
    exp_vals: Dict[int, Dict[str, float]] = defaultdict(dict)

    k = 5
    for i, r in enumerate(rows):
        # find neighbors
        dists = []
        for j, other in enumerate(rows):
            if i == j:
                continue
            d = pairwise_distance(r, other, norm_vals_list[i], norm_vals_list[j], num_inp, cat_inp)
            dists.append((d, j))
        dists.sort()
        neigh_idx = [idx for _, idx in dists[:k]]
        # expected per target
        for g in goal_cols:
            vals = []
            for ni in neigh_idx:
                v = rows[ni].get(g)
                if isinstance(v, (int, float)):
                    vals.append(v)
            if vals:
                exp = sum(vals) / len(vals)
                exp_vals[r["_rownum"]][g] = exp

    # compute differences and z-scores
    for rn, row in enumerate(rows, start=1):
        for g in goal_cols:
            actual = row.get(g)
            exp = exp_vals.get(rn, {}).get(g)
            if actual is None or exp is None:
                continue
            diff = actual - exp
            diff_by_target[g].append(diff)

    # compute mean,std per target
    thresh_z = 2.0
    flagged_entries = []
    for g in goal_cols:
        diffs = diff_by_target[g]
        if not diffs:
            continue
        mean_d = sum(diffs) / len(diffs)
        std_d = math.sqrt(sum((x - mean_d) ** 2 for x in diffs) / len(diffs))
        if std_d == 0:
            continue
        for rn, row in enumerate(rows, start=1):
            actual = row.get(g)
            exp = exp_vals.get(rn, {}).get(g)
            if actual is None or exp is None:
                continue
            diff = actual - exp
            z = abs(diff - mean_d) / std_d
            if z > thresh_z:
                flagged_entries.append({
                    "row": rn,
                    "column": g,
                    "actual": actual,
                    "expected": round(exp, 6)
                })

    # sort by abs difference descending, limit to 40
    flagged_entries.sort(key=lambda e: -abs(e["actual"] - e["expected"]))
    flagged_entries = flagged_entries[:40]

    return {"flagged": flagged_entries}


# ---------- main --------------------------------------------------------
def main(csv_path: str) -> None:
    rows, headers = parse_csv(csv_path)

    # CAR-17
    print("== CAR-17 ==")
    res17 = car17(rows)
    print(f"Flagged rows: {res17['flagged']}")
    print("Scores:")
    for rn in res17["flagged"]:
        print(f"  Row {rn}: {res17['scores'][str(rn)]}")
    print(f"ANSWER CAR-17 {json.dumps(res17, separators=(',', ':'))}")

    # CAR-18
    print("\n== CAR-18 ==")
    res18 = car18(rows)
    print(f"Flagged entries: {len(res18['flagged'])}")
    for e in res18["flagged"]:
        print(f"  Row {e['row']} {e['column']} actual {e['actual']} expected {e['expected']}")
    print(f"ANSWER CAR-18 {json.dumps(res18, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_09.py

APACHE-17 – Identify configurations whose recorded performance values
are unusually different from what their settings and comparable
historical configurations would lead us to expect.

APACHE-18 – Flag configurations whose combinations of settings differ
substantially from the most similar configurations in our historical
benchmark records, even when their measured performance is not obviously poor.

The script prints, for each request, a header line (e.g. "== APACHE-17 =="),
followed by at least three lines of textual results, and finally a
machine‑checked answer line:
    ANSWER APACHE-17 <json>
    ANSWER APACHE-18 <json>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Tuple[int, Dict[str, str]]], List[str]]:
    """Return list of (row_number, dict_of_cells) and fieldnames."""
    rows: List[Tuple[int, Dict[str, str]]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, dict(rec)))
    return rows, fieldnames


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_x(col: str) -> bool:
    return col.endswith("X")


def is_numeric(v: str) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def parse_value(v: str) -> Any:
    """Parse a CSV cell into a float or keep as string."""
    if v in ("", "?", None):
        return None
    if is_numeric(v):
        return float(v)
    return v


def distance(row_a: Dict[str, str],
             row_b: Dict[str, str],
             feature_cols: List[str]) -> float:
    """
    Euclidean distance over feature columns.
    Numeric values are used as floats; others are treated as 0/1
    (0 if equal, 1 if different). Missing values are ignored.
    If no overlapping features remain, return +inf.
    """
    sq = 0.0
    cnt = 0
    for col in feature_cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va in ("", "?", None) or vb in ("", "?", None):
            continue
        try:
            fa = float(va)
            fb = float(vb)
            sq += (fa - fb) ** 2
        except Exception:
            sq += 0 if va == vb else 1
        cnt += 1
    return math.sqrt(sq) if cnt > 0 else math.inf


def nearest_neighbors(target_idx: int,
                      rows: List[Tuple[int, Dict[str, str]]],
                      feature_cols: List[str],
                      k: int,
                      exclude_self: bool = True) -> List[int]:
    """
    Return indices (in the rows list) of the k nearest neighbors of
    the target row (by distance over feature_cols).  If exclude_self,
    the target row itself is not considered.
    """
    target = rows[target_idx][1]
    dists = []
    for idx, (_, row) in enumerate(rows):
        if exclude_self and idx == target_idx:
            continue
        d = distance(target, row, feature_cols)
        if math.isfinite(d):
            dists.append((d, idx))
    dists.sort(key=lambda x: x[0])
    return [idx for _, idx in dists[:k]]


# ----------------------------------------------------------------------
# APACHE-17 logic
# ----------------------------------------------------------------------
def apache_17(rows: List[Tuple[int, Dict[str, str]]],
              fieldnames: List[str],
              goal_col: str) -> List[Dict[str, Any]]:
    """
    Return up to 20 rows with the largest absolute residual between
    actual and predicted performance.  Each dict contains:
    {row: int, column: str, actual: float, expected: float}
    """
    # Identify feature columns
    feature_cols = [c for c in fieldnames if not is_goal(c) and not is_x(c)]

    # Pre‑parse all goal values
    actuals = {}
    for idx, (rn, rec) in enumerate(rows):
        val = parse_value(rec[goal_col])
        if val is not None:
            actuals[idx] = float(val)

    flagged: List[Tuple[int, float, float]] = []

    # For each row with an actual value, compute predicted as mean of 3 nearest neighbours
    for idx, rn, rec in [(i, rn, rec) for i, (rn, rec) in enumerate(rows)]:
        if idx not in actuals:
            continue
        neigh = nearest_neighbors(idx, rows, feature_cols, k=3)
        # collect actuals of neighbors
        neigh_vals = [actuals[n] for n in neigh if n in actuals]
        if not neigh_vals:
            continue  # cannot predict
        pred = sum(neigh_vals) / len(neigh_vals)
        actual = actuals[idx]
        resid = actual - pred
        flagged.append((rn, actual, pred, abs(resid)))

    # Sort by largest absolute residual
    flagged.sort(key=lambda x: x[3], reverse=True)
    top = flagged[:20]

    out = []
    for rn, actual, pred, _ in top:
        out.append({
            "row": rn,
            "column": goal_col,
            "actual": round(actual, 6),
            "expected": round(pred, 6)
        })
    return out


# ----------------------------------------------------------------------
# APACHE-18 logic
# ----------------------------------------------------------------------
def apache_18(rows: List[Tuple[int, Dict[str, str]]],
              fieldnames: List[str]) -> Tuple[List[int], Dict[int, float]]:
    """
    Return a list of at most 20 flagged row numbers (most unusual input
    combinations) and a mapping of row number to unusualness score.
    """
    feature_cols = [c for c in fieldnames if not is_goal(c) and not is_x(c)]

    # Compute nearest distance for each row
    nearest_dists = []
    for idx, (_, rec_a) in enumerate(rows):
        # find nearest neighbour distance
        best = math.inf
        for jdx, (_, rec_b) in enumerate(rows):
            if jdx == idx:
                continue
            d = distance(rec_a, rec_b, feature_cols)
            if d < best:
                best = d
        nearest_dists.append(best)

    # Compute mean of finite nearest distances
    finite_dists = [d for d in nearest_dists if math.isfinite(d)]
    if not finite_dists:
        mean_dist = 0.0
    else:
        mean_dist = sum(finite_dists) / len(finite_dists)

    # Compute score: distance / mean_dist (larger means more unusual)
    scores: Dict[int, float] = {}
    for idx, d in enumerate(nearest_dists):
        if not math.isfinite(d) or mean_dist == 0:
            continue
        scores[idx] = d / mean_dist

    # Select top 20 rows by score
    top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:20]
    flagged_rows = [rows[i][0] for i in top_indices]
    scores_map = {rows[i][0]: round(scores[i], 6) for i in top_indices}
    return flagged_rows, scores_map


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Identify primary goal column (first that ends with '-')
    goal_col = next((c for c in fieldnames if c.endswith("-")), None)
    if goal_col is None:
        # fall back to first '+' column
        goal_col = next((c for c in fieldnames if c.endswith("+")), None)

    # ---------------- APACHE-17 ----------------
    print("== APACHE-17 ==")
    flagged17 = apache_17(rows, fieldnames, goal_col)
    if flagged17:
        print(f"Top {len(flagged17)} unexpected performance rows:")
        for f in flagged17:
            print(f"  Row {f['row']}: actual={f['actual']} expected={f['expected']}")
    else:
        print("No outlier performance rows found.")
    print("ANSWER APACHE-17", json.dumps({"flagged": flagged17}, separators=(",", ":")))

    # ---------------- APACHE-18 ----------------
    print("\n== APACHE-18 ==")
    flagged_rows, scores_map = apache_18(rows, fieldnames)
    if flagged_rows:
        print(f"Top {len(flagged_rows)} rows with unusual setting combinations:")
        for r in flagged_rows:
            print(f"  Row {r}: unusualness score={scores_map[r]}")
    else:
        print("No unusual input combinations found.")
    print("ANSWER APACHE-18",
          json.dumps({"flagged": flagged_rows, "scores": scores_map}, separators=(",", ":")))


if __name__ == "__main__":
    main()

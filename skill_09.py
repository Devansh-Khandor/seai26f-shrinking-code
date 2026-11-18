#!/usr/bin/env python3
"""
skill_09.py

Usage: python skill_09.py <csv path>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Any, List, Tuple, Optional

# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------
def parse_value(v: str) -> Any:
    """Parse a CSV cell to int/float/str or None for '?'."""
    if v == "?":
        return None
    for fn in (int, float):
        try:
            return fn(v)
        except ValueError:
            pass
    return v

def is_numeric(x: Any) -> bool:
    return isinstance(x, (int, float))

# ------------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------------
def load_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[parse_value(v) for v in row] for row in rdr]
    return header, rows

# ------------------------------------------------------------------
# Distance over input columns
# ------------------------------------------------------------------
def compute_input_indices(header: List[str]) -> List[int]:
    """Indices of columns that are neither X nor goal."""
    return [
        i for i, h in enumerate(header)
        if not h.endswith("X") and not h.endswith(("+", "-"))
    ]

def minmax_scaling(rows: List[List[Any]], idxs: List[int]) -> dict:
    """Return dict of {idx: (mn, mx)} for numeric columns."""
    scaling = {}
    for i in idxs:
        vals = [r[i] for r in rows if r[i] is not None and is_numeric(r[i])]
        if vals:
            mn, mx = min(vals), max(vals)
            scaling[i] = (mn, mx)
    return scaling

def norm_numeric(v: Any, mn: float, mx: float) -> float:
    if mx == mn:
        return 0.5
    return (v - mn) / (mx - mn)

def distance(row1: List[Any], row2: List[Any],
             idxs: List[int], scaling: dict) -> float:
    """Average per‑column distance, in [0,1]."""
    d = 0.0
    for i in idxs:
        v1, v2 = row1[i], row2[i]
        if v1 is None or v2 is None:
            d += 1.0
        elif is_numeric(v1) and is_numeric(v2) and i in scaling:
            mn, mx = scaling[i]
            n1, n2 = norm_numeric(v1, mn, mx), norm_numeric(v2, mn, mx)
            d += abs(n1 - n2)
        else:
            d += 0.0 if v1 == v2 else 1.0
    return d / len(idxs)

# ------------------------------------------------------------------
# NASA-17: flag unusual input combinations
# ------------------------------------------------------------------
def compute_unusualness(rows: List[List[Any]], idxs: List[int],
                        scaling: dict) -> List[float]:
    n = len(rows)
    scores = []
    for i in range(n):
        dsum = 0.0
        for j in range(n):
            if i == j:
                continue
            dsum += distance(rows[i], rows[j], idxs, scaling)
        scores.append(dsum / (n - 1))
    return scores

def nasa_17(rows: List[List[Any]], idxs: List[int], scaling: dict) -> Tuple[List[int], dict]:
    scores = compute_unusualness(rows, idxs, scaling)
    # top 10 rows (1‑based) with largest scores
    ranked = sorted(enumerate(scores, 1), key=lambda x: -x[1])[:10]
    flagged = [r for r, _ in ranked]
    score_dict = {str(r): round(s, 3) for r, s in ranked}
    return flagged, score_dict

# ------------------------------------------------------------------
# NASA-18: flag outliers on Effort- or Defects-
# ------------------------------------------------------------------
def find_nearest(row: List[Any], rows: List[List[Any]], idxs: List[int],
                 scaling: dict, k: int = 5) -> List[int]:
    dists = [(distance(row, r, idxs, scaling), idx) for idx, r in enumerate(rows) if r is not row]
    dists.sort(key=lambda x: x[0])
    return [idx for _, idx in dists[:k]]

def expected_from_neighbors(neighbors: List[int], rows: List[List[Any]],
                            col: int) -> Optional[float]:
    vals = [rows[i][col] for i in neighbors if rows[i][col] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def nasa_18(rows: List[List[Any]], header: List[str], idxs: List[int],
            scaling: dict) -> List[dict]:
    eff_col = header.index("Effort-")
    def_col = header.index("Defects-")
    out = []
    for i, row in enumerate(rows):
        neigh = find_nearest(row, rows, idxs, scaling, k=5)
        for col, name in [(eff_col, "Effort-"), (def_col, "Defects-")]:
            actual = row[col]
            if actual is None:
                continue
            expected = expected_from_neighbors(neigh, rows, col)
            if expected is None:
                continue
            diff = abs(actual - expected)
            out.append({
                "row": i + 1,
                "column": name,
                "actual": round(actual, 3) if isinstance(actual, float) else actual,
                "expected": round(expected, 3),
                "diff": diff,
            })
    # pick top 10 by diff
    top = sorted(out, key=lambda x: -x["diff"])[:10]
    # keep only required fields
    return [
        {"row": e["row"], "column": e["column"],
         "actual": e["actual"], "expected": e["expected"]}
        for e in top
    ]

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = load_csv(csv_path)
    input_idxs = compute_input_indices(header)
    scaling = minmax_scaling(rows, input_idxs)

    # NASA-17
    flagged_17, scores_17 = nasa_17(rows, input_idxs, scaling)
    print("== NASA-17 ==")
    print(f"Flagged rows: {flagged_17}")
    print("Unusualness scores:")
    for r, s in scores_17.items():
        print(f"  Row {r}: {s}")
    print(f"ANSWER NASA-17 {json.dumps({'flagged': flagged_17, 'scores': scores_17}, ensure_ascii=False)}")

    # NASA-18
    flagged_18 = nasa_18(rows, header, input_idxs, scaling)
    print("\n== NASA-18 ==")
    print("Flagged projects:")
    for e in flagged_18:
        print(f"  Row {e['row']} - {e['column']}: actual={e['actual']} expected={e['expected']}")
    print(f"ANSWER NASA-18 {json.dumps({'flagged': flagged_18}, ensure_ascii=False)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

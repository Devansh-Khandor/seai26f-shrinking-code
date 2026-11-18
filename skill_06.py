#!/usr/bin/env python3
"""
skill_06.py - Apache configuration recommendation and comparison

The script implements:

  * APACHE-11 : Recommends a configuration and explains it by showing
    its nearest neighbours in the input space and the performance of each.

  * APACHE-12 : Shows a pair of configurations with different
    performance assessments, identifies the differing settings, and
    lists the goal columns where the less‑recommended configuration
    actually performs better (trade‑offs).

Usage:
    python skill_06.py <csv_path>
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
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return rows and header list.  Preserve '?' as string."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if not line:
                continue
            row = {}
            for key, val in zip(header, line):
                val = val.strip()
                if val == '':
                    val = '?'
                # Try numeric conversion
                try:
                    if '.' in val:
                        row[key] = float(val)
                    else:
                        row[key] = int(val)
                except Exception:
                    row[key] = val
            rows.append(row)
    return rows, header

# ----------------------------------------------------------------------
# Column helpers
# ----------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')

def is_input(col: str) -> bool:
    return not is_goal(col) and not col.endswith('X')

# ----------------------------------------------------------------------
# Goal statistics
# ----------------------------------------------------------------------
def goal_minmax(rows: List[Dict[str, Any]], goals: List[str]) -> Dict[str, Tuple[float, float]]:
    """Return {goal: (min, max)} over all rows with known values."""
    stats: Dict[str, Tuple[float, float]] = {}
    for g in goals:
        vals = [row[g] for row in rows if isinstance(row[g], (int, float))]
        if vals:
            stats[g] = (min(vals), max(vals))
    return stats

# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def score_row(row: Dict[str, Any], goals: List[str], gm: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Return a combined score (higher is better).  Rows with missing
    goal values return None."""
    s = 0.0
    for g in goals:
        v = row[g]
        if not isinstance(v, (int, float)):
            return None
        lo, hi = gm[g]
        if hi == lo:   # avoid division by zero
            norm = 1.0
        else:
            if g.endswith('-'):      # minimize
                norm = (hi - v) / (hi - lo)
            else:                    # maximize
                norm = (v - lo) / (hi - lo)
        s += norm
    return s

def best_row(rows: List[Dict[str, Any]], goals: List[str], gm: Dict[str, Tuple[float, float]]) -> Tuple[int, Dict[str, Any], float]:
    """Return (index, row, score) of the best scored row."""
    best_idx = -1
    best_val = -math.inf
    best_r = None
    for i, r in enumerate(rows, start=1):
        sc = score_row(r, goals, gm)
        if sc is None:
            continue
        if sc > best_val:
            best_val = sc
            best_idx = i
            best_r = r
    return best_idx, best_r, best_val  # best_r guaranteed not None

# ----------------------------------------------------------------------
# Distance
# ----------------------------------------------------------------------
def hamming_distance(r1: Dict[str, Any], r2: Dict[str, Any], inputs: List[str]) -> int:
    """Number of differing input settings."""
    diff = 0
    for c in inputs:
        v1 = r1[c]
        v2 = r2[c]
        if v1 == '?' and v2 == '?':
            continue
        if v1 != v2:
            diff += 1
    return diff

def nearest_neighbors(best_idx: int, rows: List[Dict[str, Any]], inputs: List[str],
                      goals: List[str], gm: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
    """Return at least 3 nearest neighbours (by Hamming distance, then by score)."""
    best_row_data = rows[best_idx - 1]
    neigh = []
    for i, r in enumerate(rows, start=1):
        if i == best_idx:
            continue
        dist = hamming_distance(best_row_data, r, inputs)
        sc = score_row(r, goals, gm)
        if sc is None:
            continue
        neigh.append((dist, -sc, i, r))  # negative sc for descending
    neigh.sort(key=lambda x: (x[0], x[1]))
    result = []
    for _, _, idx, r in neigh[:3]:
        result.append({"row": idx, "values": r})
    return result

# ----------------------------------------------------------------------
# Best pair difference
# ----------------------------------------------------------------------
def best_pair(rows: List[Dict[str, Any]], goals: List[str], gm: Dict[str, Tuple[float, float]]) -> Tuple[int, int, str, float, float]:
    """Return (idx_a, idx_b, preferred, score_a, score_b) with max score difference."""
    best = None
    best_diff = -1.0
    for i, r_i in enumerate(rows, start=1):
        sc_i = score_row(r_i, goals, gm)
        if sc_i is None:
            continue
        for j in range(i+1, len(rows)+1):
            r_j = rows[j-1]
            sc_j = score_row(r_j, goals, gm)
            if sc_j is None:
                continue
            diff = abs(sc_i - sc_j)
            if diff > best_diff:
                best_diff = diff
                best = (i, j, sc_i, sc_j)
    if best is None:
        raise RuntimeError("No valid pair found.")
    i, j, sc_i, sc_j = best
    preferred = "a" if sc_i >= sc_j else "b"
    return i, j, preferred, sc_i, sc_j

def diff_settings(i: int, j: int, rows: List[Dict[str, Any]], inputs: List[str]) -> List[Dict[str, Any]]:
    """Return list of differing settings between rows i and j."""
    r_i = rows[i-1]
    r_j = rows[j-1]
    diffs = []
    for c in inputs:
        v1 = r_i[c]
        v2 = r_j[c]
        if v1 != v2:
            diffs.append({"feature": c, "a": v1, "b": v2})
    return diffs

def tradeoffs(i: int, j: int, rows: List[Dict[str, Any]],
              goals: List[str], preferred: str) -> List[str]:
    """Return list of goal columns where the non‑preferred row is better."""
    # Map which row is non-preferred
    non_pref = j if preferred == 'a' else i
    pref = i if preferred == 'a' else j
    r_pref = rows[pref-1]
    r_non = rows[non_pref-1]
    trd = []
    for g in goals:
        v_pref = r_pref[g]
        v_non = r_non[g]
        if not isinstance(v_pref, (int, float)) or not isinstance(v_non, (int, float)):
            continue
        if g.endswith('-'):  # minimize
            if v_non < v_pref:
                trd.append(g)
        else:  # maximize
            if v_non > v_pref:
                trd.append(g)
    return trd

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, header = read_csv(path)

    goals = [c for c in header if is_goal(c)]
    inputs = [c for c in header if is_input(c)]

    if not goals or not inputs:
        print("Insufficient columns.")
        sys.exit(0)

    gm = goal_minmax(rows, goals)

    # APACHE-11
    idx_best, best_r, best_sc = best_row(rows, goals, gm)
    neighs = nearest_neighbors(idx_best, rows, inputs, goals, gm)

    print("== APACHE-11 ==")
    print(f"Recommended row: {idx_best}  score={best_sc:.4f}")
    for n in neighs:
        print(f"Neighbour {n['row']}  distance ignored  score={score_row(n['values'], goals, gm):.4f}")

    ans11 = {"recommended": idx_best, "neighbors": neighs}
    print("ANSWER APACHE-11", json.dumps(ans11, separators=(",", ":")))

    # APACHE-12
    i_a, i_b, pref, sc_a, sc_b = best_pair(rows, goals, gm)
    diffs = diff_settings(i_a, i_b, rows, inputs)
    trd = tradeoffs(i_a, i_b, rows, goals, pref)

    print("\n== APACHE-12 ==")
    print(f"Pair: {i_a} vs {i_b}  preferred: {pref.upper()}")
    print(f"Scores: a={sc_a:.4f}  b={sc_b:.4f}")
    print(f"Differences: {len(diffs)} features")
    for d in diffs:
        print(f"  {d['feature']}: {d['a']} vs {d['b']}")

    ans12 = {
        "a": i_a,
        "b": i_b,
        "preferred": pref,
        "scores": {"a": sc_a, "b": sc_b},
        "differences": diffs,
        "tradeoffs": trd
    }
    print("ANSWER APACHE-12", json.dumps(ans12, separators=(",", ":")))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skill 08: Targeted configuration adjustments.

APACHE-15 – For a poorly performing configuration, recommend a small change to its
adjustable settings that could lead to a lower predicted performance value,
using historically observed valid configurations as alternatives.

APACHE-16 – Find the smallest number of configuration-setting changes that could
bring a poorly performing configuration closer to a specified performance
target, while avoiding unsupported or invalid combinations.
"""

import json
import sys
import math
from typing import List, Dict, Tuple

import ezr
from ezr import same, xdist, ranker, leaf, leafs, tree, o, Tbl, Col, Cols, addRow, addCols, clone

# ----------------------------------------------------------------------
# Helpers ---------------------------------------------------------------
# ----------------------------------------------------------------------
def load_data(csv_path: str) -> Tuple[ezr.Tbl, List[ezr.ROW]]:
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)
    return tbl, rows


def goal_col(tbl: ezr.Tbl) -> ezr.COL:
    """Return the first goal column (ends with '-' or '+')."""
    for col in tbl.cols.values():
        if col.txt.endswith("-") or col.txt.endswith("+"):
            return col
    raise RuntimeError("No goal column found.")


def adjustable_cols(tbl: ezr.Tbl) -> List[ezr.COL]:
    """Return columns that are adjustable (not goal, not X)."""
    return [c for c in tbl.cols.values()
            if not c.txt.endswith(("X", "+", "-"))]


def diff_settings(row_a: ezr.ROW, row_b: ezr.ROW, tbl: ezr.Tbl) -> List[str]:
    """Return list of setting names that differ between two rows
    (ignoring missing values)."""
    diffs = []
    for col in adjustable_cols(tbl):
        va = row_a[col.at]
        vb = row_b[col.at]
        if va == "?" or vb == "?":
            # if either is missing we skip this column from diff
            continue
        if va != vb:
            diffs.append(col.txt)
    return diffs


def performance_value(row: ezr.ROW, gcol: ezr.COL) -> float:
    val = row[gcol.at]
    return float(val) if val != "?" else math.nan


def find_poor_row(rows: List[ezr.ROW], gcol: ezr.COL) -> Tuple[int, ezr.ROW]:
    """Return (index, row) of the worst performing row (max value for a minimizing goal)."""
    best_val = None
    best_idx = None
    for idx, row in enumerate(rows, start=1):
        val = performance_value(row, gcol)
        if math.isnan(val):
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_idx = idx
    return best_idx, rows[best_idx - 1]


def find_best_row(rows: List[ezr.ROW], gcol: ezr.COL) -> Tuple[int, ezr.ROW]:
    """Return (index, row) of the best performing row (min value for minimizing goal)."""
    best_val = None
    best_idx = None
    for idx, row in enumerate(rows, start=1):
        val = performance_value(row, gcol)
        if math.isnan(val):
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_idx = idx
    return best_idx, rows[best_idx - 1]


# ----------------------------------------------------------------------
# APACHE-15 -------------------------------------------------------------
# ----------------------------------------------------------------------
def apache_15(tbl: ezr.Tbl, rows: List[ezr.ROW]) -> None:
    print("== APACHE-15 ==")

    gcol = goal_col(tbl)

    # Poor row (worst)
    poor_idx, poor_row = find_poor_row(rows, gcol)
    poor_perf = performance_value(poor_row, gcol)

    # Find alternatives that differ in 1 or 2 adjustable settings
    candidates = []
    for idx, alt_row in enumerate(rows, start=1):
        if idx == poor_idx:
            continue
        alt_perf = performance_value(alt_row, gcol)
        if math.isnan(alt_perf):
            continue
        if alt_perf >= poor_perf:
            continue  # we want a better alternative

        diffs = diff_settings(poor_row, alt_row, tbl)
        if 1 <= len(diffs) <= 2:
            candidates.append((len(diffs), alt_perf, idx, alt_row, diffs))

    # Pick the best candidate (fewest changes, then best performance)
    if not candidates:
        # fallback: choose the overall best row
        best_idx, best_row = find_best_row(rows, gcol)
        alt_row = best_row
        alt_idx = best_idx
        diffs = diff_settings(poor_row, alt_row, tbl)
        alt_perf = performance_value(alt_row, gcol)
    else:
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, alt_perf, alt_idx, alt_row, diffs = candidates[0]

    changes = {k: [poor_row[tbl.cols[k].at], alt_row[tbl.cols[k].at]] for k in diffs}

    print(f"Poor row: {poor_idx} (Performance-={poor_perf:.2f})")
    print(f"Alternative row: {alt_idx} (Performance-={alt_perf:.2f})")
    print("Recommended changes:")
    for k, (old, new) in changes.items():
        print(f"  {k}: {old} -> {new}")

    ans = {
        "row": poor_idx,
        "alternative": alt_idx,
        "changes": changes,
        "performance": {"before": poor_perf, "after": alt_perf},
    }
    print("ANSWER APACHE-15", json.dumps(ans, separators=(",", ":")))


# ----------------------------------------------------------------------
# APACHE-16 -------------------------------------------------------------
# ----------------------------------------------------------------------
def apache_16(tbl: ezr.Tbl, rows: List[ezr.ROW]) -> None:
    print("== APACHE-16 ==")

    gcol = goal_col(tbl)

    # Poor row (worst)
    poor_idx, poor_row = find_poor_row(rows, gcol)
    poor_perf = performance_value(poor_row, gcol)

    # Target: best achievable performance (minimum)
    best_idx, best_row = find_best_row(rows, gcol)
    target_perf = performance_value(best_row, gcol)

    # Find alternatives better than poor and closer to target (i.e., lower)
    candidates = []
    for idx, alt_row in enumerate(rows, start=1):
        if idx == poor_idx:
            continue
        alt_perf = performance_value(alt_row, gcol)
        if math.isnan(alt_perf):
            continue
        if alt_perf >= poor_perf:
            continue  # must be better
        # distance to target
        dist_alt = abs(alt_perf - target_perf)
        dist_poor = abs(poor_perf - target_perf)
        if dist_alt >= dist_poor:
            continue  # not closer
        diffs = diff_settings(poor_row, alt_row, tbl)
        if not diffs:
            continue  # identical row (shouldn't happen)
        candidates.append((len(diffs), alt_perf, idx, alt_row, diffs))

    if not candidates:
        # fallback: best row
        alt_idx, alt_row = best_idx, best_row
        alt_perf = target_perf
        diffs = diff_settings(poor_row, alt_row, tbl)
    else:
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, alt_perf, alt_idx, alt_row, diffs = candidates[0]

    changed = diffs

    print(f"Poor row: {poor_idx} (Performance-={poor_perf:.2f})")
    print(f"Target performance: {target_perf:.2f}")
    print(f"Alternative row: {alt_idx} (Performance-={alt_perf:.2f})")
    print("Changes needed:")
    for k in changed:
        print(f"  {k}")

    ans = {
        "row": poor_idx,
        "target": target_perf,
        "alternative": alt_idx,
        "changed": changed,
    }
    print("ANSWER APACHE-16", json.dumps(ans, separators=(",", ":")))


# ----------------------------------------------------------------------
# Main -------------------------------------------------------------
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    tbl, rows = load_data(csv_path)
    apache_15(tbl, rows)
    apache_16(tbl, rows)


# ----------------------------------------------------------------------
# Entry point ----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

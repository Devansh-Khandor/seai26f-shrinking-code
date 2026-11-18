#!/usr/bin/env python3
"""
skill_08.py: Analysis of poorly rated vehicles and alternative recommendations.
Usage: python skill_08.py <csv_path>
"""

import json
import sys
import ezr
from typing import List, Dict, Any, Tuple


# ------------------------------------------------------------------
# Basic helpers (copy of skill_01's read/build)
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV and return header and rows as atoms."""
    with open(path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
        rows: List[List[Any]] = []
        for line in f:
            if line.strip():
                rows.append([ezr.atom(s) for s in line.split(",")])
    return header, rows


def build_table(header: List[str], rows: List[List[Any]]) -> ezr.TBL:
    """Create an ezr table from header and rows."""
    return ezr.Tbl([header] + rows)


# ------------------------------------------------------------------
# Utility: overall score for a row
def compute_goal_min_max(tbl: ezr.TBL, goal_cols: List[ezr.COL]) -> Dict[int, Tuple[float, float]]:
    """Return min/max for each goal column over all rows."""
    mins_maxs: Dict[int, Tuple[float, float]] = {}
    for col in goal_cols:
        vals = [r[col.at] for r in tbl.rows if r[col.at] != "?"]
        if vals:
            mins_maxs[col.at] = (min(vals), max(vals))
        else:
            mins_maxs[col.at] = (0.0, 0.0)
    return mins_maxs


def row_score(tbl: ezr.TBL, row: List[Any], goal_cols: List[ezr.COL], mins_maxs: Dict[int, Tuple[float, float]]) -> float:
    """Compute overall score of a row based on all goal columns."""
    score = 0.0
    for col in goal_cols:
        val = row[col.at]
        if val == "?":
            continue
        mn, mx = mins_maxs[col.at]
        if mx == mn:
            comp = 1.0
        else:
            if col.goal == 1:  # maximize
                comp = (val - mn) / (mx - mn)
            else:  # minimize
                comp = (mx - val) / (mx - mn)
        score += comp
    return score


# ------------------------------------------------------------------
# CAR-15 helpers
def compute_median(values: List[float]) -> float:
    """Median of a list of numbers."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def find_poor_row(tbl: ezr.TBL, goal_cols: List[ezr.COL], mins_maxs: Dict[int, Tuple[float, float]]) -> int:
    """Return 1‑based index of the worst row (lowest overall score)."""
    best = None
    best_idx = None
    for i, r in enumerate(tbl.rows):
        score = row_score(tbl, r, goal_cols, mins_maxs)
        if best is None or score < best:
            best = score
            best_idx = i + 1
    return best_idx  # 1‑based


def find_alternatives(
    tbl: ezr.TBL,
    poor_row_idx: int,
    history_rows: List[List[Any]],
    goal_cols: List[ezr.COL],
    mins_maxs: Dict[int, Tuple[float, float]],
    input_cols: List[ezr.COL],
) -> List[Dict[str, Any]]:
    """Return up to 3 alternative rows from history that are
    closer in input space and have better overall score."""
    poor_row = tbl.rows[poor_row_idx - 1]
    poor_score = row_score(tbl, poor_row, goal_cols, mins_maxs)

    alternatives = []
    for alt in history_rows:
        alt_idx = tbl.rows.index(alt) + 1  # 1‑based
        alt_score = row_score(tbl, alt, goal_cols, mins_maxs)
        if alt_score <= poor_score:
            continue  # only better
        # distance on input columns
        dist = 0.0
        for col in input_cols:
            v1, v2 = poor_row[col.at], alt[col.at]
            if v1 == "?" or v2 == "?" or v1 != v2:
                dist += 1
        alternatives.append((dist, alt_idx, alt))

    # sort by distance, then by score
    alternatives.sort(key=lambda x: (x[0], -row_score(tbl, x[2], goal_cols, mins_maxs)))
    result = []
    for _, idx, alt in alternatives[:3]:
        changes = {}
        for col in input_cols:
            v1, v2 = poor_row[col.at], alt[col.at]
            if v1 != v2:
                changes[col.txt] = [v1, v2]
        result.append({"row": idx, "changes": changes})
    return result


# ------------------------------------------------------------------
# CAR-16 helpers
def compute_thresholds(goal_cols: List[ezr.COL], tbl: ezr.TBL) -> Dict[str, float]:
    """Median threshold for each goal column."""
    thresholds: Dict[str, float] = {}
    for col in goal_cols:
        vals = [r[col.at] for r in tbl.rows if r[col.at] != "?"]
        if vals:
            thresholds[col.txt] = compute_median(vals)
    return thresholds


def met_targets_count(row: List[Any], thresholds: Dict[str, float], goal_cols: List[ezr.COL]) -> int:
    """Number of goal columns where the row meets the threshold."""
    count = 0
    for col in goal_cols:
        val = row[col.at]
        if val == "?":
            continue
        thresh = thresholds[col.txt]
        if col.goal == 1 and val >= thresh:
            count += 1
        if col.goal == -1 and val <= thresh:
            count += 1
    return count


def find_target_row(
    tbl: ezr.TBL,
    thresholds: Dict[str, float],
    goal_cols: List[ezr.COL],
) -> int:
    """Row that meets the fewest targets (1‑based)."""
    best_cnt = None
    best_idx = None
    for i, r in enumerate(tbl.rows):
        cnt = met_targets_count(r, thresholds, goal_cols)
        if best_cnt is None or cnt < best_cnt:
            best_cnt = cnt
            best_idx = i + 1
    return best_idx


def find_alternative_for_target(
    tbl: ezr.TBL,
    target_idx: int,
    thresholds: Dict[str, float],
    goal_cols: List[ezr.COL],
    input_cols: List[ezr.COL],
) -> Tuple[int, List[str]]:
    """Best alternative row (better targets, minimal input changes)."""
    target_row = tbl.rows[target_idx - 1]
    target_cnt = met_targets_count(target_row, thresholds, goal_cols)

    best_alt = None
    best_diff = None
    best_cnt = None

    for i, r in enumerate(tbl.rows):
        if i + 1 == target_idx:
            continue
        cnt = met_targets_count(r, thresholds, goal_cols)
        if cnt <= target_cnt:
            continue  # must be better
        # number of differing input columns
        diff = 0
        for col in input_cols:
            if target_row[col.at] != r[col.at]:
                diff += 1
        if best_alt is None or diff < best_diff or (diff == best_diff and cnt > best_cnt):
            best_alt = i + 1
            best_diff = diff
            best_cnt = cnt

    # compute list of changed columns
    changed = []
    if best_alt is not None:
        alt_row = tbl.rows[best_alt - 1]
        for col in input_cols:
            if target_row[col.at] != alt_row[col.at]:
                changed.append(col.txt)
    return best_alt, changed


# ------------------------------------------------------------------
# Main driver
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]

    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # Identify goal and input columns
    goal_cols = [col for col in tbl.cols.values() if col.goal != 0]
    input_cols = [col for col in tbl.cols.values() if col.goal == 0]

    mins_maxs = compute_goal_min_max(tbl, goal_cols)

    # ------------------ CAR-15 ------------------------------------
    poor_row_idx = find_poor_row(tbl, goal_cols, mins_maxs)

    # Determine arrival start (last 20% rows)
    total_rows = len(tbl.rows)
    arrivals_count = round(0.2 * total_rows)
    if arrivals_count == 0:
        arrivals_count = 1
    arrival_start = total_rows - arrivals_count
    history_rows = tbl.rows[:arrival_start]

    alternatives = find_alternatives(tbl, poor_row_idx, history_rows, goal_cols, mins_maxs, input_cols)

    print("== CAR-15 ==")
    print(f"Poorly rated vehicle row: {poor_row_idx}")
    print("Alternatives from historical inventory:")
    for a in alternatives:
        print(f"  Row {a['row']}, changes: {a['changes']}")
    print(f"ANSWER CAR-15 {json.dumps({'row': poor_row_idx, 'alternatives': alternatives}, separators=(',', ':'))}")

    # ------------------ CAR-16 ------------------------------------
    thresholds = compute_thresholds(goal_cols, tbl)
    target_row_idx = find_target_row(tbl, thresholds, goal_cols)

    alt_idx, changed_cols = find_alternative_for_target(tbl, target_row_idx, thresholds, goal_cols, input_cols)

    print("\n== CAR-16 ==")
    print(f"Row missing some targets: {target_row_idx}")
    print(f"Thresholds: {thresholds}")
    print(f"Alternative row: {alt_idx}")
    print(f"Changed input columns: {changed_cols}")
    print(f"ANSWER CAR-16 {json.dumps({'row': target_row_idx, 'targets': thresholds, 'alternative': alt_idx, 'changed': changed_cols}, separators=(',', ':'))}")


if __name__ == "__main__":
    main()

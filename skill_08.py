#!/usr/bin/env python3
"""
skill_08.py

APACHE-15 – For a poorly performing configuration, recommend a small change to its
adjustable settings that could lead to a lower predicted performance value, using
historically observed valid configurations as alternatives.

APACHE-16 – Find the smallest number of configuration‑setting changes that could
bring a poorly performing configuration closer to a specified performance target,
while avoiding unsupported or invalid combinations.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Tuple[int, Dict[str, str]]], List[str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))
    return rows, fieldnames


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_x(col: str) -> bool:
    return col.endswith("X")


def to_float(v: str) -> Optional[float]:
    if v is None or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def stats(values: List[float]) -> Tuple[float, float]:
    """Return mean and population standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)


def count_diffs(r1: Dict[str, str], r2: Dict[str, str], settings: List[str]) -> int:
    """Count differing settings where neither value is missing."""
    diff = 0
    for s in settings:
        v1 = r1.get(s)
        v2 = r2.get(s)
        if v1 in ("?", None) or v2 in ("?", None):
            continue
        if v1 != v2:
            diff += 1
    return diff


def diff_map(r1: Dict[str, str], r2: Dict[str, str], settings: List[str]) -> Dict[str, List[Optional[str]]]:
    d = {}
    for s in settings:
        v1 = r1.get(s)
        v2 = r2.get(s)
        if v1 in ("?", None) or v2 in ("?", None):
            continue
        if v1 != v2:
            d[s] = [v1, v2]
    return d


def pick_poor_row(rows: List[Tuple[int, Dict[str, str]]], goal_col: str) -> Tuple[int, Dict[str, str]]:
    """Return first row that is above mean+2*sd; if none, return worst."""
    values = [to_float(rec[goal_col]) for _, rec in rows if to_float(rec[goal_col]) is not None]
    mu, sd = stats(values)
    threshold = mu + 2 * sd if sd > 0 else mu
    # Search for above threshold
    for rn, rec in rows:
        v = to_float(rec[goal_col])
        if v is not None and v > threshold:
            return rn, rec
    # else worst
    worst = max((to_float(rec[goal_col]), rn, rec) for rn, rec in rows if to_float(rec[goal_col]) is not None)
    return worst[1], worst[2]


def find_alternative(rows: List[Tuple[int, Dict[str, str]]],
                     target_rec: Dict[str, str],
                     settings: List[str],
                     goal_col: str,
                     max_diff: int = 2) -> Optional[Tuple[int, Dict[str, str]]]:
    """Return observed row with <=max_diff settings differing and better goal."""
    target_perf = to_float(target_rec[goal_col])
    best = None
    best_perf = None
    for rn, rec in rows:
        perf = to_float(rec[goal_col])
        if perf is None or perf >= target_perf:
            continue
        diff = count_diffs(target_rec, rec, settings)
        if diff > max_diff:
            continue
        if best is None or perf < best_perf:
            best = (rn, rec)
            best_perf = perf
    return best


def find_closest(rows: List[Tuple[int, Dict[str, str]]],
                 target_rec: Dict[str, str],
                 settings: List[str],
                 goal_col: str) -> Optional[Tuple[int, Dict[str, str], int]]:
    """Return observed row better than target_rec with minimal setting differences."""
    target_perf = to_float(target_rec[goal_col])
    best = None
    best_diff = None
    best_perf = None
    for rn, rec in rows:
        perf = to_float(rec[goal_col])
        if perf is None or perf >= target_perf:
            continue
        diff = count_diffs(target_rec, rec, settings)
        if best is None or diff < best_diff or (diff == best_diff and perf < best_perf):
            best = (rn, rec)
            best_diff = diff
            best_perf = perf
    return best, best_diff


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # identify goal and setting columns
    goal_col = next((c for c in fieldnames if c.endswith("-")), None)
    if not goal_col:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    settings = [c for c in fieldnames if not is_goal(c) and not is_x(c)]

    # -------------------------------------------------
    # APACHE-15
    # -------------------------------------------------
    print("== APACHE-15 ==")
    poor_row_num, poor_row = pick_poor_row(rows, goal_col)
    alt = find_alternative(rows, poor_row, settings, goal_col, max_diff=2)
    if alt:
        alt_num, alt_row = alt
        changes = diff_map(poor_row, alt_row, settings)
        before = to_float(poor_row[goal_col])
        after = to_float(alt_row[goal_col])
    else:
        alt_num, alt_row, changes, before, after = None, None, {}, None, None

    # Results
    print(f"Poor row: {poor_row_num} with Performance- {before}")
    if alt_num is not None:
        print(f"Alternative row: {alt_num} with Performance- {after}")
        print("Changes:")
        for s, vals in changes.items():
            print(f"  {s}: {vals[0]} -> {vals[1]}")
    else:
        print("No suitable alternative found.")

    # Answer JSON
    answer_15 = {
        "row": poor_row_num,
        "alternative": alt_num,
        "changes": changes,
        "performance": {"before": before, "after": after},
    }
    print("ANSWER APACHE-15", json.dumps(answer_15, separators=(",", ":")))

    # -------------------------------------------------
    # APACHE-16
    # -------------------------------------------------
    print("\n== APACHE-16 ==")
    # Target is best (lowest) Performance-
    best_perf_val = min(
        v for _, r in rows if (v := to_float(r[goal_col])) is not None
    )
    target = best_perf_val

    best_alt, diff_cnt = find_closest(rows, poor_row, settings, goal_col) or (None, None)
    if best_alt:
        alt_num, alt_row = best_alt
        changed = [s for s in settings if poor_row.get(s) != alt_row.get(s)
                   and poor_row.get(s) not in ("?", None)
                   and alt_row.get(s) not in ("?", None)]
    else:
        alt_num, changed = None, []

    print(f"Target performance: {target}")
    if alt_num is not None:
        print(f"Alternative row: {alt_num}")
        print(f"Changed settings: {changed}")
    else:
        print("No better alternative found.")

    # Answer JSON
    answer_16 = {
        "row": poor_row_num,
        "target": target,
        "alternative": alt_num,
        "changed": changed,
    }
    print("ANSWER APACHE-16", json.dumps(answer_16, separators=(",", ":")))


if __name__ == "__main__":
    main()

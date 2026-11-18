#!/usr/bin/env python3
"""
skill_05.py

APACHE-09 – When a configuration performs poorly, identify which of its
enabled or disabled settings are most strongly associated with unfavorable
performance in our historical measurements.

APACHE-10 – Identify combinations of server settings that frequently
appear among poorly performing configurations and show how they differ
from better‑performing combinations.
"""

import csv
import json
import math
import sys
from collections import defaultdict, Counter
from itertools import combinations

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def read_csv(path: str):
    """Return list of (row_no, rec) and list of fieldnames."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))
    return rows, fieldnames

def is_goal(col: str) -> bool:
    return col.endswith("-") or col.endswith("+")

def is_x(col: str) -> bool:
    return col.endswith("X")

def to_float(v: str):
    if v in ("", "?"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def std(xs, mu):
    if len(xs) < 2:
        return 0.0
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))

# ----------------------------------------------------------------------
# APACHE-09 helpers
# ----------------------------------------------------------------------
def poor_rows(rows, goal_col):
    """Return row numbers that are unusually poor for goal_col."""
    vals = [to_float(r[goal_col]) for _, r in rows]
    vals = [v for v in vals if v is not None]
    mu = mean(vals)
    sigma = std(vals, mu)
    threshold = mu + 2 * sigma
    poor = [rn for rn, rec in rows if to_float(rec[goal_col]) is not None and
            to_float(rec[goal_col]) >= threshold]
    # Ensure at least 5 rows – if not, take the worst 5
    if len(poor) < 5:
        sorted_rows = sorted(rows, key=lambda x: to_float(x[1][goal_col]) or float("inf"))
        poor = [rn for rn, _ in sorted_rows[-5:]]
    return poor

def binary_columns(rows, fieldnames):
    """Return list of columns whose values are only 0/1 or '?'. """
    cols = []
    for col in fieldnames:
        if is_goal(col) or is_x(col):
            continue
        ok = True
        for _, rec in rows:
            v = rec[col]
            if v == "?" or v in ("0", "1"):
                continue
            ok = False
            break
        if ok:
            cols.append(col)
    return cols

def column_stats(rows, col, good_rows_set, bad_rows_set):
    """Return proportion of rows with value '1' in each set."""
    bad_ones = bad_present = 0
    good_ones = good_present = 0
    for rn, rec in rows:
        v = rec[col]
        if v not in ("0", "1"):
            continue
        if rn in bad_rows_set:
            bad_present += 1
            if v == "1":
                bad_ones += 1
        else:
            good_present += 1
            if v == "1":
                good_ones += 1
    p_bad = bad_ones / bad_present if bad_present else 0.0
    p_good = good_ones / good_present if good_present else 0.0
    return p_bad, p_good

def ranking(columns, rows, good_rows_set, bad_rows_set):
    """Return list of columns sorted by diff (bad - good)."""
    diffs = []
    for col in columns:
        p_bad, p_good = column_stats(rows, col, good_rows_set, bad_rows_set)
        diffs.append((p_bad - p_good, col))
    diffs.sort(key=lambda x: (-x[0], x[1]))
    return [col for _, col in diffs]

# ----------------------------------------------------------------------
# APACHE-10 helpers
# ----------------------------------------------------------------------
def good_rows(rows, goal_col):
    """Return row numbers that are best (lowest) for goal_col."""
    vals = [(rn, to_float(rec[goal_col])) for rn, rec in rows if to_float(rec[goal_col]) is not None]
    vals.sort(key=lambda x: x[1])  # ascending
    # take at least 5
    n = max(5, len(vals) // 20)  # 5% or 5
    return [rn for rn, _ in vals[:n]]

def combo_shares(rows, bad_set, good_set, cols, combo):
    """Return bad_share, good_share for a particular combo of values."""
    bad_match = 0
    good_match = 0
    for rn, rec in rows:
        if rn in bad_set:
            if all(rec[c] == str(v) for c, v in combo.items()):
                bad_match += 1
        elif rn in good_set:
            if all(rec[c] == str(v) for c, v in combo.items()):
                good_match += 1
    bad_share = bad_match / len(bad_set) if bad_set else 0.0
    good_share = good_match / len(good_set) if good_set else 0.0
    return bad_share, good_share

def combinations_of_bad(rows, bad_set, good_set, cols, max_combos=10):
    """Generate best combos of two settings that appear often in bad rows."""
    combos = []
    for col1, col2 in combinations(cols, 2):
        # For each pair, consider all value combinations present in bad rows
        value_sets = set()
        for rn, rec in rows:
            if rn in bad_set:
                v1 = rec[col1]
                v2 = rec[col2]
                if v1 in ("0", "1") and v2 in ("0", "1"):
                    value_sets.add((v1, v2))
        for v1, v2 in value_sets:
            combo = {col1: int(v1), col2: int(v2)}
            bad_share, good_share = combo_shares(rows, bad_set, good_set, [col1, col2], combo)
            if bad_share > good_share:
                combos.append((bad_share, good_share, combo))
    combos.sort(key=lambda x: (-x[0], x[1]))
    return combos[:max_combos]

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    rows, fieldnames = read_csv(sys.argv[1])

    # Identify goal column (choose first that ends with '-')
    goal_cols = [c for c in fieldnames if c.endswith("-")]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)
    goal_col = goal_cols[0]

    # --- APACHE-09 ---
    poor = poor_rows(rows, goal_col)
    binary_cols = binary_columns(rows, fieldnames)
    good_set = set([rn for rn, _ in rows if rn not in poor])
    bad_set = set(poor)

    # For ranking we need good rows set, but use all non-bad rows as good for association
    ranked_cols = ranking(binary_cols, rows, good_set, bad_set)

    print("== APACHE-09 ==")
    print(f"Poor rows (≥{len(poor)}): {poor[:5]} ...")
    print(f"Top 3 associated settings: {ranked_cols[:3]}")
    print("ANSWER APACHE-09", json.dumps({
        "poor": poor,
        "ranking": ranked_cols
    }, separators=(",", ":")))

    # --- APACHE-10 ---
    bad_rows_list = poor
    good_rows_list = good_rows(rows, goal_col)

    good_set2 = set(good_rows_list)
    bad_set2 = set(bad_rows_list)

    combos = combinations_of_bad(rows, bad_set2, good_set2, binary_cols, max_combos=5)
    combos_out = []
    for bad_share, good_share, combo in combos:
        combos_out.append({
            "settings": combo,
            "bad_share": round(bad_share, 6),
            "good_share": round(good_share, 6)
        })

    print("\n== APACHE-10 ==")
    print(f"Bad rows: {bad_rows_list[:5]} ...")
    print(f"Good rows: {good_rows_list[:5]} ...")
    print(f"Top combo: {combos_out[0] if combos_out else None}")
    print("ANSWER APACHE-10", json.dumps({
        "bad": bad_rows_list,
        "good": good_rows_list,
        "combinations": combos_out
    }, separators=(",", ":")))

if __name__ == "__main__":
    main()

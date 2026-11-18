#!/usr/bin/env python3
"""
Skill 08 – Apache performance analytics (APACHE-15 & APACHE-16)

Usage:
    python skill_08.py <csv_path>
"""

import json
import sys
import ezr


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def goal_column(tbl: ezr.TBL) -> ezr.COL:
    """
    Return the first column whose name ends with '+' or '-'.
    If none, raise RuntimeError.
    """
    for c in tbl.cols.values():
        if c.txt.endswith(('+', '-')):
            return c
    raise RuntimeError("No goal column found.")


def setting_columns(tbl: ezr.TBL) -> list[ezr.COL]:
    """
    Return columns that are neither goals nor X columns.
    """
    return [
        c for c in tbl.cols.values()
        if not c.txt.endswith(('+', '-', 'X'))
    ]


def row_index(tbl: ezr.TBL, row: ezr.ROW) -> int:
    """Return 1‑based index of row in tbl."""
    return tbl.rows.index(row) + 1


def row_performance(row: ezr.ROW, goal_at: int) -> float:
    """Return numeric performance of a row (goal value)."""
    val = row[goal_at]
    if val == "?":
        raise ValueError("Missing goal value.")
    return float(val)


def all_settings_filled(row: ezr.ROW, settings: list[ezr.COL]) -> bool:
    """True if none of the setting columns contain '?'."""
    return all(row[c.at] != "?" for c in settings)


def diff_settings(row_a: ezr.ROW, row_b: ezr.ROW, settings: list[ezr.COL]) -> list[tuple[ezr.COL, any, any]]:
    """Return list of (col, a_val, b_val) where values differ."""
    diffs = []
    for c in settings:
        a = row_a[c.at]
        b = row_b[c.at]
        if a != b:
            diffs.append((c, a, b))
    return diffs


# --------------------------------------------------------------------
# APACHE-15: small change recommendation
# --------------------------------------------------------------------
def apache_15(tbl: ezr.TBL) -> None:
    goal_col = goal_column(tbl)
    goal_at = goal_col.at
    settings = setting_columns(tbl)

    # Find worst fully‑specified row (higher goal value when minimizing)
    min_or_max = goal_col.txt.endswith("-")  # True if we minimize
    worst_row = None
    worst_val = None
    for row in tbl.rows:
        if not all_settings_filled(row, settings):
            continue
        if row[goal_at] == "?":
            continue
        try:
            val = row_performance(row, goal_at)
        except ValueError:
            continue
        if worst_val is None or (val > worst_val if min_or_max else val < worst_val):
            worst_val = val
            worst_row = row

    # Fallback if no fully‑specified rows
    if worst_row is None:
        for row in tbl.rows:
            if row[goal_at] == "?":
                continue
            val = row_performance(row, goal_at)
            if worst_val is None or (val > worst_val if min_or_max else val < worst_val):
                worst_val = val
                worst_row = row

    if worst_row is None:
        print("No valid rows found for APACHE-15.")
        return

    # Find alternative with minimal number of differing settings
    best_alt = None
    best_diff_len = None
    best_alt_val = None
    for row in tbl.rows:
        if row is worst_row:
            continue
        if not all_settings_filled(row, settings):
            continue
        if row[goal_at] == "?":
            continue
        alt_val = row_performance(row, goal_at)
        if min_or_max and alt_val >= worst_val:
            continue
        if not min_or_max and alt_val <= worst_val:
            continue
        diffs = diff_settings(worst_row, row, settings)
        diff_len = len(diffs)
        if best_alt is None or diff_len < best_diff_len or (diff_len == best_diff_len and alt_val < best_alt_val):
            best_alt = row
            best_diff_len = diff_len
            best_alt_val = alt_val

    # If none found, pick any better row (ignore diff count)
    if best_alt is None:
        for row in tbl.rows:
            if row is worst_row:
                continue
            if not all_settings_filled(row, settings):
                continue
            if row[goal_at] == "?":
                continue
            alt_val = row_performance(row, goal_at)
            if min_or_max and alt_val >= worst_val:
                continue
            if not min_or_max and alt_val <= worst_val:
                continue
            best_alt = row
            best_alt_val = alt_val
            best_diff_len = len(diff_settings(worst_row, row, settings))
            break

    # Prepare changes dictionary
    changes = {}
    if best_alt is not None:
        diffs = diff_settings(worst_row, best_alt, settings)
        for c, a, b in diffs:
            changes[c.txt] = [a, b]

    # Output
    print("== APACHE-15 ==")
    print(f"Worst row: {row_index(tbl, worst_row)}")
    if best_alt is not None:
        print(f"Alternative row: {row_index(tbl, best_alt)}")
        print(f"Changes: {changes}")
        print(f"Performance: before={worst_val:.0f}, after={best_alt_val:.0f}")
    else:
        print("No alternative found.")
    # Machine‑checked answer
    ans = {
        "row": row_index(tbl, worst_row),
        "alternative": row_index(tbl, best_alt) if best_alt else None,
        "changes": changes,
        "performance": {"before": worst_val, "after": best_alt_val if best_alt else None}
    }
    print(f"ANSWER APACHE-15 {json.dumps(ans)}")


# --------------------------------------------------------------------
# APACHE-16: closest to target with minimal changes
# --------------------------------------------------------------------
def apache_16(tbl: ezr.TBL) -> None:
    goal_col = goal_column(tbl)
    goal_at = goal_col.at
    settings = setting_columns(tbl)

    # Determine target: best (minimum) performance value
    best_val = None
    for row in tbl.rows:
        if row[goal_at] == "?":
            continue
        val = row_performance(row, goal_at)
        if best_val is None or val < best_val:
            best_val = val

    # Find worst fully‑specified row
    min_or_max = goal_col.txt.endswith("-")  # True if we minimize
    worst_row = None
    worst_val = None
    for row in tbl.rows:
        if not all_settings_filled(row, settings):
            continue
        if row[goal_at] == "?":
            continue
        val = row_performance(row, goal_at)
        if worst_val is None or (val > worst_val if min_or_max else val < worst_val):
            worst_val = val
            worst_row = row

    if worst_row is None:
        print("No valid rows found for APACHE-16.")
        return

    # Find alternative that is closer to target than worst,
    # with the fewest differing settings
    best_alt = None
    best_diff_len = None
    best_dist = None
    for row in tbl.rows:
        if row is worst_row:
            continue
        if not all_settings_filled(row, settings):
            continue
        if row[goal_at] == "?":
            continue
        alt_val = row_performance(row, goal_at)
        dist = abs(alt_val - best_val)
        dist_worst = abs(worst_val - best_val)
        if dist >= dist_worst:
            continue  # not closer
        diffs = diff_settings(worst_row, row, settings)
        diff_len = len(diffs)
        if best_alt is None or diff_len < best_diff_len or (
            diff_len == best_diff_len and dist < best_dist
        ):
            best_alt = row
            best_diff_len = diff_len
            best_dist = dist

    # Fallback: pick any better row if none closer
    if best_alt is None:
        for row in tbl.rows:
            if row is worst_row:
                continue
            if not all_settings_filled(row, settings):
                continue
            if row[goal_at] == "?":
                continue
            alt_val = row_performance(row, goal_at)
            if min_or_max and alt_val >= worst_val:
                continue
            if not min_or_max and alt_val <= worst_val:
                continue
            diffs = diff_settings(worst_row, row, settings)
            best_alt = row
            best_diff_len = len(diffs)
            best_dist = abs(alt_val - best_val)
            break

    changed = [c.txt for c, _, _ in diff_settings(worst_row, best_alt, settings)] if best_alt else []

    print("== APACHE-16 ==")
    print(f"Worst row: {row_index(tbl, worst_row)}")
    print(f"Target: {best_val:.0f}")
    if best_alt is not None:
        print(f"Alternative row: {row_index(tbl, best_alt)}")
        print(f"Changed settings: {changed}")
    else:
        print("No alternative found.")
    # Machine‑checked answer
    ans = {
        "row": row_index(tbl, worst_row),
        "target": best_val,
        "alternative": row_index(tbl, best_alt) if best_alt else None,
        "changed": changed
    }
    print(f"ANSWER APACHE-16 {json.dumps(ans)}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))

    apache_15(tbl)
    apache_16(tbl)


if __name__ == "__main__":
    main()

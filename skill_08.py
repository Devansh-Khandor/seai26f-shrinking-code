#!/usr/bin/env python3
"""
skill_08.py

Entry point for APACHE-15 and APACHE-16 skill.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Common utilities
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the CSV file and return list of rows (dict) and list of column names.

    Missing numeric cells ('?') are stored as None.
    Each row dict contains an additional key '_row_num' with its 1‑based
    position among data rows.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV file has no header.")
        for idx, raw_row in enumerate(reader, start=1):
            row: Dict[str, Any] = {}
            for col, val in raw_row.items():
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = int(val)
                    except ValueError:
                        row[col] = None
            row['_row_num'] = idx
            rows.append(row)
    return rows, headers

def identify_goal_column(headers: List[str]) -> str:
    """Return the first column name that ends with '-' or '+'."""
    for col in headers:
        if col.endswith('-') or col.endswith('+'):
            return col
    raise ValueError("No goal column found.")

def identify_setting_columns(headers: List[str], goal_col: str) -> List[str]:
    """Return setting columns (exclude goal and columns ending with 'X')."""
    return [c for c in headers if c != goal_col and not c.endswith('X')]

def hamming_distance(row_a: Dict[str, Any], row_b: Dict[str, Any], cols: List[str]) -> int:
    """Number of differing values among cols where both rows have a value."""
    diff = 0
    for col in cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va is None or vb is None:
            continue
        if va != vb:
            diff += 1
    return diff

# --------------------------------------------------------------------------- #
# APACHE-15 logic
# --------------------------------------------------------------------------- #

def apache_15(rows: List[Dict[str, Any]],
              goal_col: str,
              setting_cols: List[str]) -> Tuple[int, int, Dict[str, Any], List[str], Dict[str, List[int]], Dict[str, int]]:
    """
    Return poor row number, alternative row number, performance dict, changes dict, and list of changed settings.
    """
    # Find the worst row (highest value for a min goal)
    valid_rows = [r for r in rows if r[goal_col] is not None]
    if not valid_rows:
        raise ValueError("No rows with a goal value.")
    poor_row = max(valid_rows, key=lambda r: r[goal_col])  # worst for min goal
    poor_num = poor_row['_row_num']
    poor_perf = poor_row[goal_col]

    # Search for alternative rows differing in 1 or 2 settings with better performance
    candidates = []
    for r in rows:
        if r['_row_num'] == poor_num:
            continue
        if r[goal_col] is None:
            continue
        diff = hamming_distance(poor_row, r, setting_cols)
        if diff in (1, 2):
            candidates.append((diff, r[goal_col], r))

    if not candidates:
        # Fallback to the best row overall
        best_row = min(valid_rows, key=lambda r: r[goal_col])
        alt_row = best_row
    else:
        # Pick candidate with lowest performance (better)
        alt_row = min(candidates, key=lambda t: t[1])[2]

    alt_num = alt_row['_row_num']
    alt_perf = alt_row[goal_col]

    # Build changes dict
    changes = {}
    changed_settings = []
    for col in setting_cols:
        va = poor_row.get(col)
        vb = alt_row.get(col)
        if va is None or vb is None:
            continue
        if va != vb:
            changes[col] = [va, vb]
            changed_settings.append(col)

    performance = {"before": poor_perf, "after": alt_perf}
    return poor_num, alt_num, performance, changes, changed_settings

# --------------------------------------------------------------------------- #
# APACHE-16 logic
# --------------------------------------------------------------------------- #

def apache_16(rows: List[Dict[str, Any]],
              goal_col: str,
              setting_cols: List[str]) -> Tuple[int, float, int, List[str]]:
    """
    Return poor row number, target performance, alternative row number,
    and list of settings that differ between poor and alternative.
    """
    # Identify the best (target) performance
    valid_rows = [r for r in rows if r[goal_col] is not None]
    best_row = min(valid_rows, key=lambda r: r[goal_col])
    target_perf = best_row[goal_col]

    # Find the worst row
    poor_row = max(valid_rows, key=lambda r: r[goal_col])
    poor_num = poor_row['_row_num']
    poor_perf = poor_row[goal_col]
    diff_poor = abs(poor_perf - target_perf)

    # Candidates: rows closer to target than poor row
    candidates = []
    for r in rows:
        if r['_row_num'] == poor_num or r[goal_col] is None:
            continue
        diff = abs(r[goal_col] - target_perf)
        if diff < diff_poor:
            hd = hamming_distance(poor_row, r, setting_cols)
            candidates.append((hd, diff, r))

    if not candidates:
        # Fallback: use best row as alternative
        alt_row = best_row
        changed = []
    else:
        # Pick candidate with fewest differing settings; tie-break by closest to target
        candidates.sort(key=lambda t: (t[0], t[1]))
        alt_row = candidates[0][2]
        changed = [col for col in setting_cols if poor_row.get(col) != alt_row.get(col)]

    alt_num = alt_row['_row_num']
    return poor_num, target_perf, alt_num, changed

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    goal_col = identify_goal_column(headers)
    setting_cols = identify_setting_columns(headers, goal_col)

    # --------------------------------------------------------------------- #
    # APACHE-15
    # --------------------------------------------------------------------- #
    poor_num, alt_num, perf, changes, changed_settings = apache_15(rows, goal_col, setting_cols)

    print("== APACHE-15 ==")
    print(f"Poor row: {poor_num}  Performance: {perf['before']}")
    print(f"Alternative row: {alt_num}  Performance: {perf['after']}")
    print(f"Number of changed settings: {len(changed_settings)}")
    # Machine‑checked answer
    answer_15 = {
        "row": poor_num,
        "alternative": alt_num,
        "changes": changes,
        "performance": perf
    }
    print("ANSWER APACHE-15", json.dumps(answer_15, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-16
    # --------------------------------------------------------------------- #
    poor_num, target_perf, alt_num, changed = apache_16(rows, goal_col, setting_cols)

    print("\n== APACHE-16 ==")
    print(f"Poor row: {poor_num}  Target: {target_perf}")
    print(f"Alternative row: {alt_num}")
    print(f"Changed settings: {changed}")
    # Machine‑checked answer
    answer_16 = {
        "row": poor_num,
        "target": target_perf,
        "alternative": alt_num,
        "changed": changed
    }
    print("ANSWER APACHE-16", json.dumps(answer_16, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

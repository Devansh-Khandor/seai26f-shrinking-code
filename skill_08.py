#!/usr/bin/env python3
"""
skill_08.py

Implements APACHE-15 and APACHE-16 analyses.

Usage:
    python skill_08.py <csv_path>
"""

import sys
import csv
import json
from collections import defaultdict

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_float(val):
    if val is None or val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def get_adjustable_settings(columns, goal_col):
    """Return list of setting column names: not goal and not ending with 'X'."""
    return [c for c in columns
            if not c.endswith('X') and not c.endswith('+') and not c.endswith('-')]

def best_row(rows, goal_col):
    """Return (row_index, goal_value) of best (min) row."""
    best_idx = None
    best_val = None
    for idx, row in enumerate(rows, 1):
        v = to_float(row[goal_col])
        if v is None:
            continue
        if best_idx is None or v < best_val:
            best_idx = idx
            best_val = v
    return best_idx, best_val

def worst_row(rows, goal_col):
    """Return (row_index, goal_value) of worst (max) row."""
    worst_idx = None
    worst_val = None
    for idx, row in enumerate(rows, 1):
        v = to_float(row[goal_col])
        if v is None:
            continue
        if worst_idx is None or v > worst_val:
            worst_idx = idx
            worst_val = v
    return worst_idx, worst_val

def valid_rows(rows, goal_col, settings):
    """Return list of (index, row) where goal and all settings are present."""
    valid = []
    for idx, row in enumerate(rows, 1):
        if to_float(row[goal_col]) is None:
            continue
        missing = any(row[s] == '?' or row[s] == '' for s in settings)
        if missing:
            continue
        valid.append((idx, row))
    return valid

def differences(row1, row2, settings):
    """Return list of settings where row1 and row2 differ (both values present)."""
    diff = []
    for s in settings:
        v1 = row1[s]
        v2 = row2[s]
        if v1 == '?' or v2 == '?' or v1 == '' or v2 == '':
            continue
        if v1 != v2:
            diff.append(s)
    return diff

def find_alternative_for_poor(row_idx, row, valid, goal_col, settings):
    """APACHE-15: find an alternative differing in ≤2 settings with lower performance."""
    orig_perf = to_float(row[goal_col])
    best_alt = None
    best_improve = -1
    best_diff_count = 3  # start >2
    for idx, alt_row in valid:
        if idx == row_idx:
            continue
        diff = differences(row, alt_row, settings)
        if len(diff) > 2:
            continue
        alt_perf = to_float(alt_row[goal_col])
        if alt_perf is None or alt_perf >= orig_perf:
            continue
        improve = orig_perf - alt_perf
        # Prefer larger improvement, then fewer changes, then lower row number
        if (improve > best_improve or
            (improve == best_improve and len(diff) < best_diff_count) or
            (improve == best_improve and len(diff) == best_diff_count and idx < best_alt[0])):
            best_alt = (idx, alt_row, diff, alt_perf)
            best_improve = improve
            best_diff_count = len(diff)
    return best_alt

def find_closest_to_target(row_idx, row, valid, goal_col, settings, target):
    """APACHE-16: find an alternative closer to target with fewest changes."""
    orig_perf = to_float(row[goal_col])
    orig_dist = abs(orig_perf - target)
    best_candidate = None
    best_dist = None
    best_diff_count = None
    for idx, alt_row in valid:
        if idx == row_idx:
            continue
        alt_perf = to_float(alt_row[goal_col])
        if alt_perf is None:
            continue
        alt_dist = abs(alt_perf - target)
        if alt_dist >= orig_dist:
            continue  # must be closer
        diff = differences(row, alt_row, settings)
        # Prefer closer distance, then fewer changes, then lower row number
        if (best_candidate is None or
            alt_dist < best_dist or
            (alt_dist == best_dist and len(diff) < best_diff_count) or
            (alt_dist == best_dist and len(diff) == best_diff_count and idx < best_candidate[0])):
            best_candidate = (idx, alt_row, diff)
            best_dist = alt_dist
            best_diff_count = len(diff)
    return best_candidate

def main(csv_path):
    rows, columns = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    # Find goal column (first ending with + or -)
    goal_col = None
    for c in columns:
        if c.endswith('+') or c.endswith('-'):
            goal_col = c
            break
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    settings = get_adjustable_settings(columns, goal_col)

    # Valid rows (goal and all settings present)
    valid = valid_rows(rows, goal_col, settings)

    # Find worst (highest) row
    worst_idx, worst_perf = worst_row(rows, goal_col)
    if worst_idx is None:
        print("No valid performance values found.")
        sys.exit(1)
    worst_row_data = rows[worst_idx - 1]

    # ================= APACHE-15 =================
    alt15 = find_alternative_for_poor(worst_idx, worst_row_data, valid, goal_col, settings)
    print("== APACHE-15 ==")
    if alt15 is None:
        print(f"Row {worst_idx} is the only configuration or no better alternative exists.")
    else:
        alt_idx, alt_row, diff_s, alt_perf = alt15
        changes = {s: [worst_row_data[s], alt_row[s]] for s in diff_s}
        print(f"Poor row {worst_idx} has performance {worst_perf:.4f}.")
        print(f"Alternative row {alt_idx} differs in settings: {', '.join(diff_s)}.")
        print(f"Performance improves to {alt_perf:.4f}.")
    answer15 = {
        "row": worst_idx,
        "alternative": alt15[0] if alt15 else None,
        "changes": changes if alt15 else {},
        "performance": {
            "before": round(worst_perf, 4) if worst_perf is not None else None,
            "after": round(alt_perf, 4) if alt15 else None
        }
    }
    print("ANSWER APACHE-15 " + json.dumps(answer15, separators=(',', ':')))

    # ================= APACHE-16 =================
    best_idx, best_perf = best_row(rows, goal_col)
    if best_perf is None:
        print("No best performance found.")
        sys.exit(1)
    target_val = best_perf  # target better than worst
    alt16 = find_closest_to_target(worst_idx, worst_row_data, valid, goal_col, settings, target_val)
    print("\n== APACHE-16 ==")
    if alt16 is None:
        print(f"No alternative found that is closer to target {target_val:.4f}.")
    else:
        alt_idx, alt_row, diff_s = alt16
        print(f"Poor row {worst_idx} performance {worst_perf:.4f}.")
        print(f"Target performance {target_val:.4f}.")
        print(f"Alternative row {alt_idx} differs in settings: {', '.join(diff_s)}.")
    answer16 = {
        "row": worst_idx,
        "target": round(target_val, 4),
        "alternative": alt16[0] if alt16 else None,
        "changed": diff_s if alt16 else []
    }
    print("ANSWER APACHE-16 " + json.dumps(answer16, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

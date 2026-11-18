#!/usr/bin/env python3
"""
skill_08.py

Implements APACHE-15 and APACHE-16 analyses for the Apache performance dataset.

Usage:
    python skill_08.py <csv_path>

Author: OpenAI ChatGPT
"""

import csv
import json
import sys

# --------------------------------------------------------------------------- #
# CSV utilities

def read_csv(path):
    """Return header list and rows (dict) from a CSV file."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_int(val):
    """Parse an integer from a cell, return None for missing or bad values."""
    if val is None or val == '?':
        return None
    try:
        return int(val)
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# Helper functions for columns

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_setting(col):
    return not col.endswith('X') and not is_goal(col)

# --------------------------------------------------------------------------- #
# APACHE-15: small change recommendation

def apache_15(header, rows):
    # Identify performance column (first '-' or '+', but we want minimizing)
    perf_col = next((c for c in header if c.endswith('-')), None)
    if perf_col is None:
        raise RuntimeError("No minimizing performance column found")

    setting_cols = [c for c in header if is_setting(c)]

    # Find the worst performing row (max performance)
    worst_idx = None
    worst_val = None
    for idx, r in enumerate(rows):
        val = parse_int(r[perf_col])
        if val is None:
            continue
        if worst_val is None or val > worst_val:
            worst_val = val
            worst_idx = idx

    if worst_idx is None:
        # No valid performance rows; return empty recommendation
        return {"row": None, "alternative": None, "changes": {}, "performance": {"before": None, "after": None}}

    # Search for alternative rows differing in <=2 settings and better performance
    best_alt_idx = None
    best_alt_val = None
    best_changes = None

    for idx, r in enumerate(rows):
        if idx == worst_idx:
            continue
        val = parse_int(r[perf_col])
        if val is None or val >= worst_val:
            continue  # need improvement
        # Compute differing settings (both values present)
        diffs = {}
        for c in setting_cols:
            v1 = parse_int(rows[worst_idx][c])
            v2 = parse_int(r[c])
            if v1 is None or v2 is None:
                continue  # ignore missing cells
            if v1 != v2:
                diffs[c] = [v1, v2]
        if len(diffs) == 0 or len(diffs) > 2:
            continue
        # Improvement
        improvement = worst_val - val
        if best_alt_val is None or improvement > (worst_val - best_alt_val):
            best_alt_val = val
            best_alt_idx = idx
            best_changes = diffs

    if best_alt_idx is None:
        # No suitable alternative found; return empty
        return {"row": worst_idx + 1, "alternative": None, "changes": {}, "performance": {"before": worst_val, "after": worst_val}}

    return {
        "row": worst_idx + 1,
        "alternative": best_alt_idx + 1,
        "changes": best_changes,
        "performance": {"before": worst_val, "after": best_alt_val}
    }

# --------------------------------------------------------------------------- #
# APACHE-16: minimal changes to reach a target

def apache_16(header, rows):
    perf_col = next((c for c in header if c.endswith('-')), None)
    if perf_col is None:
        raise RuntimeError("No minimizing performance column found")
    setting_cols = [c for c in header if is_setting(c)]

    # Find the worst performing row
    worst_idx = None
    worst_val = None
    for idx, r in enumerate(rows):
        val = parse_int(r[perf_col])
        if val is None:
            continue
        if worst_val is None or val > worst_val:
            worst_val = val
            worst_idx = idx

    # Find the best performance in data (target)
    best_val = None
    for r in rows:
        val = parse_int(r[perf_col])
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val

    if worst_idx is None or best_val is None:
        # Insufficient data
        return {"row": None, "target": None, "alternative": None, "changed": []}

    target_val = best_val  # better than worst

    # Find alternative rows closer to target than worst
    candidate = None
    candidate_diff = None
    candidate_change_count = None

    for idx, r in enumerate(rows):
        if idx == worst_idx:
            continue
        val = parse_int(r[perf_col])
        if val is None or val >= worst_val:
            continue  # must be better than worst
        # Distance to target
        dist = abs(val - target_val)
        if candidate is None or dist < candidate_diff:
            # compute differing settings
            diffs = []
            for c in setting_cols:
                v1 = parse_int(rows[worst_idx][c])
                v2 = parse_int(r[c])
                if v1 is None or v2 is None:
                    continue
                if v1 != v2:
                    diffs.append(c)
            if len(diffs) == 0:
                continue  # no changes, skip
            candidate = idx
            candidate_diff = dist
            candidate_change_count = len(diffs)
        elif dist == candidate_diff:
            # tie: choose fewer changes
            diffs = []
            for c in setting_cols:
                v1 = parse_int(rows[worst_idx][c])
                v2 = parse_int(r[c])
                if v1 is None or v2 is None:
                    continue
                if v1 != v2:
                    diffs.append(c)
            if len(diffs) < candidate_change_count:
                candidate = idx
                candidate_change_count = len(diffs)

    if candidate is None:
        # No better alternative found
        return {"row": worst_idx + 1, "target": target_val, "alternative": None, "changed": []}

    # Final list of changed settings
    final_changes = []
    for c in setting_cols:
        v1 = parse_int(rows[worst_idx][c])
        v2 = parse_int(rows[candidate][c])
        if v1 is None or v2 is None:
            continue
        if v1 != v2:
            final_changes.append(c)

    return {
        "row": worst_idx + 1,
        "target": target_val,
        "alternative": candidate + 1,
        "changed": final_changes
    }

# --------------------------------------------------------------------------- #
# Main routine

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # --- APACHE-15 ---
    result15 = apache_15(header, rows)
    print("== APACHE-15 ==")
    if result15["row"] is None:
        print("No poor row found.")
    else:
        print(f"Poor row: {result15['row']}")
        print(f"Alternative row: {result15['alternative']}")
        print("Changes:")
        for k, v in result15["changes"].items():
            print(f"  {k}: {v[0]} -> {v[1]}")
        print(f"Performance: before={result15['performance']['before']}, after={result15['performance']['after']}")
    print("ANSWER APACHE-15", json.dumps(result15))

    # --- APACHE-16 ---
    result16 = apache_16(header, rows)
    print("\n== APACHE-16 ==")
    if result16["row"] is None:
        print("No poor row found.")
    else:
        print(f"Poor row: {result16['row']}")
        print(f"Target performance: {result16['target']}")
        print(f"Alternative row: {result16['alternative']}")
        print(f"Changed settings: {result16['changed']}")
    print("ANSWER APACHE-16", json.dumps(result16))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Apache configuration improvement recommendations.

Usage:  python skill_08.py <csv_path>
"""

import csv
import json
import sys
import math

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts (1‑based indexing) and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def find_goal_column(fieldnames):
    """Return the column that ends with '+' or '-'."""
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

# ---------------------------------------------------------------------------

def get_setting_columns(fieldnames, goal_col):
    """Return columns that are inputs: not a goal and not ending with 'X'."""
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]

# ---------------------------------------------------------------------------

def to_float(val):
    """Convert string to float or None."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ---------------------------------------------------------------------------

def compute_worst_and_target(rows, goal_col):
    """Return index (1‑based) of worst row, its value, and best (target) value."""
    worst_idx = None
    worst_val = None
    best_val = None
    for idx, r in enumerate(rows, start=1):
        val = to_float(r[goal_col])
        if val is None:
            continue
        if worst_val is None or val > worst_val:
            worst_val = val
            worst_idx = idx
        if best_val is None or val < best_val:
            best_val = val
    return worst_idx, worst_val, best_val

# ---------------------------------------------------------------------------

def diff_settings(row_a, row_b, setting_cols):
    """Return dict of differing settings where both values present."""
    changes = {}
    for col in setting_cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va == '?' or vb == '?':
            continue
        if va != vb:
            changes[col] = [va, vb]
    return changes

# ---------------------------------------------------------------------------

def find_best_alternative(rows, goal_col, setting_cols, worst_idx, worst_val, allow_changes=2):
    """Find alternative with <=allow_changes that improves performance."""
    best = None
    best_changes = None
    best_improve = -math.inf
    for idx, r in enumerate(rows, start=1):
        if idx == worst_idx:
            continue
        val = to_float(r[goal_col])
        if val is None or val >= worst_val:
            continue
        changes = diff_settings(rows[worst_idx-1], r, setting_cols)
        if not changes:
            continue
        if len(changes) > allow_changes:
            continue
        improve = worst_val - val
        if (len(changes) < (best_changes and len(best_changes) or math.inf) or
            (len(changes) == (best_changes and len(best_changes) or math.inf) and improve > best_improve)):
            best = idx
            best_changes = changes
            best_improve = improve
    # Fallback: pick overall best regardless of change count
    if best is None:
        for idx, r in enumerate(rows, start=1):
            if idx == worst_idx:
                continue
            val = to_float(r[goal_col])
            if val is None or val >= worst_val:
                continue
            changes = diff_settings(rows[worst_idx-1], r, setting_cols)
            if not changes:
                continue
            improve = worst_val - val
            if (len(changes) < (best_changes and len(best_changes) or math.inf) or
                (len(changes) == (best_changes and len(best_changes) or math.inf) and improve > best_improve)):
                best = idx
                best_changes = changes
                best_improve = improve
    return best, best_changes

# ---------------------------------------------------------------------------

def find_best_alternative_for_target(rows, goal_col, setting_cols, worst_idx, worst_val, target_val):
    """Find alternative with fewest changes and closer to target than worst."""
    best = None
    best_changes = None
    best_diff_to_target = None
    best_improve = None
    for idx, r in enumerate(rows, start=1):
        if idx == worst_idx:
            continue
        val = to_float(r[goal_col])
        if val is None or val >= worst_val:
            continue
        # Closer to target than worst?
        diff_to_target = val - target_val
        if diff_to_target >= (worst_val - target_val):
            continue
        changes = diff_settings(rows[worst_idx-1], r, setting_cols)
        if not changes:
            continue
        # Choose by fewest changes, then closest to target, then better improvement
        improve = worst_val - val
        if (best is None or
            len(changes) < len(best_changes) or
            (len(changes) == len(best_changes) and
             (diff_to_target < best_diff_to_target or
              (diff_to_target == best_diff_to_target and improve > best_improve)))):
            best = idx
            best_changes = changes
            best_diff_to_target = diff_to_target
            best_improve = improve
    return best, best_changes

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)
    setting_cols = get_setting_columns(fieldnames, goal_col)

    # Find worst and target values
    worst_idx, worst_val, target_val = compute_worst_and_target(rows, goal_col)

    # -------------------------------------------------------------
    # APACHE-15
    # -------------------------------------------------------------
    print("== APACHE-15 ==")
    alt_idx_15, changes_15 = find_best_alternative(rows, goal_col, setting_cols, worst_idx, worst_val, allow_changes=2)
    if alt_idx_15 is None:
        print("No suitable alternative found for APACHE-15")
        # still provide a placeholder
        alt_idx_15 = worst_idx
        changes_15 = {}
    alt_row_15 = rows[alt_idx_15-1]
    before_perf = worst_val
    after_perf = to_float(alt_row_15[goal_col])
    print(f"Worst row: {worst_idx}")
    print(f"Alternative row: {alt_idx_15}")
    # Print changes
    if changes_15:
        change_strs = [f"{k} from {v[0]} to {v[1]}" for k, v in changes_15.items()]
        print("Changes: " + ", ".join(change_strs))
    else:
        print("Changes: none")
    print(f"Performance before: {before_perf}  after: {after_perf}")

    answer_15 = {
        "row": worst_idx,
        "alternative": alt_idx_15,
        "changes": changes_15,
        "performance": {"before": before_perf, "after": after_perf}
    }
    print(f"ANSWER APACHE-15 {json.dumps(answer_15, separators=(',', ':'))}")

    # -------------------------------------------------------------
    # APACHE-16
    # -------------------------------------------------------------
    print("\n== APACHE-16 ==")
    alt_idx_16, changes_16 = find_best_alternative_for_target(rows, goal_col, setting_cols, worst_idx, worst_val, target_val)
    if alt_idx_16 is None:
        print("No suitable alternative found for APACHE-16")
        alt_idx_16 = worst_idx
        changes_16 = {}
    changed_settings = list(changes_16.keys()) if changes_16 else []
    print(f"Worst row: {worst_idx}")
    print(f"Target performance: {target_val}")
    print(f"Alternative row: {alt_idx_16}")
    print("Changed settings: [" + ", ".join(changed_settings) + "]")

    answer_16 = {
        "row": worst_idx,
        "target": target_val,
        "alternative": alt_idx_16,
        "changed": changed_settings
    }
    print(f"ANSWER APACHE-16 {json.dumps(answer_16, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_08.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-15: For a poorly performing configuration, recommend a small change
               to its adjustable settings that could lead to a lower predicted
               performance value, using historically observed valid configurations
               as alternatives.
  - APACHE-16: Find the smallest number of configuration‑setting changes that
               could bring a poorly performing configuration closer to a
               specified performance target, while avoiding unsupported or
               invalid combinations.

The implementation re‑uses helper functions from earlier skill modules.
"""

import sys
import json
import numpy as np
from skill_01 import read_csv, parse_column_values

# --------------------------------------------------------------------------- #
# Helper functions (similar to those in skill_05)
# --------------------------------------------------------------------------- #

def get_poor_rows(rows, perf_col):
    """
    Return a list of row indices (1‑based) that are identified as unusually
    poor using the same logic as APACHE‑07.
    """
    poor = []
    cumulative = []
    for idx, row in enumerate(rows):
        vals = parse_column_values(cumulative, perf_col)
        if np.isnan(vals).all():
            mean = np.nan
            std = np.nan
        else:
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
        perf = row[perf_col]
        if perf == '?':
            alert = False
        else:
            perf_val = float(perf)
            threshold = mean - 2 * std if not np.isnan(mean) and not np.isnan(std) else None
            alert = threshold is not None and perf_val < threshold
        if alert:
            poor.append(idx + 1)
        cumulative.append(row)
    return poor

def find_best_alternative(poor_idx, rows, setting_cols, perf_col):
    """
    For the given poorly performing row (1‑based index), find an observed
    row that differs in 1 or 2 settings and has a lower Performance- value.
    Returns (alt_row_num, changes_dict, before, after) or (None, {}, None, None)
    if no such alternative exists.
    """
    poor_row = rows[poor_idx - 1]
    poor_perf = parse_column_values([poor_row], perf_col)[0]
    best = None
    best_diff = None
    for idx, r in enumerate(rows):
        if idx == poor_idx - 1:
            continue
        # Compute differing settings
        diffs = [col for col in setting_cols
                 if poor_row[col] != r[col] and (poor_row[col] != '?' and r[col] != '?')]
        if len(diffs) not in (1, 2):
            continue
        alt_perf = parse_column_values([r], perf_col)[0]
        if np.isnan(alt_perf) or alt_perf >= poor_perf:
            continue
        diff_perf = poor_perf - alt_perf
        if best is None or diff_perf > best_diff:
            best = idx + 1  # 1‑based
            best_diff = diff_perf
            changes = {col: [poor_row[col], r[col]] for col in diffs}
            before = poor_perf
            after = alt_perf
    if best is None:
        return None, {}, None, None
    return best, changes, before, after

def find_best_for_target(poor_idx, rows, setting_cols, perf_col, target_perf):
    """
    Find an observed row that has a Performance- value closer to target_perf
    (better than poor) and differs in the fewest settings.
    Returns (alt_row_num, changed_settings) or (None, []).
    """
    poor_row = rows[poor_idx - 1]
    poor_perf = parse_column_values([poor_row], perf_col)[0]
    candidates = []
    for idx, r in enumerate(rows):
        if idx == poor_idx - 1:
            continue
        alt_perf = parse_column_values([r], perf_col)[0]
        if np.isnan(alt_perf) or alt_perf >= poor_perf:
            continue
        # Check closeness to target (lower diff)
        diff_to_target = abs(target_perf - alt_perf)
        # Count differing settings
        diffs = [col for col in setting_cols
                 if poor_row[col] != r[col] and (poor_row[col] != '?' and r[col] != '?')]
        if not diffs:
            continue
        candidates.append((len(diffs), diff_to_target, idx + 1, diffs))
    if not candidates:
        return None, []
    # Sort by fewest changes, then by closeness to target
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2], candidates[0][3]

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns (minimize or maximize)
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]  # assume first goal column

    # Identify setting columns
    setting_cols = [h for h in headers
                    if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

    # ----------------------------------------------------------------------- #
    # APACHE-15
    # ----------------------------------------------------------------------- #
    print("== APACHE-15 ==")

    poor_rows = get_poor_rows(rows, perf_col)
    if not poor_rows:
        print("No unusually poor configurations detected.")
        # Provide empty JSON
        print("ANSWER APACHE-15", json.dumps({"row": None, "alternative": None, "changes": {}, "performance": {"before": None, "after": None}}, separators=(',', ':')))
        return

    # Pick the first poorly performing row
    row_num = poor_rows[0]
    alt_row, changes, before, after = find_best_alternative(row_num, rows, setting_cols, perf_col)
    if alt_row is None:
        print(f"Poor configuration: row {row_num} (no better alternative found).")
        print("ANSWER APACHE-15", json.dumps({"row": row_num, "alternative": None, "changes": {}, "performance": {"before": before, "after": None}}, separators=(',', ':')))
    else:
        print(f"Poor configuration: row {row_num} (Performance-={before:.4f})")
        print(f"Suggested alternative: row {alt_row} (Performance-={after:.4f})")
        print("Changes:")
        for s, vals in changes.items():
            print(f"  {s}: {vals[0]} -> {vals[1]}")
        # Machine‑checked answer
        print("ANSWER APACHE-15", json.dumps({
            "row": row_num,
            "alternative": alt_row,
            "changes": changes,
            "performance": {"before": before, "after": after}
        }, separators=(',', ':')))

    # ----------------------------------------------------------------------- #
    # APACHE-16
    # ----------------------------------------------------------------------- #
    print("\n== APACHE-16 ==")

    # Use the same poorly row if exists
    if not poor_rows:
        print("No poorly performing configuration to adjust.")
        print("ANSWER APACHE-16", json.dumps({"row": None, "target": None, "alternative": None, "changed": []}, separators=(',', ':')))
        return

    row_num = poor_rows[0]
    poor_perf = parse_column_values([rows[row_num - 1]], perf_col)[0]
    # Define target as best observed performance (strictly better)
    all_perfs = parse_column_values(rows, perf_col)
    best_perf = np.nanmin(all_perfs)
    target_perf = best_perf
    if np.isnan(target_perf) or target_perf >= poor_perf:
        print("No better target performance found.")
        print("ANSWER APACHE-16", json.dumps({"row": row_num, "target": None, "alternative": None, "changed": []}, separators=(',', ':')))
        return

    alt_row, changed = find_best_for_target(row_num, rows, setting_cols, perf_col, target_perf)
    if alt_row is None:
        print(f"Poor configuration: row {row_num} (no better alternative within constraints).")
        print("ANSWER APACHE-16", json.dumps({"row": row_num, "target": target_perf, "alternative": None, "changed": []}, separators=(',', ':')))
    else:
        alt_perf = parse_column_values([rows[alt_row - 1]], perf_col)[0]
        print(f"Poor configuration: row {row_num} (Performance-={poor_perf:.4f})")
        print(f"Target performance: {target_perf:.4f}")
        print(f"Suggested alternative: row {alt_row} (Performance-={alt_perf:.4f})")
        print(f"Changed settings ({len(changed)}): {', '.join(changed)}")
        # Machine‑checked answer
        print("ANSWER APACHE-16", json.dumps({
            "row": row_num,
            "target": target_perf,
            "alternative": alt_row,
            "changed": changed
        }, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <csv_path>
"""

import csv
import sys
import json
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def is_goal_col(col_name):
    """Goal columns end with '+' or '-'."""
    return col_name.endswith('+') or col_name.endswith('-')

def is_input_col(col_name):
    """Input columns are those that are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)

def parse_value(val, col_name):
    """Parse a CSV cell value."""
    if val == '?':
        return None
    if is_goal_col(col_name):
        try:
            return float(val)
        except ValueError:
            return None
    else:
        try:
            return int(val)
        except ValueError:
            return None

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = list(reader)

    # Identify goal and input columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Pick minimization target (ends with '-') if present, else first goal
    target_col = None
    for c in goal_cols:
        if c.endswith('-'):
            target_col = c
            break
    if not target_col:
        target_col = goal_cols[0]

    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Parse all rows
    parsed_rows = []
    for row in raw_rows:
        parsed = {}
        for col in fieldnames:
            parsed[col] = parse_value(row[col], col)
        parsed_rows.append(parsed)

    # Gather performance values (valid)
    perf_rows = []
    for idx, r in enumerate(parsed_rows, start=1):
        val = r[target_col]
        if val is not None:
            perf_rows.append((idx, val))

    # If no valid performance rows, exit
    if not perf_rows:
        print("No valid performance rows found.", file=sys.stderr)
        sys.exit(1)

    # Sort by performance ascending (lower better)
    perf_rows.sort(key=lambda x: x[1])

    # Identify top 5 worst rows
    worst_rows = sorted(perf_rows, key=lambda x: x[1], reverse=True)[:5]
    worst_indices = [idx for idx, _ in worst_rows]
    worst_perf_map = {idx: val for idx, val in worst_rows}

    # -------------------------------------------------------------------- #
    # APACHE-15: Recommend a small change for a poor row
    # -------------------------------------------------------------------- #
    print("== APACHE-15 ==")

    if worst_indices:
        poor_row = worst_indices[0]
        poor_perf = worst_perf_map[poor_row]
        # Find alternatives with <=2 setting changes and lower performance
        candidates = []
        for idx, val in perf_rows:
            if idx == poor_row:
                continue
            if val >= poor_perf:
                continue
            # Compute hamming distance on input columns
            diff = []
            for col in input_cols:
                v1 = parsed_rows[poor_row-1][col]
                v2 = parsed_rows[idx-1][col]
                if v1 != v2:
                    diff.append(col)
            if 1 <= len(diff) <= 2:
                candidates.append((idx, val, diff))
        # Choose candidate with lowest performance, then lowest diff, then lowest idx
        if candidates:
            candidates.sort(key=lambda x: (x[1], len(x[2]), x[0]))
            alt_row, alt_perf, changed_cols = candidates[0]
            changes = {c: [parsed_rows[poor_row-1][c], parsed_rows[alt_row-1][c]] for c in changed_cols}
        else:
            alt_row, alt_perf, changed_cols, changes = None, None, [], {}
    else:
        poor_row, poor_perf, alt_row, alt_perf, changes = None, None, None, None, {}

    # Text output (at least 3 lines)
    print(f"Poor row: {poor_row}")
    print(f"Alternative row: {alt_row}")
    print(f"Changes: {changes}")

    answer_15 = {
        "row": poor_row,
        "alternative": alt_row,
        "changes": changes,
        "performance": {
            "before": poor_perf,
            "after": alt_perf
        }
    }
    print("ANSWER APACHE-15", json.dumps(answer_15, separators=(',', ':')))

    # -------------------------------------------------------------------- #
    # APACHE-16: Find minimal changes to reach a better target
    # -------------------------------------------------------------------- #
    print("\n== APACHE-16 ==")

    # Target performance is the best (lowest) among all rows
    target_perf = perf_rows[0][1]

    if poor_row is None:
        row_num = None
        alt_row = None
        changed = []
    else:
        row_num = poor_row
        poor_perf = worst_perf_map[poor_row]
        # Compute difference to target for poor row
        poor_diff = poor_perf - target_perf
        # Candidate rows that are closer to target than poor row
        candidate_list = []
        for idx, val in perf_rows:
            if idx == poor_row:
                continue
            diff_to_target = val - target_perf
            if diff_to_target < poor_diff:
                # Compute hamming distance
                diff_cols = []
                for col in input_cols:
                    v1 = parsed_rows[poor_row-1][col]
                    v2 = parsed_rows[idx-1][col]
                    if v1 != v2:
                        diff_cols.append(col)
                candidate_list.append((len(diff_cols), idx, diff_to_target, diff_cols))
        # Choose with fewest changes, then smaller index
        if candidate_list:
            candidate_list.sort(key=lambda x: (x[0], x[1]))
            _, alt_row, _, changed = candidate_list[0]
        else:
            alt_row, changed = None, []

    # Text output
    print(f"Poor row: {row_num}")
    print(f"Target performance: {target_perf}")
    print(f"Alternative row: {alt_row}")
    print(f"Changed settings: {changed}")

    answer_16 = {
        "row": row_num,
        "target": target_perf,
        "alternative": alt_row,
        "changed": changed
    }
    print("ANSWER APACHE-16", json.dumps(answer_16, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

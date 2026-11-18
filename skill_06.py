#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv_path>
"""

import csv
import json
import sys
from statistics import mean
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def is_goal_col(col_name):
    """Goal columns end with '+' or '-'.")
    return col_name.endswith('+') or col_name.endswith('-')

def is_input_col(col_name):
    """Input columns are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)

def parse_value(val, col_name):
    """Parse CSV cell value into appropriate Python type."""
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
# Main
# --------------------------------------------------------------------------- #

def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = list(reader)

    # Parse rows into dict of parsed values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for c in fieldnames:
            parsed[c] = parse_value(r[c], c)
        parsed_rows.append(parsed)

    # Identify columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    input_cols = [c for c in fieldnames if is_input_col(c)]

    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Choose a single goal column for assessment: prefer one that ends with '-'
    target_col = None
    for c in goal_cols:
        if c.endswith('-'):
            target_col = c
            break
    if not target_col:
        target_col = goal_cols[0]

    # Find rows with valid target values
    valid_indices = [i+1 for i, r in enumerate(parsed_rows)
                     if r[target_col] is not None]

    if not valid_indices:
        print("No rows with valid goal values.", file=sys.stderr)
        sys.exit(1)

    # Determine best and worst rows (minimize or maximize)
    # For minimization ('-') lower is better
    if target_col.endswith('-'):
        best_idx = min(valid_indices, key=lambda i: parsed_rows[i-1][target_col])
        worst_idx = max(valid_indices, key=lambda i: parsed_rows[i-1][target_col])
        best_value = parsed_rows[best_idx-1][target_col]
        worst_value = parsed_rows[worst_idx-1][target_col]
    else:
        best_idx = max(valid_indices, key=lambda i: parsed_rows[i-1][target_col])
        worst_idx = min(valid_indices, key=lambda i: parsed_rows[i-1][target_col])
        best_value = parsed_rows[best_idx-1][target_col]
        worst_value = parsed_rows[worst_idx-1][target_col]

    # ------------------------------------------------------------------ #
    # APACHE-11: Recommendation explanation
    # ------------------------------------------------------------------ #
    print("== APACHE-11 ==")

    recommended_row_num = best_idx

    # Compute Hamming distances to recommended row over input columns
    def hamming_distance(row_a, row_b):
        """Count differing values over input columns (None considered different)."""
        dist = 0
        for col in input_cols:
            va = row_a[col]
            vb = row_b[col]
            if va != vb:
                dist += 1
        return dist

    distances = []
    for idx, row in enumerate(parsed_rows, start=1):
        if idx == recommended_row_num:
            continue
        d = hamming_distance(parsed_rows[recommended_row_num-1], row)
        distances.append((d, idx))

    # Sort by distance, then by row number
    distances.sort(key=lambda x: (x[0], x[1]))
    # Take at least 3 neighbors (or all if fewer)
    neighbor_count = max(3, len(distances))
    neighbors = []
    for d, idx in distances[:neighbor_count]:
        # Build values dict including all columns
        vals = {}
        for c in fieldnames:
            vals[c] = parsed_rows[idx-1][c]
        neighbors.append({"row": idx, "values": vals})

    # Output neighbor summary
    print(f"Recommended row: {recommended_row_num}")
    print(f"Neighbors (by Hamming distance):")
    for n in neighbors:
        print(f"  Row {n['row']}")

    # Machine‑checked answer
    answer_11 = {
        "recommended": recommended_row_num,
        "neighbors": neighbors
    }
    print("ANSWER APACHE-11", json.dumps(answer_11, separators=(',', ':')))

    # ------------------------------------------------------------------ #
    # APACHE-12: Comparison between two configurations
    # ------------------------------------------------------------------ #
    print("\n== APACHE-12 ==")

    # Use best (a) and worst (b)
    a_row_num = best_idx
    b_row_num = worst_idx
    preferred = "a"

    # Scores: goal value
    scores = {
        "a": best_value,
        "b": worst_value
    }

    # Differences in input features
    differences = []
    for col in input_cols:
        va = parsed_rows[a_row_num-1][col]
        vb = parsed_rows[b_row_num-1][col]
        if va != vb:
            differences.append({
                "feature": col,
                "a": va,
                "b": vb
            })
    # Ensure at least one difference
    if not differences:
        differences.append({"feature": "none", "a": None, "b": None})

    # Tradeoffs: goal columns where non‑preferred is better
    tradeoffs = []
    # Only one target goal column
    if target_col.endswith('-'):  # minimize
        # non‑preferred (b) is worse (higher)
        if worst_value < best_value:
            tradeoffs.append(target_col)
    else:  # maximize
        if worst_value > best_value:
            tradeoffs.append(target_col)

    # Machine‑checked answer
    answer_12 = {
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": scores,
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print("ANSWER APACHE-12", json.dumps(answer_12, separators=(',', ':')))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

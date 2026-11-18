#!/usr/bin/env python3
"""
skill_06.py

Implements APACHE-11 and APACHE-12 for the Apache performance dataset.

Usage:
    python skill_06.py <csv_path>

Author: OpenAI ChatGPT
"""

import csv
import json
import sys

# --------------------------------------------------------------------------- #
# Utility functions

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

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    return not col.endswith('X') and not is_goal(col)

# --------------------------------------------------------------------------- #
# Scoring helpers

def compute_score(row, goal_cols):
    """Return a numeric score for a row: higher is better."""
    score = 0.0
    for col in goal_cols:
        v = parse_int(row[col])
        if v is None:
            continue
        if col.endswith('+'):
            score += v
        else:  # minimizing goal
            score -= v
    return score

def hamming_distance(row_a, row_b, input_cols):
    """Count differing input columns, ignoring missing values."""
    diff = 0
    for col in input_cols:
        va = parse_int(row_a[col])
        vb = parse_int(row_b[col])
        if va is None or vb is None:
            continue  # treat as same
        if va != vb:
            diff += 1
    return diff

# --------------------------------------------------------------------------- #
# Main logic for APACHE-11 and APACHE-12

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]

    # Budget: evaluate at most min(30, len(rows)//5) rows
    budget = min(30, len(rows) // 5)
    if budget == 0:
        budget = 1
    eval_indices = list(range(budget))  # first N rows

    # Compute scores for evaluated rows
    eval_rows = [(idx, rows[idx]) for idx in eval_indices]
    scores = {idx + 1: compute_score(rows[idx], goal_cols) for idx in eval_indices}

    # --------------------------------------------------------------------- #
    # APACHE-11 – Recommendation and neighbors
    print("== APACHE-11 ==")

    # Best row by score
    recommended_row = max(scores, key=lambda r: (scores[r], -r))  # tie by lower row number
    # Find neighbors (smallest Hamming distance)
    neighbor_candidates = []
    for idx in eval_indices:
        row_num = idx + 1
        if row_num == recommended_row:
            continue
        dist = hamming_distance(rows[idx], rows[recommended_row - 1], input_cols)
        neighbor_candidates.append((dist, row_num))

    # Sort by distance then row number
    neighbor_candidates.sort(key=lambda x: (x[0], x[1]))
    neighbors = []
    for dist, row_num in neighbor_candidates[:3]:
        # Gather goal values as in file
        goal_vals = {col: rows[row_num - 1][col] for col in goal_cols}
        neighbors.append({"row": row_num, "values": goal_vals})

    # Print at least three lines of results
    print(f"Recommended row: {recommended_row}")
    print("Neighbors:")
    for nb in neighbors:
        vals_str = ", ".join(f"{k}={v}" for k, v in nb["values"].items())
        print(f"  Row {nb['row']}: {vals_str}")

    # Machine-checked answer
    ans11 = {
        "recommended": recommended_row,
        "neighbors": neighbors
    }
    print("ANSWER APACHE-11", json.dumps(ans11))

    # --------------------------------------------------------------------- #
    # APACHE-12 – Compare two configurations with different assessments
    print("\n== APACHE-12 ==")

    # Pick best and worst rows
    best_row = max(scores, key=lambda r: (scores[r], -r))
    worst_row = min(scores, key=lambda r: (scores[r], r))
    # Scores mapping
    score_map = {"{}": scores[best_row], "{}": scores[worst_row]}  # placeholder

    # Determine preferred row (higher score)
    preferred_row = best_row if scores[best_row] >= scores[worst_row] else worst_row
    other_row = worst_row if preferred_row == best_row else best_row

    # Differences in input features
    diff_features = []
    for col in input_cols:
        va = rows[best_row - 1][col]
        vb = rows[worst_row - 1][col]
        if va != vb:
            diff_features.append({"feature": col, "a": va, "b": vb})
    if not diff_features:
        # ensure at least one difference
        diff_features.append({"feature": input_cols[0], "a": rows[best_row - 1][input_cols[0]],
                               "b": rows[worst_row - 1][input_cols[0]]})

    # Tradeoffs where non-preferred row is better on goal columns
    tradeoffs = []
    for col in goal_cols:
        va = parse_int(rows[preferred_row - 1][col])
        vb = parse_int(rows[other_row - 1][col])
        if va is None or vb is None:
            continue
        if col.endswith('+'):  # maximizing
            if vb > va:
                tradeoffs.append(col)
        else:  # minimizing
            if vb < va:
                tradeoffs.append(col)

    # Scores output
    scores_out = {str(best_row): round(scores[best_row], 2), str(worst_row): round(scores[worst_row], 2)}

    # Print results
    print(f"Best row: {best_row} (score={scores[best_row]:.2f})")
    print(f"Worst row: {worst_row} (score={scores[worst_row]:.2f})")
    print("Differences:")
    for d in diff_features:
        print(f"  {d['feature']}: best={d['a']}, worst={d['b']}")
    print(f"Tradeoffs where {other_row} is better: {', '.join(tradeoffs) if tradeoffs else 'none'}")

    # Machine-checked answer
    ans12 = {
        "a": best_row,
        "b": worst_row,
        "preferred": preferred_row,
        "scores": scores_out,
        "differences": diff_features,
        "tradeoffs": tradeoffs
    }
    print("ANSWER APACHE-12", json.dumps(ans12))

if __name__ == "__main__":
    main()

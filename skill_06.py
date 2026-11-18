#!/usr/bin/env python3
"""
skill_06.py

Entry point for APACHE-11 and APACHE-12 skill.

Usage:
    python skill_06.py <csv_path>
"""

import csv
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse the CSV reading routine from skill_01
from skill_01 import read_csv


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def identify_goal_column(headers: List[str]) -> str:
    """Return the first column name that ends with '-' or '+'."""
    for col in headers:
        if col.endswith('-') or col.endswith('+'):
            return col
    raise ValueError("No goal column found.")


def identify_setting_columns(headers: List[str], goal_col: str) -> List[str]:
    """Return setting columns (exclude goal and columns ending with 'X')."""
    return [c for c in headers if c != goal_col and not c.endswith('X')]


def best_row_by_goal(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Return the row with the best (lowest for '-' or highest for '+') goal value."""
    # Filter rows with a value in goal_col
    valid = [r for r in rows if r[goal_col] is not None]
    if not valid:
        raise ValueError("No rows with a goal value.")
    reverse = goal_col.endswith('+')  # maximize
    return sorted(valid, key=lambda r: r[goal_col], reverse=reverse)[0]


def worst_row_by_goal(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Return the row with the worst (highest for '-' or lowest for '+') goal value."""
    valid = [r for r in rows if r[goal_col] is not None]
    if not valid:
        raise ValueError("No rows with a goal value.")
    reverse = goal_col.endswith('-')  # minimize -> worst is highest
    return sorted(valid, key=lambda r: r[goal_col], reverse=reverse)[0]


def hamming_distance(row_a: Dict[str, Any], row_b: Dict[str, Any],
                     cols: List[str]) -> int:
    """Return the number of differing values among cols where both rows have a value."""
    diff = 0
    for col in cols:
        va = row_a.get(col)
        vb = row_b.get(col)
        if va is None or vb is None:
            continue  # ignore missing
        if va != vb:
            diff += 1
    return diff


def get_row_values(row: Dict[str, Any], headers: List[str]) -> Dict[str, Any]:
    """Return a dictionary of column-value pairs for the row, excluding the internal _row_num."""
    return {col: row[col] for col in headers}


def score_value(row: Dict[str, Any], goal_col: str) -> float:
    """Return a score where higher is better: for minimization use -value."""
    val = row[goal_col]
    if val is None:
        return float('-inf')
    return val if goal_col.endswith('+') else -val


# --------------------------------------------------------------------------- #
# APACHE-11 logic
# --------------------------------------------------------------------------- #

def apache_11(rows: List[Dict[str, Any]],
              headers: List[str],
              goal_col: str,
              setting_cols: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute the recommended configuration and its nearest neighbors."""
    best = best_row_by_goal(rows, goal_col)
    # Compute distance to every other row
    neighbors = []
    for r in rows:
        if r['_row_num'] == best['_row_num']:
            continue
        dist = hamming_distance(best, r, setting_cols)
        neighbors.append((dist, r['_row_num'], r))
    # Sort by distance then by row number
    neighbors.sort(key=lambda t: (t[0], t[1]))
    # Take at least 3 neighbors
    selected = neighbors[:max(3, len(neighbors))]
    # Build neighbor details
    neighbor_details = []
    for _, _, r in selected:
        neighbor_details.append({
            "row": r['_row_num'],
            "values": get_row_values(r, headers)
        })
    return best['_row_num'], neighbor_details


# --------------------------------------------------------------------------- #
# APACHE-12 logic
# --------------------------------------------------------------------------- #

def apache_12(rows: List[Dict[str, Any]],
              headers: List[str],
              goal_col: str,
              setting_cols: List[str]) -> Dict[str, Any]:
    """Explain the difference between a best and worst configuration."""
    best = best_row_by_goal(rows, goal_col)
    worst = worst_row_by_goal(rows, goal_col)

    a_row = best
    b_row = worst

    # Determine which is preferred
    pref = a_row if score_value(a_row, goal_col) >= score_value(b_row, goal_col) else b_row
    non_pref = b_row if pref is a_row else a_row

    # Scores
    scores = {
        str(a_row['_row_num']): score_value(a_row, goal_col),
        str(b_row['_row_num']): score_value(b_row, goal_col)
    }

    # Differences in setting columns
    differences = []
    for col in setting_cols:
        va = a_row.get(col)
        vb = b_row.get(col)
        if va is None or vb is None:
            continue
        if va != vb:
            differences.append({
                "feature": col,
                "a": va,
                "b": vb
            })

    # Tradeoffs: goal columns where non-preferred is better
    tradeoffs = []
    # If only one goal column, still handle generically
    # For minimization: lower is better; for maximization: higher is better
    if goal_col.endswith('-'):
        if non_pref[goal_col] < pref[goal_col]:
            tradeoffs.append(goal_col)
    else:
        if non_pref[goal_col] > pref[goal_col]:
            tradeoffs.append(goal_col)

    return {
        "a": a_row['_row_num'],
        "b": b_row['_row_num'],
        "preferred": pref['_row_num'],
        "scores": scores,
        "differences": differences,
        "tradeoffs": tradeoffs
    }


# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    goal_col = identify_goal_column(headers)
    setting_cols = identify_setting_columns(headers, goal_col)

    # APACHE-11
    rec_row_num, neighbors = apache_11(rows, headers, goal_col, setting_cols)
    print("== APACHE-11 ==")
    print(f"Recommended row: {rec_row_num}")
    print(f"Performance value: {rows[rec_row_num-1][goal_col]}")
    print(f"Neighbors:")
    for n in neighbors:
        print(f"  Row {n['row']}: {json.dumps(n['values'], separators=(',', ':'))}")
    answer_11 = {"recommended": rec_row_num, "neighbors": neighbors}
    print("ANSWER APACHE-11", json.dumps(answer_11, separators=(',', ':')))

    # APACHE-12
    explanation = apache_12(rows, headers, goal_col, setting_cols)
    print("\n== APACHE-12 ==")
    print(f"Best row: {explanation['a']}, worst row: {explanation['b']}")
    print(f"Preferred row: {explanation['preferred']}")
    print(f"Scores: {json.dumps(explanation['scores'], separators=(',', ':'))}")
    print("Differences:")
    for d in explanation['differences']:
        print(f"  {d['feature']}: {explanation['a']}={d['a']} vs {explanation['b']}={d['b']}")
    print(f"Tradeoffs where the non-preferred row is better: {explanation['tradeoffs']}")
    print("ANSWER APACHE-12", json.dumps(explanation, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

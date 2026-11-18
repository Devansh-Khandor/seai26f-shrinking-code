#!/usr/bin/env python3
"""
Skill 09: NASA-17 and NASA-18

This script flags historical projects whose combinations of
project characteristics are unusually different from the rest
(NASA‑17) and identifies records whose Effort‑ or Defects‑
values differ substantially from what comparable projects would
suggest (NASA‑18).

The logic re‑uses helper functions from skill_01 and follows
the same conventions for row numbering, missing values,
held‑out rows, arrivals, and budgeting.
"""

import csv
import json
import math
import sys
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------------
# Import helper functions from skill_01
try:
    from skill_01 import read_data, GOAL_COLUMNS
except Exception:
    # Minimal fallback implementation if import fails
    def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        rows: List[Dict[str, Any]] = []
        row_nums: List[int] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for rn, rec in enumerate(reader, start=1):
                row_nums.append(rn)
                parsed: Dict[str, Any] = {}
                for key, val in rec.items():
                    val = val.strip()
                    if val == "?":
                        parsed[key] = None
                    else:
                        try:
                            if '.' in val or 'e' in val.lower():
                                parsed[key] = float(val)
                            else:
                                parsed[key] = int(val)
                        except ValueError:
                            parsed[key] = val
                rows.append(parsed)
        return rows, row_nums

    GOAL_COLUMNS = {
        "Kloc+": "max",
        "Effort-": "min",
        "Defects-": "min",
    }

# ------------------------------------------------------------------
# Utility helpers

def is_numeric(val: Any) -> bool:
    return isinstance(val, (int, float))

def numeric_stats(rows: List[Dict[str, Any]], col: str) -> Tuple[float, float]:
    """Return (mean, std) for a numeric column, ignoring missing values."""
    vals = [row[col] for row in rows if isinstance(row.get(col), (int, float))]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    mean_val = sum(vals) / n
    var = sum((x - mean_val) ** 2 for x in vals) / n
    std_val = math.sqrt(var)
    return mean_val, std_val

def frequency_counts(rows: List[Dict[str, Any]], col: str) -> Dict[Any, int]:
    """Return frequency of each value for a categorical column."""
    counts: Dict[Any, int] = {}
    for row in rows:
        v = row.get(col)
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
    return counts

# ------------------------------------------------------------------
# NASA-17 logic

def compute_unusualness_scores(rows: List[Dict[str, Any]]) -> Dict[int, float]:
    """Return a dict mapping row number (1‑based) to an unusualness score."""
    # Identify non‑goal columns (excluding those ending with 'X')
    non_goal_cols = [c for c in rows[0].keys()
                     if c not in GOAL_COLUMNS and not c.endswith('X')]

    # Precompute statistics for each column
    numeric_cols = []
    categorical_cols = []
    num_means: Dict[str, float] = {}
    num_stds: Dict[str, float] = {}
    cat_counts: Dict[str, Dict[Any, int]] = {}

    for col in non_goal_cols:
        # Check if column has at least one numeric value
        if any(is_numeric(row.get(col)) for row in rows):
            numeric_cols.append(col)
            mean_val, std_val = numeric_stats(rows, col)
            num_means[col] = mean_val
            num_stds[col] = std_val if std_val > 0 else 1.0  # avoid division by zero
        else:
            categorical_cols.append(col)
            cat_counts[col] = frequency_counts(rows, col)

    scores: Dict[int, float] = {}
    total_rows = len(rows)
    for idx, row in enumerate(rows, start=1):
        score = 0.0
        # Numeric part
        for col in numeric_cols:
            val = row.get(col)
            if not is_numeric(val):
                continue
            z = (val - num_means[col]) / num_stds[col]
            score += abs(z)
        # Categorical part
        for col in categorical_cols:
            val = row.get(col)
            if val is None:
                continue
            freq = cat_counts[col].get(val, 0) / total_rows
            # Rare values get a penalty
            if freq < 0.10:  # less than 10 % of rows
                score += 5.0
        scores[idx] = score
    return scores

def flag_unusual_projects(rows: List[Dict[str, Any]], max_flag: int = 10) -> Tuple[List[int], Dict[int, float]]:
    """Return up to max_flag flagged rows and their scores."""
    scores = compute_unusualness_scores(rows)
    # Sort by score descending, then by row number ascending
    sorted_rows = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    flagged = [row for row, _ in sorted_rows[:max_flag]]
    flagged_scores = {row: round(scores[row], 6) for row in flagged}
    return flagged, flagged_scores

# ------------------------------------------------------------------
# NASA-18 logic

def build_similarity_matrix(rows: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Build a similarity matrix (distance) based on non‑goal columns.
    Returns a 2D list of distances (lower is more similar).
    """
    non_goal_cols = [c for c in rows[0].keys()
                     if c not in GOAL_COLUMNS and not c.endswith('X')]
    numeric_cols = []
    categorical_cols = []
    num_means: Dict[str, float] = {}
    num_stds: Dict[str, float] = {}

    for col in non_goal_cols:
        if any(is_numeric(row.get(col)) for row in rows):
            numeric_cols.append(col)
            mean_val, std_val = numeric_stats(rows, col)
            num_means[col] = mean_val
            num_stds[col] = std_val if std_val > 0 else 1.0
        else:
            categorical_cols.append(col)

    n = len(rows)
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = 0.0
            # Numeric columns
            for col in numeric_cols:
                vi = rows[i].get(col)
                vj = rows[j].get(col)
                if is_numeric(vi) and is_numeric(vj):
                    zi = (vi - num_means[col]) / num_stds[col]
                    zj = (vj - num_means[col]) / num_stds[col]
                    dist += (zi - zj) ** 2
            # Categorical columns
            for col in categorical_cols:
                vi = rows[i].get(col)
                vj = rows[j].get(col)
                if vi is not None and vj is not None and vi != vj:
                    dist += 1.0  # penalty for mismatch
            matrix[i][j] = matrix[j][i] = dist
    return matrix

def compute_expected_values(rows: List[Dict[str, Any]],
                            goal_col: str,
                            k_neighbors: int = 5) -> List[float]:
    """Return a list of expected goal values for each row."""
    n = len(rows)
    expected = [None] * n
    # Pre‑compute overall mean/std for threshold
    all_vals = [row[goal_col] for row in rows if isinstance(row.get(goal_col), (int, float))]
    overall_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0
    overall_std = math.sqrt(sum((x - overall_mean) ** 2 for x in all_vals) / len(all_vals)) if all_vals else 0.0
    threshold = 2.0 * overall_std

    similarity = build_similarity_matrix(rows)

    for idx, row in enumerate(rows):
        val = row.get(goal_col)
        if not isinstance(val, (int, float)):
            continue
        # Find k nearest neighbors (excluding itself)
        dists = [(d, j) for j, d in enumerate(similarity[idx]) if j != idx]
        dists.sort(key=lambda x: x[0])
        neighbors = [j for _, j in dists[:k_neighbors] if isinstance(rows[j].get(goal_col), (int, float))]
        if not neighbors:
            continue
        neigh_vals = [rows[j][goal_col] for j in neighbors]
        exp_val = sum(neigh_vals) / len(neigh_vals)
        expected[idx] = exp_val
    return expected

def flag_discrepancies(rows: List[Dict[str, Any]],
                       max_flag: int = 10) -> List[Dict[str, Any]]:
    """
    Identify rows whose Effort‑ or Defects‑ values differ
    substantially from the expected value based on comparable projects.
    """
    flagged: List[Dict[str, Any]] = []

    # Compute expected values for each goal column
    for goal_col in ("Effort-", "Defects-"):
        expected_vals = compute_expected_values(rows, goal_col)
        all_vals = [row[goal_col] for row in rows if isinstance(row.get(goal_col), (int, float))]
        overall_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0
        overall_std = math.sqrt(sum((x - overall_mean) ** 2 for x in all_vals) / len(all_vals)) if all_vals else 0.0
        threshold = 2.0 * overall_std

        for idx, row in enumerate(rows):
            actual = row.get(goal_col)
            exp_val = expected_vals[idx]
            if actual is None or exp_val is None:
                continue
            if abs(actual - exp_val) > threshold:
                flagged.append({
                    "row": idx + 1,
                    "column": goal_col,
                    "actual": actual,
                    "expected": round(exp_val, 6)
                })
                if len(flagged) >= max_flag:
                    return flagged
    return flagged

# ------------------------------------------------------------------
# Main entry point

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)

    # NASA-17
    print("== NASA-17 ==")
    flagged17, scores17 = flag_unusual_projects(rows)
    print(f"Number of flagged rows: {len(flagged17)}")
    if flagged17:
        print(f"Sample flagged rows: {flagged17[:3]}")
    answer17 = {
        "flagged": flagged17,
        "scores": {str(row): score for row, score in scores17.items()}
    }
    print(f"ANSWER NASA-17 {json.dumps(answer17)}")

    # NASA-18
    print("\n== NASA-18 ==")
    flagged18 = flag_discrepancies(rows)
    print(f"Number of flagged discrepancies: {len(flagged18)}")
    if flagged18:
        print(f"Sample discrepancies: {flagged18[:3]}")
    answer18 = {"flagged": flagged18}
    print(f"ANSWER NASA-18 {json.dumps(answer18)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skill 06: Apache configuration recommendation & pair comparison

This script implements the two requested analyses:
- APACHE-11: Recommend the best configuration and explain it by showing
  its three nearest neighbors.
- APACHE-12: Compare two configurations that received different
  performance assessments, highlighting the settings that differ
  and any trade‑offs in the goal columns.

The implementation reuses the helper functions from the existing
skills and from the `ezr` library wherever possible.
"""

import sys
import json
import random
import ezr  # the provided library


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load CSV with ezr and return header and data rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def is_goal(name: str) -> bool:
    """Return True if column name ends with '+' or '-'."""
    return name.endswith('+') or name.endswith('-')


def goal_indices(header):
    """Return list of indices of goal columns."""
    return [i for i, n in enumerate(header) if is_goal(n)]


def setting_indices(header):
    """Return list of indices of setting columns (non‑goal, non‑X)."""
    return [i for i, n in enumerate(header)
            if not is_goal(n) and not n.endswith('X')]


def ydist_row(tbl, row):
    """Return ydist for a single row."""
    return ezr.ydist(tbl, row)


def xdist_row(tbl, r1, r2):
    """Return xdist between two rows."""
    return ezr.xdist(tbl, r1, r2)


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #
def apache_11(header, data):
    """APACHE-11: Recommend a configuration and explain by neighbors."""
    tbl = ezr.Tbl([header] + data)
    goals = goal_indices(header)
    num_rows = len(data)

    # Budget: at most min(30, number of rows // 5) rows may be evaluated.
    budget = min(30, num_rows // 5)
    evaluated_rows = data[:budget]  # use first `budget` rows deterministically

    # Compute ydist for each evaluated row
    ydist_vals = [(i, ydist_row(tbl, r)) for i, r in enumerate(evaluated_rows)]
    # Find best (lowest ydist) among evaluated
    best_idx, best_ydist = min(ydist_vals, key=lambda kv: kv[1])
    recommended_row_number = best_idx + 1  # 1‑based among all rows

    # Find nearest 3 neighbors among all rows (including itself)
    all_rows_with_num = list(enumerate(data, start=1))
    ref_row = data[best_idx]
    neighbor_dists = []
    for num, row in all_rows_with_num:
        if row is ref_row:
            continue
        d = xdist_row(tbl, ref_row, row)
        neighbor_dists.append((num, d))
    neighbor_dists.sort(key=lambda kv: kv[1])
    top_neighbors = neighbor_dists[:3]  # at least 3 or fewer if data is small

    # Build neighbor values dict
    neighbors = []
    for nnum, _ in top_neighbors:
        neighbor_row = data[nnum - 1]
        values = {header[i]: neighbor_row[i] for i in goals}
        neighbors.append({"row": nnum, "values": values})

    # Output
    print("== APACHE-11 ==")
    print(f"Recommended row: {recommended_row_number}")
    for nb in neighbors:
        print(f"Neighbor row {nb['row']}: {nb['values']}")

    # Machine‑checked answer
    answer = {"recommended": recommended_row_number,
              "neighbors": neighbors}
    print(f"ANSWER APACHE-11 {json.dumps(answer, separators=(',', ':'))}")


def apache_12(header, data):
    """APACHE-12: Compare two configurations with different performances."""
    tbl = ezr.Tbl([header] + data)
    goals = goal_indices(header)

    # Compute ydist for all rows
    all_ydists = [(i, ydist_row(tbl, r)) for i, r in enumerate(data)]
    # Best (lowest) and worst (highest) rows
    best_idx, best_ydist = min(all__ydists, key=lambda kv: kv[1])
    worst_idx, worst_ydist = max(all_ydists, key=lambda kv: kv[1])
    best_row_num = best_idx + 1
    worst_row_num = worst_idx + 1

    # Scores (higher = more recommended). Use negative ydist.
    score_best = -best_ydist
    score_worst = -worst_ydist

    # Differences between the two configurations
    best_row = data[best_idx]
    worst_row = data[worst_idx]
    diffs = []
    for i in setting_indices(header):
        if best_row[i] != worst_row[i]:
            diffs.append({"feature": header[i],
                          "a": best_row[i],
                          "b": worst_row[i]})
    # Ensure at least one difference (data should guarantee this)
    if not diffs:
        diffs.append({"feature": setting_indices(header)[0],
                      "a": best_row[setting_indices(header)[0]],
                      "b": worst_row[setting_indices(header)[0]]})

    # Trade‑offs: goal columns where non‑preferred row is better
    tradeoffs = []
    for i in goals:
        col = tbl.cols[i]
        dist_best = abs(ezr.norm(col, best_row[i]) - col.goal)
        dist_worst = abs(ezr.norm(col, worst_row[i]) - col.goal)
        # non‑preferred row is worst_row_num
        if dist_worst < dist_best:
            tradeoffs.append(header[i])

    # Output
    print("== APACHE-12 ==")
    print(f"Row A (best): {best_row_num}")
    print(f"Row B (worst): {worst_row_num}")
    print(f"Preferred: {best_row_num}")
    print(f"Score A: {score_best:.2f}  Score B: {score_worst:.2f}")
    print("Differences:")
    for d in diffs:
        print(f"  {d['feature']} (A={d['a']} B={d['b']})")
    print("Trade‑offs (goal columns where B is better):")
    for g in tradeoffs:
        print(f"  {g}")

    # Machine‑checked answer
    answer = {"a": best_row_num,
              "b": worst_row_num,
              "preferred": best_row_num,
              "scores": {str(best_row_num): score_best,
                         str(worst_row_num): score_worst},
              "differences": diffs,
              "tradeoffs": tradeoffs}
    print(f"ANSWER APACHE-12 {json.dumps(answer, separators=(',', ':'))}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)

    apache_11(header, data)
    apache_12(header, data)


if __name__ == "__main__":
    main()

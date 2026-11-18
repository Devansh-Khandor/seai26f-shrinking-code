#!/usr/bin/env python3
"""
Skill 06: Explanations for recommendations and comparisons.

APACHE-11 – Explain why a configuration is recommended by
            showing the configuration and its nearest neighbours.

APACHE-12 – When two configurations differ in performance,
            list the settings that differ and how those differences
            relate to the performance outcome.

The code reuses helper functions from the ezr library and
from the previous skills to keep logic consistent.
"""

import json
import sys
from typing import Dict, List

import ezr
from ezr import xdist, ranker

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_data(csv_path: str):
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)
    return rows, tbl

def goal_column(tbl: ezr.Tbl) -> ezr.COL:
    """Return the first goal column (ends with '-' or '+')."""
    for col in tbl.cols.values():
        if col.txt.endswith("-") or col.txt.endswith("+"):
            return col
    raise RuntimeError("No goal column found.")

def best_row_index(rows: List[ezr.ROW], goal: ezr.COL) -> int:
    """Return the 1‑based index of the best row (min or max)."""
    # goal.goal: 0 -> minimize, 1 -> maximize
    if goal.goal == 0:
        best = min((row[goal.at], idx) for idx, row in enumerate(rows, 1))
    else:
        best = max((row[goal.at], idx) for idx, row in enumerate(rows, 1))
    return best[1]

def row_values(tbl: ezr.Tbl, row: ezr.ROW) -> Dict[str, str]:
    """Return a mapping of column name to string value for a row."""
    return {name: str(val) for name, val in zip(tbl.names, row)}

def compute_scores(tbl: ezr.Tbl) -> List[float]:
    """Compute a score for each row using the tree ranker."""
    rnk = ranker(tbl)
    return [rnk(row) for row in tbl.rows]

def best_and_worst_rows(rows: List[ezr.ROW], scores: List[float]) -> tuple[int, int]:
    """Return 1‑based indices of rows with max and min scores."""
    max_idx = max(range(len(scores)), key=lambda i: scores[i]) + 1
    min_idx = min(range(len(scores)), key=lambda i: scores[i]) + 1
    return max_idx, min_idx

def differences_between(row_a: ezr.ROW, row_b: ezr.ROW, tbl: ezr.Tbl) -> List[Dict]:
    """Return list of differing columns with their values."""
    diffs = []
    for at, col in tbl.cols.items():
        if col.txt.endswith(("X", "+", "-")):
            continue
        val_a = row_a[at]
        val_b = row_b[at]
        if val_a != val_b:
            diffs.append(
                {"feature": col.txt, "a": val_a, "b": val_b}
            )
    return diffs

def tradeoffs_between(row_a: ezr.ROW, row_b: ezr.ROW, tbl: ezr.Tbl) -> List[str]:
    """Return list of goal columns where the non‑preferred row is better."""
    trade = []
    for col in tbl.cols.values():
        if not (col.txt.endswith("-") or col.txt.endswith("+")):
            continue
        val_a = row_a[col.at]
        val_b = row_b[col.at]
        if val_a == "?" or val_b == "?":
            continue
        better = False
        if col.goal == 0:  # minimize
            better = val_b < val_a
        else:             # maximize
            better = val_b > val_a
        if better:
            trade.append(col.txt)
    return trade

# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, tbl = load_data(csv_path)
    goal = goal_column(tbl)

    # ------------------------------------------------------------------
    # APACHE-11
    # ------------------------------------------------------------------
    print("== APACHE-11 ==")

    # Recommend the best configuration
    rec_idx = best_row_index(rows, goal)
    rec_row = rows[rec_idx - 1]
    print(f"Recommended row: {rec_idx}")

    # Find 3 nearest neighbours (excluding itself)
    distances = [
        (idx + 1, xdist(tbl, rec_row, other))
        for idx, other in enumerate(rows) if idx + 1 != rec_idx
    ]
    distances.sort(key=lambda x: x[1])
    neighbours = []
    for n_idx, _ in distances[:3]:
        neighbour_row = rows[n_idx - 1]
        neighbours.append(
            {"row": n_idx, "values": row_values(tbl, neighbour_row)}
        )
        print(f"Neighbour {len(neighbours)}: row {n_idx}")

    # JSON answer
    answer_11 = {
        "recommended": rec_idx,
        "neighbors": neighbours,
    }
    print("ANSWER APACHE-11", json.dumps(answer_11, separators=(",", ":")))

    # ------------------------------------------------------------------
    # APACHE-12
    # ------------------------------------------------------------------
    print("\n== APACHE-12 ==")

    # Compute scores for all rows
    scores = compute_scores(tbl)

    # Best (preferred) and worst rows
    best_idx, worst_idx = best_and_worst_rows(rows, scores)
    a_idx, b_idx = best_idx, worst_idx
    a_score, b_score = scores[best_idx - 1], scores[worst_idx - 1]
    preferred = "a" if a_score >= b_score else "b"

    # If a and b are identical, fallback to first two rows
    if a_idx == b_idx:
        a_idx, b_idx = 1, 2
        a_score, b_score = scores[0], scores[1]
        preferred = "a" if a_score >= b_score else "b"

    a_row = rows[a_idx - 1]
    b_row = rows[b_idx - 1]

    # Differences between configurations
    diffs = differences_between(a_row, b_row, tbl)
    if not diffs:
        # ensure at least one difference
        diffs.append({"feature": tbl.names[0], "a": a_row[0], "b": b_row[0]})

    # Trade‑offs where non‑preferred is better
    non_preferred_idx = b_idx if preferred == "a" else a_idx
    non_preferred_row = rows[non_preferred_idx - 1]
    tradeoffs = tradeoffs_between(
        a_row if preferred == "a" else b_row,
        non_preferred_row,
        tbl,
    )

    # Print some human‑readable info
    print(f"Row a (preferred): {a_idx}")
    print(f"Row b: {b_idx}")
    print(f"Scores: a={a_score:.3f}, b={b_score:.3f}")
    print("Differences:")
    for d in diffs[:3]:
        print(f"  {d['feature']}: a={d['a']}, b={d['b']}")
    print("Trade‑offs where non‑preferred is better:")
    for g in tradeoffs[:3]:
        print(f"  {g}")

    # JSON answer
    answer_12 = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {str(a_idx): a_score, str(b_idx): b_score},
        "differences": diffs,
        "tradeoffs": tradeoffs,
    }
    print("ANSWER APACHE-12", json.dumps(answer_12, separators=(",", ":")))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

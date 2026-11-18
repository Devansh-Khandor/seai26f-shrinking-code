#!/usr/bin/env python3
"""
Skill 07: Active configuration discovery.

APACHE-13 – Discover a low‑performance configuration while evaluating as few
rows as possible.

APACHE-14 – Recommend an order for evaluating configurations so that
promising settings are found early, using only the goal values of rows
that have already been evaluated.
"""

import json
import sys
from typing import List, Tuple

import ezr

# ----------------------------------------------------------------------
# Helpers ---------------------------------------------------------------
# ----------------------------------------------------------------------
def goal_col(tbl: ezr.Tbl) -> ezr.COL:
    """Return the goal column (ends with '-' or '+')."""
    for col in tbl.cols.values():
        if col.txt.endswith("-") or col.txt.endswith("+"):
            return col
    raise RuntimeError("No goal column found.")


def normalized_sum(row: ezr.ROW, tbl: ezr.Tbl) -> float:
    """
    Sum of normalized values for all X columns of a row.
    Normalization uses the column statistics stored in tbl.
    """
    s = 0.0
    for col in tbl.x:
        val = row[col.at]
        if val == "?":
            # Treat missing as 0.5 (neutral) – this case should rarely happen
            s += 0.5
        else:
            s += ezr.norm(col, val)
    return s


def first_candidate(tbl: ezr.Tbl) -> int:
    """Return the 1‑based row number with the smallest normalized sum."""
    best_idx = None
    best_score = float("inf")
    for idx, row in enumerate(tbl.rows, start=1):
        score = normalized_sum(row, tbl)
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def build_tree_on_evaluated(tbl: ezr.Tbl, eval_rows: List[int]) -> ezr.NODE:
    """Return a decision tree built only on the rows that have already been evaluated."""
    rows = [tbl.rows[i - 1] for i in eval_rows]
    return ezr.tree(tbl, rows)


def best_predicted_row(
    tbl: ezr.Tbl, tree: ezr.NODE, remaining: List[int]
) -> int:
    """
    Among remaining rows, return the 1‑based index of the row with the
    lowest leaf mean ydist according to the given tree.
    """
    best_idx = None
    best_mu = float("inf")
    for idx in remaining:
        leaf = ezr.leaf(tree, tbl.rows[idx - 1])
        mu = leaf.mu
        if mu < best_mu:
            best_mu = mu
            best_idx = idx
    return best_idx


def generate_order(tbl: ezr.Tbl, budget: int) -> List[int]:
    """
    Generate an evaluation order of length <= budget by iteratively
    building a tree on already evaluated rows and selecting the best
    predicted remaining row.
    """
    evaluated: List[int] = []
    remaining = list(range(1, len(tbl.rows) + 1))

    # First selection – heuristic based on feature norm
    first = first_candidate(tbl)
    evaluated.append(first)
    remaining.remove(first)

    while len(evaluated) < budget and remaining:
        # Build tree on currently evaluated rows
        tree = build_tree_on_evaluated(tbl, evaluated)
        # Pick next row with lowest predicted leaf mean
        nxt = best_predicted_row(tbl, tree, remaining)
        if nxt is None:
            break
        evaluated.append(nxt)
        remaining.remove(nxt)

    return evaluated


def evaluate_rows(order: List[int], tbl: ezr.Tbl, gcol: ezr.COL) -> Tuple[List[int], int]:
    """
    Evaluate rows in the given order.
    Returns (evaluated list, best row number).
    """
    best_row = None
    best_val = None
    for idx in order:
        val = tbl.rows[idx - 1][gcol.at]
        if val == "?":
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_row = idx
    if best_row is None:
        # fallback – pick first evaluated row that has a value
        for idx in order:
            val = tbl.rows[idx - 1][gcol.at]
            if val != "?":
                best_row = idx
                break
    return order, best_row


# ----------------------------------------------------------------------
# Main -------------------------------------------------------------
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)
    gcol = goal_col(tbl)

    # Budget: at most min(30, rows//5)
    budget = min(30, len(rows) // 5)

    # --------- APACHE-13 ------------------------------------------------
    order_13 = generate_order(tbl, budget)
    evaluated_13, best_13 = evaluate_rows(order_13, tbl, gcol)

    print("== APACHE-13 ==")
    print(f"Evaluated rows: {evaluated_13}")
    if best_13 is not None:
        best_val = tbl.rows[best_13 - 1][gcol.at]
        print(f"Best row: {best_13} with goal {best_val}")
    else:
        print("Best row: None")
    print(f"Budget used: {len(evaluated_13)}")
    ans_13 = {"evaluated": evaluated_13, "best": best_13}
    print("ANSWER APACHE-13", json.dumps(ans_13, separators=(",", ":")))

    # --------- APACHE-14 ------------------------------------------------
    order_14 = generate_order(tbl, budget)

    print("\n== APACHE-14 ==")
    print(f"Evaluation order: {order_14}")
    print(f"Number of rows in order: {len(order_14)}")
    ans_14 = {"order": order_14}
    print("ANSWER APACHE-14", json.dumps(ans_14, separators=(",", ":")))


# ----------------------------------------------------------------------
# Entry point ----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
Skill 07 – Apache performance analytics (APACHE-13 & APACHE-14)

Usage:
    python skill_07.py <csv_path>
"""

import json
import random
import sys

import ezr


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def find_goal_min_col(tbl: ezr.TBL):
    """
    Return the column that should be minimized.
    Prefer a column ending with '-' (performance metric to minimize).
    If none, return the first goal column.
    """
    for c in tbl.cols.values():
        if c.txt.endswith("-"):
            return c
    # fallback
    for c in tbl.cols.values():
        if c.txt.endswith(("+", "-")):
            return c
    raise RuntimeError("No goal column found.")


def get_candidate_indices(tbl: ezr.TBL):
    """
    Return list of 1‑based row indices that have all goal values present.
    Rows with '?' in any goal column are excluded because they cannot be evaluated.
    """
    goal_cols = [c for c in tbl.cols.values() if c.txt.endswith(("+", "-"))]
    candidates = []
    for i, row in enumerate(tbl.rows, start=1):
        if all(row[c.at] != "?" for c in goal_cols):
            candidates.append(i)
    return candidates


def build_ranker(tbl: ezr.TBL, rows):
    """
    Build a tree‑based ranker from the evaluated rows.
    """
    node = ezr.tree(tbl, rows)
    return lambda r: ezr.leaf(node, r).mu


def select_next(rows, candidate_indices, evaluated_set, tbl, rank_func):
    """
    Return the 1‑based index of the next candidate with the lowest
    predicted performance (lowest leaf µ).
    """
    best_idx = None
    best_val = float("inf")
    for idx in candidate_indices:
        if idx in evaluated_set:
            continue
        row = tbl.rows[idx - 1]
        val = rank_func(row)
        if val < best_val:
            best_val = val
            best_idx = idx
    return best_idx


def sequential_evaluation(tbl: ezr.TBL, budget):
    """
    Perform a greedy sequential evaluation to discover a low
    performance configuration with minimal budget.
    Returns the list of evaluated 1‑based indices (in order)
    and the index of the best evaluated row.
    """
    # Ensure reproducible order
    random.seed(ezr.the.Seed)

    candidates = get_candidate_indices(tbl)
    if not candidates:
        raise RuntimeError("No evaluable rows.")

    evaluated = []
    evaluated_set = set()

    # Start with a random candidate
    first = random.choice(candidates)
    evaluated.append(first)
    evaluated_set.add(first)

    # If budget is already met
    while len(evaluated) < budget:
        # Build ranker on already evaluated rows
        eval_rows = [tbl.rows[i - 1] for i in evaluated]
        rank = build_ranker(tbl, eval_rows)

        # Select next best predicted
        nxt = select_next(tbl.rows, candidates, evaluated_set, tbl, rank)
        if nxt is None:
            break  # no more candidates
        evaluated.append(nxt)
        evaluated_set.add(nxt)

    # Determine best evaluated row (lowest goal value)
    goal_col = find_goal_min_col(tbl)
    best_idx = min(evaluated, key=lambda i: tbl.rows[i - 1][goal_col.at])
    return evaluated, best_idx


def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))

    total_rows = len(tbl.rows)
    budget = min(30, total_rows // 5)

    # --------------------------------------------------------------------
    # APACHE-13: sequential discovery of a low‑performance row
    # --------------------------------------------------------------------
    evaluated, best_row = sequential_evaluation(tbl, budget)

    print("== APACHE-13 ==")
    print(f"Evaluated rows ({len(evaluated)}): {evaluated}")
    goal_col = find_goal_min_col(tbl)
    best_val = tbl.rows[best_row - 1][goal_col.at]
    print(f"Best row: {best_row} (value={best_val})")
    print(f"ANSWER APACHE-13 {json.dumps({'evaluated': evaluated, 'best': best_row})}")

    # --------------------------------------------------------------------
    # APACHE-14: recommended evaluation order
    # --------------------------------------------------------------------
    # The order is exactly the sequence we used for APACHE‑13
    order = evaluated

    print("\n== APACHE-14 ==")
    print(f"Recommended order ({len(order)} rows): {order}")
    print(f"ANSWER APACHE-14 {json.dumps({'order': order})}")


if __name__ == "__main__":
    main()

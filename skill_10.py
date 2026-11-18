#!/usr/bin/env python3
"""
Skill 10: Comparative evaluation of configuration selection and prediction.

APACHE-19 – Compare our configuration‑selection system against random selection
under the same benchmarking budget, and show which approach discovers
better‑performing configurations.

APACHE-20 – Compare our performance predictions against a simple
historical‑average estimate and show whether our system provides a
meaningful improvement on previously unseen configurations.
"""

import json
import math
import random
import sys
from typing import List, Dict, Tuple

import ezr
from ezr import leaf, tree, o, xdist, ranker

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def goal_col(tbl: ezr.Tbl) -> ezr.COL:
    """Return the first goal column (ends with '-' or '+')."""
    for col in tbl.cols.values():
        if col.txt.endswith("-") or col.txt.endswith("+"):
            return col
    raise RuntimeError("No goal column found.")

def heldout_indices(rows: List[ezr.ROW]) -> List[int]:
    """Return 1‑based indices of rows whose number is divisible by 5."""
    return [i + 1 for i, _ in enumerate(rows) if (i + 1) % 5 == 0]

def budget_limit(rows: List[ezr.ROW]) -> int:
    """Maximum number of rows that may be evaluated."""
    return min(30, len(rows) // 5)

# ----------------------------------------------------------------------
# APACHE-19: selection comparison
# ----------------------------------------------------------------------
def select_ours(tbl: ezr.Tbl, budget: int) -> List[int]:
    """
    Return an ordered list of row indices chosen by our strategy.
    Deterministic: uses the same algorithm as skill_07 generate_order.
    """
    # First candidate (smallest normalized sum)
    def first_candidate() -> int:
        best_idx = None
        best_score = float("inf")
        for idx, row in enumerate(tbl.rows, start=1):
            score = sum(
                ezr.norm(col, row[col.at]) if col not in tbl.x else 0
                for col in tbl.x
            )
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def build_tree(eval_rows: List[int]) -> ezr.NODE:
        return tree(tbl, [tbl.rows[i - 1] for i in eval_rows])

    def best_predicted(eval_rows: List[int], remaining: List[int]) -> int:
        tree_node = build_tree(eval_rows)
        best_idx = None
        best_mu = float("inf")
        for idx in remaining:
            leaf_node = leaf(tree_node, tbl.rows[idx - 1])
            if leaf_node.mu < best_mu:
                best_mu = leaf_node.mu
                best_idx = idx
        return best_idx

    evaluated: List[int] = []
    remaining = list(range(1, len(tbl.rows) + 1))
    first = first_candidate()
    evaluated.append(first)
    remaining.remove(first)

    while len(evaluated) < budget and remaining:
        nxt = best_predicted(evaluated, remaining)
        if nxt is None:
            break
        evaluated.append(nxt)
        remaining.remove(nxt)
    return evaluated

def best_overall(row_indices: List[int], gcol: ezr.COL) -> Tuple[int, float]:
    """Return the 1‑based index with the best (lowest) goal value."""
    best_idx = None
    best_val = None
    for idx in row_indices:
        val = row(gcol, idx)
        if val == "?":
            continue
        v = float(val)
        if best_val is None or v < best_val:
            best_val = v
            best_idx = idx
    return best_idx, best_val

def row(col: ezr.COL, idx: int) -> ezr.ATOM:
    """Get cell value at given 1‑based index."""
    return col.tbl.rows[idx - 1][col.at]

def apache_19(rows: List[ezr.ROW], tbl: ezr.Tbl, gcol: ezr.COL) -> None:
    print("== APACHE-19 ==")
    budget = budget_limit(rows)
    repeats = 20
    ours_best: List[int] = []
    random_best: List[int] = []

    all_indices = list(range(1, len(rows) + 1))

    for r in range(repeats):
        # Random selection
        rnd = random.Random(r + 100)  # deterministic per repeat
        rand_sel = rnd.sample(all_indices, budget)
        idx, _ = best_overall(rand_sel, gcol)
        random_best.append(idx)

        # Our selection
        our_sel = select_ours(tbl, budget)
        idx, _ = best_overall(our_sel, gcol)
        ours_best.append(idx)

    # Determine verdict
    ours_wins = 0
    random_wins = 0
    for o, r in zip(ours_best, random_best):
        o_val = row(gcol, o)
        r_val = row(gcol, r)
        if o_val == "?" or r_val == "?":
            continue
        if float(o_val) < float(r_val):
            ours_wins += 1
        elif float(r_val) < float(o_val):
            random_wins += 1

    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Print at least three result lines
    print(f"Budget: {budget}")
    print(f"Random best rows: {random_best}")
    print(f"Our best rows: {ours_best}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict,
    }
    print("ANSWER APACHE-19", json.dumps(answer, separators=(",", ":")))

# ----------------------------------------------------------------------
# APACHE-20: prediction comparison
# ----------------------------------------------------------------------
def build_tree_with_mean(tbl: ezr.Tbl, rows: List[ezr.ROW]) -> ezr.NODE:
    """Build a tree on rows and compute mean goal value for each leaf."""
    node = tree(tbl, rows)

    def compute_leaf_means(n: ezr.NODE) -> None:
        if n.kids:
            for kid in n.kids:
                compute_leaf_means(kid)
        else:
            vals = [row[goal_col(tbl).at] for row in n.rows if row[goal_col(tbl).at] != "?"]
            if vals:
                n.mu_perf = sum(float(v) for v in vals) / len(vals)
            else:
                n.mu_perf = float("nan")

    compute_leaf_means(node)
    return node

def apache_20(rows: List[ezr.ROW], tbl: ezr.Tbl, gcol: ezr.COL) -> None:
    print("== APACHE-20 ==")
    held_indices = heldout_indices(rows)
    training_rows = [row for idx, row in enumerate(rows, start=1) if idx not in held_indices]

    # Baseline mean of non-held-out rows
    baseline_vals = [float(r[gcol.at]) for r in training_rows if r[gcol.at] != "?"]
    baseline = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    # Model: tree with leaf means
    tree_node = build_tree_with_mean(tbl, training_rows)

    predictions: Dict[int, float] = {}
    mae_model = 0.0
    mae_base = 0.0
    count = 0

    for idx in held_indices:
        row = rows[idx - 1]
        actual = row[gcol.at]
        if actual == "?":
            continue
        act = float(actual)
        leaf_node = leaf(tree_node, row)
        pred = getattr(leaf_node, "mu_perf", baseline)
        if math.isnan(pred):
            pred = baseline
        predictions[idx] = round(pred, 6)
        mae_model += abs(pred - act)
        mae_base += abs(baseline - act)
        count += 1

    mae_model = mae_model / count if count else 0.0
    mae_base = mae_base / count if count else 0.0

    # Determine verdict
    if mae_model < mae_base:
        verdict = "better"
    elif mae_base < mae_model:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Print at least three lines
    print(f"Baseline mean: {baseline:.4f}")
    print(f"MAE baseline: {mae_base:.4f}")
    print(f"MAE model: {mae_model:.4f}")

    # Print predictions for first few rows
    for idx in sorted(predictions)[:3]:
        print(f"Row {idx}: predicted={predictions[idx]:.4f} actual={rows[idx-1][gcol.at]}")

    answer = {
        "predictions": {str(k): v for k, v in predictions.items()},
        "baseline": round(baseline, 6),
        "mae_model": round(mae_model, 6),
        "mae_baseline": round(mae_base, 6),
        "verdict": verdict,
    }
    print("ANSWER APACHE-20", json.dumps(answer, separators=(",", ":")))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)
    gcol = goal_col(tbl)
    tbl.rows = rows  # ensure tbl has reference for addRow
    tbl.cols = {at: col for at, col in tbl.cols.items()}

    apache_19(rows, tbl, gcol)
    apache_20(rows, tbl, gcol)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

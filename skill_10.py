#!/usr/bin/env python3
"""
Skill 10: Apache configuration selection & prediction comparison

This script implements the requested analyses:
- APACHE-19: Compare a tree‑based selection strategy against random selection
  under a fixed benchmarking budget.
- APACHE-20: Compare predictions from that tree against a simple historical
  mean baseline for held‑out rows.

The script follows the conventions used in the earlier skills:
  * rows are 1‑based (excluding the header row)
  * goal columns end with '+' (maximize) or '-' (minimize)
  * missing values are indicated by '?'
  * held‑out rows are those whose row number is divisible by 5
"""

import sys
import json
import random
import math
import ezr

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Load CSV using ezr and return header and data rows."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data

def goal_indices(header):
    """Return indices of goal columns (+/-)."""
    return [i for i, n in enumerate(header) if n.endswith(('+', '-'))]

def is_goal(name: str) -> bool:
    return name.endswith('+') or name.endswith('-')

def goal_value(row, idx):
    """Return numeric goal value or None if missing."""
    val = row[idx]
    return None if val == '?' else val

def is_better(a: float, b: float, minimize: bool) -> bool:
    """Return True if a is better than b under the given goal direction."""
    if minimize:
        return a < b
    return a > b

def best_of_rows(rows, idx, minimize: bool):
    """Return (best_row_index, best_value) among given row numbers (1‑based)."""
    best = None
    best_val = None
    for r in rows:
        val = goal_value(rows[r-1], idx)
        if val is None:
            continue
        if best_val is None or is_better(val, best_val, minimize):
            best = r
            best_val = val
    return best, best_val

# --------------------------------------------------------------------------- #
# APACHE‑19: selection comparison
# --------------------------------------------------------------------------- #
def build_ours_order(header, data, budget):
    """
    Return an ordered list of row numbers (1‑based) that will be evaluated
    by the tree‑based selection strategy.  The algorithm starts with a few
    known rows, then repeatedly builds a ranker on evaluated rows and
    selects the remaining row with the lowest predicted goal.
    """
    # Identify rows with known goal values
    goal_idx = goal_indices(header)[0]
    known_rows = [i+1 for i, r in enumerate(data) if goal_value(r, goal_idx) is not None]

    # Initial known set: first few known rows (deterministic)
    initial = min(5, len(known_rows))
    evaluated = known_rows[:initial]
    remaining = set(known_rows[initial:])

    # Keep all rows in the ordering; those without known goals are still
    # evaluated when they get selected via the ranking process.
    all_rows = list(range(1, len(data)+1))
    order_set = set(evaluated)

    # Add the rest (in appearance order)
    for r in all_rows:
        if r not in order_set:
            order_set.add(r)

    # Iterative selection based on predictions
    while len(evaluated) < budget and remaining:
        eval_rows = [data[i-1] for i in evaluated]
        tbl = ezr.Tbl([header] + eval_rows)
        rank = ezr.ranker(tbl)

        # Find remaining known row with lowest predicted goal
        best_candidate = None
        best_score = None
        for r in sorted(remaining):
            row = data[r-1]
            score = rank(row)
            # For a minimization goal, lower score is better
            if best_score is None or score < best_score or (score == best_score and r < best_candidate):
                best_candidate = r
                best_score = score

        evaluated.append(best_candidate)
        remaining.remove(best_candidate)

    return evaluated

def evaluate_ours(header, data, budget):
    """Return best row number found by the tree‑based selection."""
    ordered = build_ours_order(header, data, budget)
    goal_idx = goal_indices(header)[0]
    # Find best among evaluated rows
    best_row, best_val = best_of_rows([r for r in ordered], goal_idx, minimize=True)
    return best_row, best_val

def evaluate_random(header, data, budget, rng):
    """Return best row number found by randomly sampling rows."""
    all_indices = list(range(1, len(data)+1))
    sample = rng.sample(all_indices, budget)
    goal_idx = goal_indices(header)[0]
    best_row, best_val = best_of_rows(sample, goal_idx, minimize=True)
    return best_row, best_val

# --------------------------------------------------------------------------- #
# APACHE‑20: prediction vs baseline
# --------------------------------------------------------------------------- #
def build_prediction_tree(header, data, training_rows):
    """Return a tree built on training_rows and a mapping leaf->mean goal."""
    goal_idx = goal_indices(header)[0]
    tbl = ezr.Tbl([header] + training_rows)
    tree = ezr.tree(tbl, training_rows)

    # Compute mean goal for each leaf
    leaf_means = {}
    def traverse(node):
        if not node.kids:
            # leaf: compute mean goal of its rows
            vals = [goal_value(r, goal_idx) for r in node.rows]
            vals = [v for v in vals if v is not None]
            mean_val = sum(vals)/len(vals) if vals else None
            leaf_means[node] = mean_val
        else:
            for kid in node.kids:
                traverse(kid)
    traverse(tree)
    return tree, leaf_means

def predict_from_leaf(tree, leaf_means, row):
    """Return the mean goal for the leaf containing ROW."""
    node = ezr.leaf(tree, row)
    return leaf_means.get(node)

def compute_mae(predictions, actuals):
    """Compute mean absolute error given dict predictions and actual values."""
    if not predictions:
        return 0.0
    total = 0.0
    n = 0
    for k, pred in predictions.items():
        act = actuals.get(k)
        if act is None:
            continue
        total += abs(pred - act)
        n += 1
    return total/n if n else 0.0

# --------------------------------------------------------------------------- #
# Main analysis functions
# --------------------------------------------------------------------------- #
def apache_19(header, data):
    """APACHE-19: compare selection strategy against random."""
    budget = min(30, len(data)//5)
    repeats = 20
    ours_best = []
    random_best = []
    goal_idx = goal_indices(header)[0]

    for rep in range(repeats):
        rng = random.Random(rep + 1000)  # deterministic per repeat
        # OURS
        ours_row, ours_val = evaluate_ours(header, data, budget)
        ours_best.append((ours_row, ours_val))

        # RANDOM
        rand_row, rand_val = evaluate_random(header, data, budget, rng)
        random_best.append((rand_row, rand_val))

    # Compute average best performances
    ours_vals = [v for _, v in ours_best if v is not None]
    rand_vals = [v for _, v in random_best if v is not None]
    mean_ours = sum(ours_vals)/len(ours_vals) if ours_vals else 0.0
    mean_rand = sum(rand_vals)/len(rand_vals) if rand_vals else 0.0

    # Decide verdict
    eps = 1e-8
    if mean_ours + eps < mean_rand:
        verdict = "ours better"
    elif mean_rand + eps < mean_ours:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Prepare output
    print("== APACHE-19 ==")
    print(f"Budget per repeat: {budget}")
    print(f"Repetitions: {repeats}")
    print(f"Average best performance ours: {round(mean_ours,2)}")
    print(f"Average best performance random: {round(mean_rand,2)}")
    print(f"Verdict: {verdict}")

    # JSON answer
    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": [row for row, _ in ours_best],
        "random": [row for row, _ in random_best],
        "verdict": verdict
    }
    print(f"ANSWER APACHE-19 {json.dumps(answer, separators=(',', ':'))}")

def apache_20(header, data):
    """APACHE-20: prediction vs baseline for held‑out rows."""
    goal_idx = goal_indices(header)[0]
    # Identify held‑out and training rows
    held_out_indices = [i+1 for i in range(len(data)) if (i+1) % 5 == 0]
    training_rows = [row for i, row in enumerate(data, start=1) if i not in held_out_indices]

    # Build tree and leaf means on training rows
    tree, leaf_means = build_prediction_tree(header, data, training_rows)

    # Baseline: mean of training goal values
    train_goals = [goal_value(r, goal_idx) for r in training_rows if goal_value(r, goal_idx) is not None]
    baseline_mean = sum(train_goals)/len(train_goals) if train_goals else 0.0

    # Predictions for held‑out rows
    predictions = {}
    actuals = {}
    for idx in held_out_indices:
        row = data[idx-1]
        act = goal_value(row, goal_idx)
        if act is None:
            continue
        act_val = act
        pred_val = predict_from_leaf(tree, leaf_means, row)
        # If leaf mean is None (unlikely), fallback to baseline
        if pred_val is None:
            pred_val = baseline_mean
        predictions[str(idx)] = round(pred_val, 4)
        actuals[str(idx)] = act_val

    mae_model = compute_mae(predictions, actuals)
    mae_baseline = compute_mae({k: baseline_mean for k in predictions}, actuals)

    # Verdict
    eps = 1e-8
    if mae_model + eps < mae_baseline:
        verdict = "better"
    elif mae_baseline + eps < mae_model:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-20 ==")
    print(f"Number of held‑out rows: {len(predictions)}")
    print(f"Baseline mean: {round(baseline_mean,4)}")
    print(f"MAE model: {round(mae_model,4)}")
    print(f"MAE baseline: {round(mae_baseline,4)}")
    print(f"Verdict: {verdict}")

    # JSON answer
    answer = {
        "predictions": predictions,
        "baseline": round(baseline_mean,4),
        "mae_model": round(mae_model,4),
        "mae_baseline": round(mae_baseline,4),
        "verdict": verdict
    }
    print(f"ANSWER APACHE-20 {json.dumps(answer, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)

    apache_19(header, data)
    apache_20(header, data)

if __name__ == "__main__":
    main()

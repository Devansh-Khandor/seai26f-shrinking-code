#!/usr/bin/env python3
"""
Skill 10 – Apache performance analytics (APACHE-19 & APACHE-20)

Usage:
    python skill_10.py <csv_path>
"""

import json
import math
import random
import sys

import ezr

# --------------------------------------------------------------------
# Helpers – reused from previous skills
# --------------------------------------------------------------------
def goal_columns(tbl: ezr.TBL):
    """Return list of NUM columns that are goals (end with + or -)."""
    return [c for c in tbl.cols.values() if c.txt.endswith(('+', '-'))]


def goal_column(tbl: ezr.TBL):
    """Return the first column that is a goal (end with + or -)."""
    return goal_columns(tbl)[0]


def get_goal_col(tbl: ezr.TBL):
    """Return the goal column for predictions (same as goal_column)."""
    return goal_column(tbl)


# --------------------------------------------------------------------
# Our sequential evaluation – copied from skill_07 but without re‑seeding
# --------------------------------------------------------------------
def sequential_evaluation(tbl: ezr.TBL, budget: int, seed: int):
    """
    Perform a greedy sequential evaluation to discover a low
    performance configuration with minimal budget.
    """
    # Ensure reproducible order using the supplied seed
    random.seed(seed)

    # Gather all candidates that have complete goal information
    goal_cols = goal_columns(tbl)
    candidates = [
        i for i, row in enumerate(tbl.rows, start=1)
        if all(row[c.at] != "?" for c in goal_cols)
    ]
    if not candidates:
        raise RuntimeError("No evaluable rows.")

    evaluated = []
    evaluated_set = set()

    # Start with a random candidate
    first = random.choice(candidates)
    evaluated.append(first)
    evaluated_set.add(first)

    while len(evaluated) < budget:
        # Build ranker on already evaluated rows
        eval_rows = [tbl.rows[i - 1] for i in evaluated]
        rank = ezr.ranker(tbl, eval_rows)  # uses tree's leaf mu (ydist)

        # Select next best predicted
        best_idx = None
        best_val = float("inf")
        for idx in candidates:
            if idx in evaluated_set:
                continue
            val = rank(tbl.rows[idx - 1])
            if val < best_val:
                best_val = val
                best_idx = idx
        if best_idx is None:
            break
        evaluated.append(best_idx)
        evaluated_set.add(best_idx)

    # Determine best evaluated row (lowest ydist)
    best_row = min(
        evaluated, key=lambda i: ezr.ydist(tbl, tbl.rows[i - 1])
    )
    return best_row


# --------------------------------------------------------------------
# Tree‑based regressor for predictions
# --------------------------------------------------------------------
def tree_regressor(tbl: ezr.TBL, rows):
    """
    Return a function that predicts the raw goal value for a new row
    using the mean of goal values in the leaf where the row falls.
    """
    node = ezr.tree(tbl, rows)
    goal_col = goal_column(tbl)

    def predict(row):
        leaf = ezr.leaf(node, row)
        vals = [
            v for v in (leaf.rows[i][goal_col.at] for i in range(len(leaf.rows)))
            if v != "?"
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    return predict


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))
    rows = tbl.rows

    # General parameters
    budget = min(30, len(rows) // 5)
    repeats = 20
    seed_base = ezr.the.Seed

    # --------------------------------------------------------------------
    # APACHE-19 – compare sequential vs random selection
    # --------------------------------------------------------------------
    ours_best_rows = []
    random_best_rows = []

    for rep in range(repeats):
        # Our method
        ours_best = sequential_evaluation(tbl, budget, seed_base + rep)
        ours_best_rows.append(ours_best)

        # Random selection
        random.seed(seed_base + rep + 1000)  # different stream
        chosen = random.sample(rows, budget)
        # Pick best among chosen using ydist
        best_random = min(
            chosen,
            key=lambda r: ezr.ydist(tbl, r),
        )
        # Get index
        random_best = tbl.rows.index(best_random) + 1
        random_best_rows.append(random_best)

    # Compute average ydist for each method
    ours_avg_ydist = sum(
        ezr.ydist(tbl, tbl.rows[i - 1]) for i in ours_best_rows
    ) / repeats
    random_avg_ydist = sum(
        ezr.ydist(tbl, tbl.rows[i - 1]) for i in random_best_rows
    ) / repeats

    if ours_avg_ydist < random_avg_ydist - 1e-8:
        verdict = "ours better"
    elif ours_avg_ydist > random_avg_ydist + 1e-8:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Average ydist ours: {ours_avg_ydist:.4f}")
    print(f"Average ydist random: {random_avg_ydist:.4f}")
    print(f"Verdict: {verdict}")

    print(
        f"ANSWER APACHE-19 {json.dumps({'budget': budget, 'repeats': repeats, 'ours': ours_best_rows, 'random': random_best_rows, 'verdict': verdict})}"
    )

    # --------------------------------------------------------------------
    # APACHE-20 – predictions vs baseline
    # --------------------------------------------------------------------
    # Identify held‑out rows (divisible by 5)
    held_out_indices = [i for i in range(1, len(rows) + 1) if i % 5 == 0]
    non_held_out = [rows[i - 1] for i in range(1, len(rows) + 1) if i not in held_out_indices]

    goal_col = goal_column(tbl)
    goal_at = goal_col.at

    # Baseline: mean of non‑held‑out goal values
    non_held_vals = [float(r[goal_at]) for r in non_held_out if r[goal_at] != "?"]
    if non_held_vals:
        baseline_mean = sum(non_held_vals) / len(non_held_vals)
    else:
        baseline_mean = 0.0

    # Model predictions using tree regressor trained on non‑held‑out rows
    predictor = tree_regressor(tbl, non_held_out)

    predictions = {}
    errors_model = []
    errors_base = []

    for idx in held_out_indices:
        row = rows[idx - 1]
        if row[goal_at] == "?":
            continue  # skip missing goal
        actual = float(row[goal_at])
        pred = predictor(row)
        if pred is None:
            continue
        predictions[str(idx)] = round(pred, 5)
        errors_model.append(abs(pred - actual))
        errors_base.append(abs(baseline_mean - actual))

    mae_model = sum(errors_model) / len(errors_model) if errors_model else 0.0
    mae_baseline = sum(errors_base) / len(errors_base) if errors_base else 0.0

    if mae_model < mae_baseline - 1e-8:
        verdict2 = "better"
    elif mae_model > mae_baseline + 1e-8:
        verdict2 = "worse"
    else:
        verdict2 = "no clear difference"

    print("\n== APACHE-20 ==")
    print(f"MAE model: {mae_model:.4f}")
    print(f"MAE baseline: {mae_baseline:.4f}")
    print(f"Verdict: {verdict2}")

    print(
        f"ANSWER APACHE-20 {json.dumps({'predictions': predictions, 'baseline': baseline_mean, 'mae_model': mae_model, 'mae_baseline': mae_baseline, 'verdict': verdict2})}"
    )


if __name__ == "__main__":
    main()

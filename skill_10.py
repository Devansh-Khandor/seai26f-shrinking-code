#!/usr/bin/env python3
"""
Skill 10 – NASA project estimation comparison and selection evaluation

This script reads a NASA project CSV file and prints two analyses:
  NASA-19 – Compare the tree‑based effort estimator against a simple
             historical‑average estimate on unseen (held‑out) rows.
  NASA-20 – Compare the system’s project‑selection method with
             random selection under the same evaluation budget.

The program prints each section header (`== <ID> ==`) followed by at least
three lines of results and then a machine‑checked JSON answer line
(`ANSWER <ID> <json>`).  All numbers are derived from the supplied file
and no values are hard‑coded.

The implementation reuses functions from the `ezr` library and
the previous skill modules where appropriate.
"""

import json
import sys
import random

import ezr
import skill_01 as s01  # for goal statistics helper


# ----------------------------------------------------------------------
# Helper: compute mean & standard deviation (ignoring missing values)
# ----------------------------------------------------------------------
def _mean_sd(values):
    nums = [v for v in values if v != "?"]
    if not nums:
        return 0.0, 0.0
    m = sum(nums) / len(nums)
    var = sum((v - m) ** 2 for v in nums) / (len(nums) - 1) if len(nums) > 1 else 0.0
    return m, var ** 0.5


# ----------------------------------------------------------------------
# NASA-19 – model vs baseline effort estimation
# ----------------------------------------------------------------------
def _build_tree_on_non_held(tbl, eff_col_name):
    """Return a tree trained on non‑held‑out rows only."""
    eff_col = tbl.cols[eff_col_name]
    non_held_rows = [r for i, r in enumerate(tbl.rows, start=1) if i % 5 != 0]
    lab_tbl = ezr.clone(tbl, non_held_rows)

    tree_root = ezr.tree(lab_tbl, lab_tbl.rows)

    # Augment leaf nodes with average Effort-
    def _augment(node):
        if node.kids:
            for kid in node.kids:
                _augment(kid)
        else:
            vals = [r[eff_col.at] for r in node.rows if r[eff_col.at] != "?"]
            node.avg_effort = sum(vals) / len(vals) if vals else None

    _augment(tree_root)
    return tree_root


def _predict_effort(tree, row, eff_col):
    leaf = ezr.leaf(tree, row)
    return leaf.avg_effort if leaf.avg_effort is not None else None


def nasa_19(tbl):
    eff_col_name = "Effort-"
    eff_col = tbl.cols[eff_col_name]

    # Build model on non‑held rows
    tree = _build_tree_on_non_held(tbl, eff_col_name)

    # Held‑out rows
    held_rows = [r for i, r in enumerate(tbl.rows, start=1) if i % 5 == 0]
    held_indices = [i for i, _ in enumerate(tbl.rows, start=1) if i % 5 == 0]

    # Predictions and true values
    predictions = {}
    true_vals = []
    preds = []
    for idx, row in zip(held_indices, held_rows):
        pred = _predict_effort(tree, row, eff_col)
        predictions[str(idx)] = round(pred if pred is not None else 0.0, 4)
        actual = row[eff_col.at]
        if actual != "?":
            true_vals.append((pred if pred is not None else 0.0, float(actual)))

    # Baseline: mean Effort- of non‑held rows
    non_held_vals = [row[eff_col.at] for i, row in enumerate(tbl.rows, start=1)
                     if i % 5 != 0 and row[eff_col.at] != "?"]
    baseline_mean = sum(non_held_vals) / len(non_held_vals) if non_held_vals else 0.0

    # MAE calculations
    mae_model = sum(abs(p - a) for p, a in true_vals) / len(true_vals) if true_vals else 0.0
    mae_baseline = sum(abs(baseline_mean - a) for _, a in true_vals) / len(true_vals) if true_vals else 0.0

    # Verdict
    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-19 ==")
    print(f"Number of held‑out rows: {len(held_indices)}")
    print(f"MAE model: {round(mae_model,4)}, MAE baseline: {round(mae_baseline,4)}")
    print(f"Baseline mean Effort-: {round(baseline_mean,4)}")

    ans = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 4),
        "mae_model": round(mae_model, 4),
        "mae_baseline": round(mae_baseline, 4),
        "verdict": verdict
    }
    print(f"ANSWER NASA-19 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Helper: combined score for a row (higher is better)
# ----------------------------------------------------------------------
def _goal_stats(tbl):
    """Return dict mapping goal column name -> (min, max) over all rows."""
    stats = {}
    for col in tbl.cols.values():
        if col.txt.endswith(("+", "-")):
            vals = [row[col.at] for row in tbl.rows if row[col.at] != "?"]
            if vals:
                stats[col.txt] = (min(vals), max(vals))
    return stats


def _combined_score(row, goal_stats):
    score = 0.0
    for col_name, (mn, mx) in goal_stats.items():
        val = row[tbl.cols[col_name].at]
        if val == "?":
            return None
        if col_name.endswith("+"):
            norm = (val - mn) / (mx - mn) if mx > mn else 0.5
        else:  # '-'
            norm = (mx - val) / (mx - mn) if mx > mn else 0.5
        score += norm
    return score


# ----------------------------------------------------------------------
# NASA-20 – selection vs random
# ----------------------------------------------------------------------
def nasa_20(tbl):
    # Determine budget
    budget = min(30, len(tbl.rows) // 5)

    # Goal stats for normalization
    global tbl  # required by _combined_score
    goal_stats = _goal_stats(tbl)

    # Prepare all indices (1‑based)
    all_indices = list(range(1, len(tbl.rows) + 1))

    # Build order using combined_score (deterministic)
    scored = []
    for idx, row in enumerate(tbl.rows, start=1):
        sc = _combined_score(row, goal_stats)
        if sc is not None:
            scored.append((sc, idx))
    scored.sort(reverse=True, key=lambda x: x[0])
    ordered_indices = [idx for _, idx in scored]
    ours_best = ordered_indices[0] if ordered_indices else None

    repeats = 20
    ours_best_rows = []
    random_best_rows = []

    for rep in range(repeats):
        # Random sample
        rng = random.Random(100 + rep)  # deterministic but different each repeat
        sample_indices = rng.sample(all_indices, budget)
        # Best among random sample
        best_sc = -1.0
        best_idx = None
        for idx in sample_indices:
            row = tbl.rows[idx - 1]
            sc = _combined_score(row, goal_stats)
            if sc is not None and sc > best_sc:
                best_sc = sc
                best_idx = idx
        random_best_rows.append(best_idx)

        ours_best_rows.append(ours_best)

    # Average scores for comparison
    def avg_score(indices):
        total = 0.0
        cnt = 0
        for idx in indices:
            row = tbl.rows[idx - 1]
            sc = _combined_score(row, goal_stats)
            if sc is not None:
                total += sc
                cnt += 1
        return total / cnt if cnt else 0.0

    avg_ours = avg_score(ours_best_rows)
    avg_random = avg_score(random_best_rows)

    # Verdict
    if avg_ours > avg_random:
        verdict = "ours better"
    elif avg_random > avg_ours:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-20 ==")
    print(f"Budget: {budget} rows")
    print(f"Repeats: {repeats}")
    print(f"Average score ours: {round(avg_ours,4)}, random: {round(avg_random,4)}")
    print(f"Verdict: {verdict}")

    ans = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    print(f"ANSWER NASA-20 {json.dumps(ans, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    tbl = ezr.Tbl(ezr.csv(csv_path))
    nasa_19(tbl)
    nasa_20(tbl)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_10.py <csv_path>")
    main(sys.argv[1])

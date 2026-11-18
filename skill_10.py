#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <path to Apache_AllMeasurements.csv>
"""

import random
import json
import sys

# Reuse helpers from previous skills
import skill_01 as s01
import skill_07 as s07
import skill_02 as s02


def best_value(rows, goal_col, idx):
    """
    Return the goal value for row idx (1-based). If missing, return None.
    """
    r = rows[idx - 1]
    return r.get(goal_col)


def better_than(val_a, val_b, goal_col):
    """
    Return True if val_a is better than val_b for the given goal column.
    Lower values are better for '-' goals; higher for '+' goals.
    """
    if val_a is None or val_b is None:
        return False
    if goal_col.endswith('-'):
        return val_a < val_b
    else:
        return val_a > val_b


def compute_verdict(best_vals, random_vals, goal_col):
    """
    Compare two lists of best values (one from ours, one from random).
    Return verdict string.
    """
    ours_better = True
    random_better = True
    for v_ours, v_rand in zip(best_vals, random_vals):
        if not better_than(v_ours, v_rand, goal_col):
            ours_better = False
        if not better_than(v_rand, v_ours, goal_col):
            random_better = False
    if ours_better and not random_better:
        return "ours better"
    if random_better and not ours_better:
        return "random better"
    return "no clear difference"


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    # Identify goal columns (use first one)
    goal_cols = [c for c in header if s01.is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20

    # ----------------- APACHE-19 -----------------
    print("== APACHE-19 ==")

    # Our method: adaptive selection
    order, best_row_ours = s07.adaptive_selection(rows, goal_col, s07.setting_columns(header), budget)
    val_ours = best_value(rows, goal_col, best_row_ours) if best_row_ours else None

    ours_best_list = [best_row_ours] * repeats
    ours_vals = [val_ours] * repeats

    # Random method: repeated random samples
    random_best_rows = []
    random_best_vals = []
    all_indices = list(range(1, n_rows + 1))
    for _ in range(repeats):
        sample = random.sample(all_indices, k=budget)
        # Find best row in sample
        best_idx = None
        best_val = None
        for idx in sample:
            val = best_value(rows, goal_col, idx)
            if val is None:
                continue
            if best_val is None or better_than(val, best_val, goal_col):
                best_val = val
                best_idx = idx
        if best_idx is None:
            best_idx = sample[0]  # fallback
            best_val = best_value(rows, goal_col, best_idx)
        random_best_rows.append(best_idx)
        random_best_vals.append(best_val)

    verdict19 = compute_verdict(ours_vals, random_best_vals, goal_col)

    # Human‑readable output
    print(f"Budget per run: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best row from our method (repeated): {ours_best_list[:3]} ...")
    print(f"Best random rows (first 3 of {repeats}): {random_best_rows[:3]} ...")
    print(f"Verdict: {verdict19}")

    # JSON answer
    apache19_answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_list,
        "random": random_best_rows,
        "verdict": verdict19,
    }
    print("ANSWER APACHE-19", json.dumps(apache19_answer, separators=(",", ":")))

    # ----------------- APACHE-20 -----------------
    print("\n== APACHE-20 ==")

    # Held‑out rows: indices divisible by 5
    held_out_indices = [i for i in range(1, n_rows + 1) if i % 5 == 0]
    training_rows = [rows[i - 1] for i in range(1, n_rows + 1) if i not in held_out_indices]

    # Fit model on training data
    try:
        coeff = s02._fit_linear(training_rows, goal_col, s07.setting_columns(header))
    except Exception:
        coeff = None

    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    baseline_values = [r[goal_col] for r in training_rows if r[goal_col] is not None]
    baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0

    for idx in held_out_indices:
        r = rows[idx - 1]
        actual = r[goal_col]
        if actual is None:
            continue
        if coeff is not None:
            pred = s02._predict_row(r, coeff, s07.setting_columns(header))
        else:
            pred = None
        predictions[str(idx)] = pred
        if pred is not None:
            mae_model += abs(pred - actual)
        mae_baseline += abs(baseline_mean - actual)

    if held_out_indices:
        mae_model /= len(held_out_indices)
        mae_baseline /= len(held_out_indices)
    else:
        mae_model = 0.0
        mae_baseline = 0.0

    if mae_model < mae_baseline - 1e-6:
        verdict20 = "better"
    elif mae_model > mae_baseline + 1e-6:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    # Human‑readable output
    print(f"Baseline mean: {baseline_mean:.2f}")
    print(f"MAE model: {mae_model:.2f}")
    print(f"MAE baseline: {mae_baseline:.2f}")
    print(f"Verdict: {verdict20}")
    print(f"Sample predictions (first 3): {list(predictions.items())[:3]}")

    apache20_answer = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict20,
    }
    print("ANSWER APACHE-20", json.dumps(apache20_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()

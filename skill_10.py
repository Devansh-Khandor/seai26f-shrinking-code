#!/usr/bin/env python3
"""
Skill 10: Evaluation vs random and model vs baseline

Usage:
    python skill_10.py <path to auto93.csv>
"""

import json
import random
import sys

from skill_01 import read_csv
from skill_04 import compute_means_stds, compute_score
from skill_02 import train_and_predict


# ----------------------------------------------------------------------
# Utility to compute composite score using overall mean/std
def composite_score(row, means, stds):
    return compute_score(row, means, stds)


# ----------------------------------------------------------------------
# CAR-19: Selection vs Random
def car_19(rows):
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20

    # Overall statistics for score calculation
    means, stds = compute_means_stds(rows, list(range(n_rows)))

    # Composite scores for all rows
    all_scores = []
    for idx, row in enumerate(rows, start=1):
        if any(row.get(col) is None for col in ["Lbs-", "Acc+", "Mpg+"]):
            continue
        sc = composite_score(row, means, stds)
        all_scores.append((sc, idx))

    # Order by score descending, tie by row number ascending
    order = [idx for _, idx in sorted(all_scores, key=lambda t: (-t[0], t[1]))]

    ours_best_rows = []
    random_best_rows = []

    ours_best_scores = []
    random_best_scores = []

    for repeat in range(1, repeats + 1):
        # Selection method: evaluate first `budget` rows of the order
        selected = order[:budget]
        best_score = None
        best_row = None
        for row_no in selected:
            row = rows[row_no - 1]
            sc = composite_score(row, means, stds)
            if best_score is None or sc > best_score or (sc == best_score and row_no < best_row):
                best_score = sc
                best_row = row_no
        ours_best_rows.append(best_row)
        ours_best_scores.append(best_score)

        # Random method
        rng = random.Random(12345 + repeat)
        sampled = rng.sample(range(1, n_rows + 1), budget)
        best_score = None
        best_row = None
        for row_no in sampled:
            row = rows[row_no - 1]
            sc = composite_score(row, means, stds)
            if best_score is None or sc > best_score or (sc == best_score and row_no < best_row):
                best_score = sc
                best_row = row_no
        random_best_rows.append(best_row)
        random_best_scores.append(best_score)

    # Count who wins
    ours_better = 0
    random_better = 0
    for o_score, r_score in zip(ours_best_scores, random_best_scores):
        if o_score > r_score:
            ours_better += 1
        elif r_score > o_score:
            random_better += 1
    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== CAR-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best rows (selection): {ours_best_rows[:3]}")
    print(f"Best rows (random): {random_best_rows[:3]}")
    print(f"Verdict: {verdict}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    print("ANSWER CAR-19 " + json.dumps(answer, separators=(",", ":")))


# ----------------------------------------------------------------------
# CAR-20: Model vs baseline
def car_20(rows):
    n_rows = len(rows)

    # Identify held-out rows (divisible by 5)
    held_out_rows = [rows[i] for i in range(n_rows) if (i + 1) % 5 == 0]
    held_out_indices = [i + 1 for i in range(n_rows) if (i + 1) % 5 == 0]

    # Train model on non held-out rows
    predictions, mae_model_unused = train_and_predict(
        "Mpg+", rows, held_out_rows
    )  # predictions: dict of str(row_no) -> float

    # Baseline: mean Mpg+ of non held-out rows
    mpg_values = [
        r["Mpg+"] for r in rows if (r["Mpg+"] is not None) and ((rows.index(r) + 1) % 5 != 0)
    ]
    if mpg_values:
        baseline_mean = sum(mpg_values) / len(mpg_values)
    else:
        baseline_mean = 0.0

    # Compute MAE for model and baseline on held-out rows with available actual Mpg+
    mae_model = 0.0
    mae_baseline = 0.0
    n_valid = 0
    for row_no in held_out_indices:
        row = rows[row_no - 1]
        actual = row["Mpg+"]
        if actual is None:
            continue
        pred = predictions.get(str(row_no))
        if pred is None:
            continue
        mae_model += abs(pred - actual)
        mae_baseline += abs(baseline_mean - actual)
        n_valid += 1

    mae_model = mae_model / n_valid if n_valid > 0 else 0.0
    mae_baseline = mae_baseline / n_valid if n_valid > 0 else 0.0

    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("\n== CAR-20 ==")
    print(f"Baseline mean Mpg+: {baseline_mean:.3f}")
    print(f"MAE model: {mae_model:.3f}")
    print(f"MAE baseline: {mae_baseline:.3f}")
    print(f"Verdict: {verdict}")

    answer = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict,
    }
    print("ANSWER CAR-20 " + json.dumps(answer, separators=(",", ":")))


# ----------------------------------------------------------------------
# Main entry point
def main(csv_path):
    rows = read_csv(csv_path)
    car_19(rows)
    car_20(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

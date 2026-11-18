#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑19 and CAR‑20).

Usage:
    python skill_10.py <path_to_csv>
"""

import sys
import json
import random
import numpy as np
from skill_01 import load_data, compute_overall_top10
from skill_02 import prepare_features, predict_with_model, compute_mae
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# CAR-19: Comparison with random selection
# --------------------------------------------------------------------
def evaluate_selection_vs_random(data, budget=30, repeats=20, rng=None):
    """
    Compare the vehicle‑selection system (top‑score based) against random sampling.
    Returns a dict containing the best row chosen by each approach for each repeat,
    and a verdict string.
    """
    if rng is None:
        rng = random.Random(0)

    # Overall scores for all rows
    all_scores = compute_overall_top10(data)
    score_map = {row: score for row, score in all_scores}

    # Our selection: top `budget` rows by score
    ours_rows = sorted(score_map.keys(), key=lambda r: score_map[r], reverse=True)[:budget]

    ours_best_rows = []
    random_best_rows = []

    for _ in range(repeats):
        # Random selection of rows (without replacement)
        rand_rows = rng.sample(range(1, len(data) + 1), budget)

        # Find best row in our selection
        best_ours_row = min(
            (r for r in ours_rows),
            key=lambda r: (-score_map[r], r)  # higher score, tie break by smaller row
        )
        ours_best_rows.append(best_ours_row)

        # Find best row in random selection
        best_rand_row = min(
            (r for r in rand_rows),
            key=lambda r: (-score_map.get(r, -float("inf")), r)
        )
        random_best_rows.append(best_rand_row)

    # Determine verdict
    ours_better = all(
        score_map[ours_best_rows[i]] >= score_map[random_best_rows[i]]
        for i in range(repeats)
    )
    random_better = all(
        score_map[random_best_rows[i]] >= score_map[ours_best_rows[i]]
        for i in range(repeats)
    )

    if ours_better and any(
        score_map[ours_best_rows[i]] > score_map[random_best_rows[i]]
        for i in range(repeats)
    ):
        verdict = "ours better"
    elif random_better and any(
        score_map[random_best_rows[i]] > score_map[ours_best_rows[i]]
        for i in range(repeats)
    ):
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict,
    }

# --------------------------------------------------------------------
# CAR-20: Model vs baseline for held‑out Mpg+
# --------------------------------------------------------------------
def evaluate_model_vs_baseline(data):
    """
    Fit a linear regression model on all non‑held‑out rows to predict Mpg+.
    Compare MAE against a baseline that uses the historical mean.
    Returns a dict with predictions, baseline mean, MAEs, and verdict.
    """
    # Identify held‑out rows (divisible by 5)
    held_out_rows = [row for row in data if row["row"] % 5 == 0]
    held_out_nums = [row["row"] for row in held_out_rows]
    training_rows = [row for row in data if row["row"] % 5 != 0]

    # Feature columns for Mpg+ prediction
    mpg_feature_cols = ["Clndrs", "Volume", "HpX", "Model", "origin", "Lbs-", "Acc+"]

    # Prepare training data
    X_mpg, y_mpg, mpg_means = prepare_features(
        training_rows, mpg_feature_cols, "Mpg+"
    )

    if X_mpg is None:
        # No training data; return empty results
        predictions = {}
        baseline_mean = 0.0
        mae_model = mae_baseline = 0.0
    else:
        # Train model
        model = LinearRegression().fit(X_mpg, y_mpg)

        # Predict on held‑out rows
        predictions = predict_with_model(
            model, data, mpg_feature_cols, mpg_means, "Mpg+", held_out_nums
        )

        # Compute MAE for model
        mae_model = compute_mae(predictions, data, "Mpg+", held_out_nums)

        # Baseline: mean Mpg+ of training data
        baseline_mean = float(np.mean(y_mpg))
        # MAE of baseline
        baseline_errors = [
            abs(row["values"]["Mpg+"] - baseline_mean)
            for row in held_out_rows
            if row["values"]["Mpg+"] is not None
        ]
        mae_baseline = float(np.mean(baseline_errors)) if baseline_errors else 0.0

    # Verdict
    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": {str(r): v for r, v in predictions.items()},
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict,
    }

# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)

    # ----- CAR-19 ----------------------------------------------------
    print("== CAR-19 ==")
    # Evaluate 20 repeats with budget 30
    result_19 = evaluate_selection_vs_random(data, budget=30, repeats=20)
    print(f"Number of repeats: {result_19['repeats']}")
    print(f"Budget per evaluation: {result_19['budget']}")
    print(f"Best rows from our system (first 5): {result_19['ours'][:5]}")
    print(f"Best rows from random (first 5): {result_19['random'][:5]}")
    # Machine‑checked answer
    answer_json = json.dumps(result_19, separators=(",", ":"))
    print(f"ANSWER CAR-19 {answer_json}")

    # ----- CAR-20 ----------------------------------------------------
    print("\n== CAR-20 ==")
    result_20 = evaluate_model_vs_baseline(data)
    held_out_count = len([row for row in data if row["row"] % 5 == 0])
    print(f"Total rows: {len(data)}")
    print(f"Number of held‑out rows: {held_out_count}")
    print(f"MAE of model: {result_20['mae_model']:.4f}")
    print(f"MAE of baseline: {result_20['mae_baseline']:.4f}")
    print(f"Verdict: {result_20['verdict']}")
    # Machine‑checked answer
    answer_json = json.dumps(result_20, separators=(",", ":"))
    print(f"ANSWER CAR-20 {answer_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

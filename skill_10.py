#!/usr/bin/env python3
"""
Skill 10 – Comparison of vehicle selection and prediction accuracy.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Helpers – import from skill_08
# --------------------------------------------------------------------------- #
try:
    from skill_08 import compute_overall_scores
except Exception:
    # Fallback minimal implementation
    def compute_overall_scores(
        data, goal_cols, goal_directions
    ):  # pragma: no cover
        # compute min/max per goal
        min_vals = {}
        max_vals = {}
        for col in goal_cols:
            vals = [row[col] for row in data if row[col] is not None]
            if vals:
                min_vals[col] = min(vals)
                max_vals[col] = max(vals)
            else:
                min_vals[col] = max_vals[col] = 0.0
        scores = []
        for idx, row in enumerate(data, start=1):
            if any(row[col] is None for col in goal_cols):
                continue
            vals = [
                (row[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
                if max_vals[col] != min_vals[col]
                else 1.0
                for col in goal_cols
                if goal_directions[col]
            ] + [
                (max_vals[col] - row[col]) / (max_vals[col] - min_vals[col])
                if max_vals[col] != min_vals[col]
                else 1.0
                for col in goal_cols
                if not goal_directions[col]
            ]
            scores.append((idx, sum(vals) / len(vals)))
        return scores

# --------------------------------------------------------------------------- #
# Parse helper (identical to earlier skills)
# --------------------------------------------------------------------------- #
def parse_float(s: str) -> float | None:
    s = s.strip()
    if s == "" or s == "?":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Read CSV
    data: List[Dict[str, float | None]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    total_rows = len(data)
    budget = min(30, total_rows // 5)  # as per spec (30 for the given file)
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    goal_directions = {"Lbs-": False, "Acc+": True, "Mpg+": True}

    # Pre‑compute overall scores for all rows with complete goals
    overall_scores = compute_overall_scores(data, goal_cols, goal_directions)
    score_dict = dict(overall_scores)  # row_num -> score
    complete_rows = list(score_dict.keys())

    # -------------------------------------------------------------- #
    # CAR-19: Selection vs random
    # -------------------------------------------------------------- #
    repeats = 20
    ours_best: List[int] = []
    random_best: List[int] = []
    ours_better = 0
    random_better = 0

    # Prepare features for regression (non‑goal, non‑X columns)
    non_goal_cols = [
        c for c in data[0].keys() if not (c.endswith("+") or c.endswith("-") or c.endswith("X"))
    ]

    # Build regression model to predict overall score
    X_all = np.array(
        [
            [row.get(col) if row.get(col) is not None else np.nan for col in non_goal_cols]
            for row in data
        ]
    )
    # Use rows with complete goal values to compute target
    y_all = np.array(
        [score_dict[row_num] for row_num in complete_rows], dtype=float
    )
    # Mask rows with NaN in features
    mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)
    X_train = X_all[mask]
    y_train = y_all[mask]

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)

    reg = LinearRegression()
    reg.fit(X_train_imp, y_train)

    # Predict on all rows
    X_full = np.array(
        [
            [row.get(col) if row.get(col) is not None else np.nan for col in non_goal_cols]
            for row in data
        ]
    )
    X_full_imp = imputer.transform(X_full)
    pred_scores_all = reg.predict(X_full_imp)

    # Map row_num to predicted score (only for rows with complete goals)
    pred_scores = {
        row_num: float(pred_scores_all[row_num - 1]) for row_num in complete_rows
    }

    for rep in range(repeats):
        random.seed(rep)
        # Random selection of rows with complete goals
        if len(complete_rows) <= budget:
            random_eval = complete_rows.copy()
        else:
            random_eval = random.sample(complete_rows, budget)
        # Best row in random selection
        rand_best_row = max(random_eval, key=lambda rn: score_dict[rn])
        random_best.append(rand_best_row)

        # Our selection: top predicted scores among rows with complete goals
        sorted_pred = sorted(complete_rows, key=lambda rn: pred_scores[rn], reverse=True)
        ours_eval = sorted_pred[:budget] if len(sorted_pred) >= budget else sorted_pred
        ours_best_row = max(ours_eval, key=lambda rn: score_dict[rn])
        ours_best.append(ours_best_row)

        # Compare
        if score_dict[ours_best_row] > score_dict[rand_best_row]:
            ours_better += 1
        elif score_dict[rand_best_row] > score_dict[ours_best_row]:
            random_better += 1
        # ties ignored

    if ours_better > random_better:
        verdict19 = "ours better"
    elif random_better > ours_better:
        verdict19 = "random better"
    else:
        verdict19 = "no clear difference"

    # Output CAR-19
    print("== CAR-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best rows (ours): {ours_best}")
    print(f"Best rows (random): {random_best}")
    print(f"Ours better times: {ours_better}, Random better times: {random_better}")
    ans_car_19 = json.dumps(
        {
            "budget": budget,
            "repeats": repeats,
            "ours": ours_best,
            "random": random_best,
            "verdict": verdict19,
        },
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-19 {ans_car_19}")

    # -------------------------------------------------------------- #
    # CAR-20: Prediction vs baseline
    # -------------------------------------------------------------- #
    # Identify held‑out rows (divisible by 5)
    held_out_rows = [i + 1 for i in range(total_rows) if (i + 1) % 5 == 0]

    # Features for Mpg+ prediction: all columns except Mpg+
    mpg_features = [c for c in data[0].keys() if c != "Mpg+"]

    # Training data: rows with non‑missing Mpg+
    train_rows = [row for row in data if row["Mpg+"] is not None]
    if not train_rows:
        print("\n== CAR-20 ==")
        print("No training data for Mpg+ prediction.")
        print("ANSWER CAR-20 {}")
        return

    X_train_mpg = np.array(
        [[row.get(col) if row.get(col) is not None else np.nan for col in mpg_features]
         for row in train_rows]
    )
    y_train_mpg = np.array([row["Mpg+"] for row in train_rows], dtype=float)

    mask_mpg = ~np.isnan(X_train_mpg).any(axis=1) & ~np.isnan(y_train_mpg)
    X_train_mpg = X_train_mpg[mask_mpg]
    y_train_mpg = y_train_mpg[mask_mpg]

    imputer_mpg = SimpleImputer(strategy="mean")
    X_train_imp_mpg = imputer_mpg.fit_transform(X_train_mpg)

    reg_mpg = LinearRegression()
    reg_mpg.fit(X_train_imp_mpg, y_train_mpg)

    # Predict on held‑out rows
    preds_mpg: Dict[int, float] = {}
    actuals_mpg: Dict[int, float] = {}
    for row_num in held_out_rows:
        row = data[row_num - 1]
        if row["Mpg+"] is None:
            continue  # cannot compute MAE if missing actual; skip
        x_vec = np.array(
            [[row.get(col) if row.get(col) is not None else np.nan for col in mpg_features]]
        )
        x_imp = imputer_mpg.transform(x_vec)
        pred = reg_mpg.predict(x_imp)[0]
        preds_mpg[row_num] = round(float(pred), 6)
        actuals_mpg[row_num] = row["Mpg+"]

    # Baseline: mean Mpg+ of non‑held‑out rows
    non_held_vals = [row["Mpg+"] for row in data if row["Mpg+"] is not None and (row is not None) and ((data.index(row)+1) not in held_out_rows)]
    baseline_mean = sum(non_held_vals) / len(non_held_vals) if non_held_vals else 0.0
    baseline_pred = baseline_mean

    mae_model = np.mean([abs(preds_mpg[r] - actuals_mpg[r]) for r in preds_mpg])
    mae_baseline = np.mean([abs(baseline_pred - actuals_mpg[r]) for r in preds_mpg])

    if mae_model < mae_baseline:
        verdict20 = "better"
    elif mae_model > mae_baseline:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    # Output CAR-20
    print("\n== CAR-20 ==")
    print(f"Predictions for held‑out rows: {preds_mpg}")
    print(f"Baseline mean Mpg+: {round(baseline_mean, 6)}")
    print(f"MAE model: {round(mae_model, 6)}, MAE baseline: {round(mae_baseline, 6)}")
    ans_car_20 = json.dumps(
        {
            "predictions": preds_mpg,
            "baseline": round(baseline_mean, 6),
            "mae_model": round(mae_model, 6),
            "mae_baseline": round(mae_baseline, 6),
            "verdict": verdict20,
        },
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-20 {ans_car_20}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

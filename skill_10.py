#!/usr/bin/env python3
"""
skill_10.py: NASA-19 and NASA-20 comparisons.
Usage:  python skill_10.py <csv path>
"""

import csv
import json
import random
import sys
from collections import defaultdict

# Import helpers from previous skills
from skill_01 import parse_float, std_dev
from skill_04 import compute_combined_score

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def compute_mean_std(values):
    """Return mean and population std for list of floats."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(values) / n
    var = sum((v - mean_val) ** 2 for v in values) / n
    return mean_val, var ** 0.5

# ----------------------------------------------------------------------
# NASA-19: Effort prediction comparison
# ----------------------------------------------------------------------
def estimate_effort_nn(hold_row, all_rows, non_held_rows, numeric_cols):
    """Estimate Effort- for a held-out row using weighted NN (3 neighbors)."""
    # Distances to all non-held rows
    dist_rows = []
    for r in non_held_rows:
        d_sq = 0.0
        for col in numeric_cols:
            v1 = parse_float(hold_row.get(col))
            v2 = parse_float(r.get(col))
            if v1 is None or v2 is None:
                continue
            diff = v1 - v2
            d_sq += diff * diff
        if d_sq == 0.0:
            # identical on all numeric columns; avoid infinite weight
            d_sq = 1e-12
        dist_rows.append((d_sq, r))

    dist_rows.sort(key=lambda x: x[0])
    neighbors = []
    for d_sq, r in dist_rows:
        eff = parse_float(r.get("Effort-"))
        if eff is not None:
            neighbors.append((d_sq, eff))
        if len(neighbors) >= 3:
            break

    if not neighbors:
        # Fall back to mean Effort- of training set
        all_eff = [parse_float(r.get("Effort-")) for r in non_held_rows]
        all_eff = [v for v in all_eff if v is not None]
        return round(sum(all_eff) / len(all_eff) if all_eff else 0.0, 4)

    # Weighted average by inverse distance
    weighted = 0.0
    weight_sum = 0.0
    for d_sq, eff in neighbors:
        w = 1.0 / (d_sq + 1e-6)
        weighted += w * eff
        weight_sum += w
    return round(weighted / weight_sum, 4)

def nasa19(csv_path):
    # Load rows
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            r = dict(r)
            r["row_num"] = idx
            rows.append(r)

    # Identify held-out rows
    held_out_nums = {r["row_num"] for r in rows if r["row_num"] % 5 == 0}
    non_held_rows = [r for r in rows if r["row_num"] not in held_out_nums]

    # Determine numeric columns (exclude goals and X)
    all_cols = reader.fieldnames
    numeric_cols = []
    for col in all_cols:
        if col.endswith('+') or col.endswith('-') or col.endswith('X'):
            continue
        # Check if numeric in at least one row
        for r in rows:
            if parse_float(r.get(col)) is not None:
                numeric_cols.append(col)
                break

    # Predict for each held-out row
    predictions = {}
    actuals = []
    for r in rows:
        if r["row_num"] in held_out_nums:
            est = estimate_effort_nn(r, rows, non_held_rows, numeric_cols)
            predictions[str(r["row_num"])] = est
            actual = parse_float(r.get("Effort-"))
            if actual is not None:
                actuals.append((est, actual))

    # Baseline mean Effort- of non-held rows
    baseline_vals = [parse_float(r.get("Effort-")) for r in non_held_rows]
    baseline_vals = [v for v in baseline_vals if v is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    # MAE calculations
    if actuals:
        mae_model = sum(abs(est - act) for est, act in actuals) / len(actuals)
        mae_baseline = sum(abs(baseline_mean - act) for _, act in actuals) / len(actuals)
    else:
        mae_model = mae_baseline = 0.0

    # Verdict
    if mae_model < mae_baseline - 1e-6:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-6:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-19 ==")
    print(f"MAE model: {round(mae_model,4)}")
    print(f"MAE baseline: {round(mae_baseline,4)}")
    print(f"Baseline mean Effort-: {round(baseline_mean,4)}")
    print(f"Predictions for held-out rows:")
    for rnum in sorted(predictions, key=lambda x: int(x)):
        print(f"  Row {rnum}: {predictions[rnum]}")
    nasa19_json = {
        "predictions": predictions,
        "baseline": round(baseline_mean,4),
        "mae_model": round(mae_model,4),
        "mae_baseline": round(mae_baseline,4),
        "verdict": verdict
    }
    print(f"ANSWER NASA-19 {json.dumps(nasa19_json)}")

# ----------------------------------------------------------------------
# NASA-20: Selection comparison
# ----------------------------------------------------------------------
def nasa20(csv_path):
    # Load rows
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            r = dict(r)
            r["row_num"] = idx
            rows.append(r)

    # Budget
    budget = min(30, len(rows) // 5)

    # Compute means and stds for goal columns
    goal_cols = ["Kloc+", "Effort-", "Defects-"]
    goal_vals = {col: [] for col in goal_cols}
    for r in rows:
        for col in goal_cols:
            v = parse_float(r.get(col))
            if v is not None:
                goal_vals[col].append(v)

    means = {}
    stds = {}
    for col in goal_cols:
        m, s = compute_mean_std(goal_vals[col])
        means[col] = m
        stds[col] = s if s > 0 else 1.0

    # Precompute score for each row
    row_scores = {}
    for r in rows:
        score = compute_combined_score(r, means, stds)
        if score is not None:
            row_scores[r["row_num"]] = score

    # Our deterministic selection: top budget rows by score
    top_rows = sorted(row_scores.items(), key=lambda x: -x[1])[:budget]
    ours_best = top_rows[0][0] if top_rows else None

    # Random selections
    random.seed(0)  # deterministic
    random_best_rows = []
    random_scores = []
    for _ in range(20):
        sample = random.sample(rows, budget)
        best = None
        best_score = None
        for r in sample:
            sc = row_scores.get(r["row_num"])
            if sc is None:
                continue
            if best is None or sc > best_score:
                best = r["row_num"]
                best_score = sc
        if best is not None:
            random_best_rows.append(best)
            random_scores.append(best_score)

    # Average scores
    ours_avg_score = row_scores[ours_best] if ours_best is not None else 0.0
    rand_avg_score = sum(random_scores) / len(random_scores) if random_scores else 0.0

    # Verdict
    if ours_avg_score > rand_avg_score + 1e-6:
        verdict = "ours better"
    elif ours_avg_score < rand_avg_score - 1e-6:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-20 ==")
    print(f"Budget: {budget}")
    print(f"Our best row: {ours_best} (score {round(ours_avg_score,4)})")
    print(f"Random best rows: {random_best_rows}")
    print(f"Random best avg score: {round(rand_avg_score,4)}")
    print("First 3 random best rows:")
    for i, rnum in enumerate(random_best_rows[:3], start=1):
        print(f"  {i}. Row {rnum}")

    nasa20_json = {
        "budget": budget,
        "repeats": 20,
        "ours": [ours_best],
        "random": random_best_rows,
        "verdict": verdict
    }
    print(f"ANSWER NASA-20 {json.dumps(nasa20_json)}")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(csv_path):
    nasa19(csv_path)
    print()
    nasa20(csv_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
Skill 10 – NASA-19 & NASA-20
"""
import csv
import json
import math
import random
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Re‑use helpers from earlier skills
# ----------------------------------------------------------------------
from skill_01 import parse_value, normalise_composite
from skill_02 import train_linreg, predict_linreg

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GOALS = ["Kloc+", "Effort-", "Defects-"]

# ----------------------------------------------------------------------
# Utility: load CSV and parse goal columns
# ----------------------------------------------------------------------
def load_and_parse(csv_path: str) -> Tuple[List[Dict[str, str]], List[Dict[str, float]]]:
    """Return (raw_rows, parsed_rows). parsed_rows has goal columns as float."""
    raw_rows: List[Dict[str, str]] = []
    parsed_rows: List[Dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row.copy())
            parsed = {}
            for col, val in row.items():
                if col in GOALS:
                    parsed[col] = parse_value(val)
                else:
                    parsed[col] = val
            parsed_rows.append(parsed)
    return raw_rows, parsed_rows

# ----------------------------------------------------------------------
# NASA-19
# ----------------------------------------------------------------------
def nasa_19(csv_path: str) -> None:
    raw_rows, parsed_rows = load_and_parse(csv_path)
    n_rows = len(raw_rows)

    # Identify held‑out rows (number divisible by 5)
    held_out = [i for i in range(n_rows) if (i + 1) % 5 == 0]
    training_rows = [parsed_rows[i] for i in range(n_rows) if (i + 1) % 5 != 0]

    # Baseline mean Effort- over training set
    effort_vals = [r["Effort-"] for r in training_rows if r["Effort-"] is not None]
    if effort_vals:
        baseline_mean = sum(effort_vals) / len(effort_vals)
    else:
        baseline_mean = None

    # Train linear regression on YearX
    train_years = []
    train_effort = []
    for r in training_rows:
        year = parse_value(r["YearX"])
        eff = r["Effort-"]
        if year is not None and eff is not None:
            train_years.append(year)
            train_effort.append(eff)
    a_eff, b_eff = train_linreg(train_years, train_effort)

    # Predict on held‑out rows
    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for idx in held_out:
        row_raw = raw_rows[idx]
        year = parse_value(row_raw["YearX"])
        actual = parse_value(row_raw["Effort-"])
        pred = predict_linreg(a_eff, b_eff, year) if year is not None else None
        if pred is not None and actual is not None:
            predictions[str(idx + 1)] = round(pred, 3)
            mae_model += abs(pred - actual)
            mae_baseline += abs(actual - baseline_mean)
            count += 1

    mae_model = round(mae_model / count if count else None, 3)
    mae_baseline = round(mae_baseline / count if count else None, 3)

    # Verdict
    verdict = "no clear difference"
    if mae_model is not None and mae_baseline is not None:
        if mae_model < mae_baseline:
            verdict = "better"
        elif mae_model > mae_baseline:
            verdict = "worse"

    # Print results
    print("== NASA-19 ==")
    print(f"Number of held‑out rows: {len(held_out)}")
    print(f"Baseline mean Effort-: {baseline_mean:.3f}" if baseline_mean is not None else "Baseline mean Effort-: None")
    print(f"Model MAE: {mae_model:.3f}   Baseline MAE: {mae_baseline:.3f}")

    answer = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 3) if baseline_mean is not None else None,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    print("ANSWER NASA-19 " + json.dumps(answer, separators=(",", ":")))

# ----------------------------------------------------------------------
# NASA-20
# ----------------------------------------------------------------------
def nasa_20(csv_path: str) -> None:
    raw_rows, parsed_rows = load_and_parse(csv_path)
    n_rows = len(raw_rows)

    budget = min(30, n_rows // 5)
    repeats = 20
    random.seed(0)  # deterministic

    # Compute composite scores for all rows
    composite_list = normalise_composite(parsed_rows)  # list of (row_num, score)
    composite_dict = {row_num: score for row_num, score in composite_list}

    # Our selection: top budget rows by composite
    ours_candidates = sorted(composite_list, key=lambda x: x[1], reverse=True)[:budget]
    ours_selected_rows = [row_num for row_num, _ in ours_candidates]

    ours_best_row = max(ours_selected_rows, key=lambda r: composite_dict[r])
    ours_best_score = composite_dict[ours_best_row]

    # Random selections
    random_best_rows = []
    for _ in range(repeats):
        sample = random.sample(range(1, n_rows + 1), budget)
        best_row = max(sample, key=lambda r: composite_dict[r])
        random_best_rows.append(best_row)

    # Compute wins
    ours_wins = 0
    random_wins = 0
    for best in random_best_rows:
        ours_score = composite_dict[ours_best_row]
        random_score = composite_dict[best]
        if ours_score > random_score:
            ours_wins += 1
        elif ours_score < random_score:
            random_wins += 1
        # ties do nothing

    # Verdict
    if ours_wins > random_wins:
        verdict = "ours better"
    elif ours_wins < random_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Print results
    print("\n== NASA-20 ==")
    print(f"Budget: {budget} rows, repeats: {repeats}")
    print(f"Our best row: {ours_best_row} (score {ours_best_score:.3f})")
    print(f"Random best rows sample: {random_best_rows[:5]}")
    print(f"Our wins: {ours_wins}  Random wins: {random_wins}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": [ours_best_row] * repeats,  # same best each repeat
        "random": random_best_rows,
        "verdict": verdict
    }
    print("ANSWER NASA-20 " + json.dumps(answer, separators=(",", ":")))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    nasa_19(csv_path)
    nasa_20(csv_path)

if __name__ == "__main__":
    main()

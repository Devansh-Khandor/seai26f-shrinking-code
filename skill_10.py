#!/usr/bin/env python3
"""
skill_10.py

NASA-19: Compare project‑effort estimation system with historical‑average estimate.
NASA-20: Compare our project‑selection method with random selection under the same budget.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ------------------------------------------------------------------
# CSV and type helpers
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    return rows, header


def is_numeric(val: str) -> bool:
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def to_float(val: str) -> float:
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------
def build_pipeline(feature_rows: List[Dict[str, float]], feature_names: List[str]):
    """Return a sklearn pipeline for regression on the given feature rows."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    # Detect numeric vs categorical columns
    numeric_cols: set = set()
    categorical_cols: set = set()

    for col in feature_names:
        found_num = False
        for row in feature_rows:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, float) or isinstance(val, int):
                found_num = True
                break
            if isinstance(val, str) and val.strip() != "?":
                try:
                    float(val)
                    found_num = True
                    break
                except ValueError:
                    pass
        if found_num:
            numeric_cols.add(col)
        else:
            categorical_cols.add(col)

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ]
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                          ("model", LinearRegression())])
    return pipe


def prepare_feature_dict(row: Dict[str, str], feature_names: List[str], numeric_cols: set) -> Dict[str, object]:
    """Convert a row dict to a feature dict suitable for the pipeline."""
    d: Dict[str, object] = {}
    for col in feature_names:
        val = row[col]
        if val.strip() == "?":
            # missing values: numeric -> NaN, categorical -> None (treated as a category)
            d[col] = None if col not in numeric_cols else float("nan")
        else:
            if col in numeric_cols:
                d[col] = float(val)
            else:
                d[col] = val
    return d


# ------------------------------------------------------------------
# NASA-19: Effort estimation comparison
# ------------------------------------------------------------------
def nasa_19(rows: List[Dict[str, str]], header: List[str]) -> Tuple[Dict[str, float], float, float, float, str]:
    # Determine columns
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    x_cols = {c for c in header if c.endswith("X")}
    feature_names = [c for c in header if c not in goal_cols and c not in x_cols]

    # Build numeric column set
    numeric_cols = set()
    for col in feature_names:
        for r in rows:
            if is_numeric(r[col]):
                numeric_cols.add(col)
                break

    # Separate training (non‑held‑out) and held‑out
    held_out_idx = {i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0}
    train_rows = [r for i, r in enumerate(rows, start=1) if i not in held_out_idx]
    held_rows = [r for i, r in enumerate(rows, start=1) if i in held_out_idx]

    # Prepare training data
    X_train = [prepare_feature_dict(r, feature_names, numeric_cols) for r in train_rows if not math.isnan(to_float(r["Effort-"]))]
    y_train = [to_float(r["Effort-"]) for r in train_rows if not math.isnan(to_float(r["Effort-"]))]

    # Build and train pipeline
    pipe = build_pipeline(X_train, feature_names)
    pipe.fit(X_train, y_train)

    # Predictions for held‑out rows
    predictions = {}
    actuals = {}
    for i, r in enumerate(held_rows, start=1):
        idx = held_out_idx.pop()  # pop in order
        if math.isnan(to_float(r["Effort-"])):
            continue
        feat = prepare_feature_dict(r, feature_names, numeric_cols)
        pred = pipe.predict([feat])[0]
        predictions[str(idx)] = round(pred, 4)
        actuals[idx] = to_float(r["Effort-"])

    # Baseline: mean of training Effort-
    baseline = sum(y_train) / len(y_train) if y_train else 0.0
    baseline = round(baseline, 4)

    # MAEs
    mae_model = sum(abs(actuals[i] - predictions[str(i)]) for i in actuals) / len(actuals) if actuals else float("nan")
    mae_baseline = sum(abs(actuals[i] - baseline) for i in actuals) / len(actuals) if actuals else float("nan")
    mae_model = round(mae_model, 4)
    mae_baseline = round(mae_baseline, 4)

    # Verdict
    if math.isnan(mae_model) or math.isnan(mae_baseline):
        verdict = "no clear difference"
    elif mae_model < mae_baseline - 1e-4:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-4:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return predictions, baseline, mae_model, mae_baseline, verdict


# ------------------------------------------------------------------
# NASA-20: Project selection comparison
# ------------------------------------------------------------------
def composite_score(row: Dict[str, str]) -> float:
    if any(row[col].strip() == "?" for col in ("Kloc+", "Effort-", "Defects-")):
        return float("-inf")
    return to_float(row["Kloc+"]) - to_float(row["Effort-"]) - to_float(row["Defects-"])


def nasa_20(rows: List[Dict[str, str]], header: List[str]) -> Tuple[int, List[int], List[int], str]:
    n_rows = len(rows)
    budget = min(30, n_rows // 5)

    # Compute composite for all rows
    comps = {}
    for idx, r in enumerate(rows, start=1):
        comps[idx] = composite_score(r)

    # Our selection: top budget rows by composite
    sorted_by_comp = sorted(comps.items(), key=lambda x: (-x[1], x[0]))
    selected = [idx for idx, _ in sorted_by_comp[:budget]]
    best_ours = selected[0] if selected else None

    # Random repeats
    random.seed(42)
    repeats = 20
    ours_list = [best_ours] * repeats
    random_best_list = []

    for _ in range(repeats):
        sample = random.sample(range(1, n_rows + 1), budget)
        sample_comps = [(i, comps[i]) for i in sample]
        best_i = min(sample_comps, key=lambda x: ( -x[1], x[0]))[0]
        random_best_list.append(best_i)

    # Compare wins
    ours_wins = 0
    random_wins = 0
    for ours, rnd in zip(ours_list, random_best_list):
        if comps[ours] > comps[rnd]:
            ours_wins += 1
        elif comps[rnd] > comps[ours]:
            random_wins += 1

    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return budget, ours_list, random_best_list, verdict


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # NASA-19
    preds, baseline, mae_model, mae_baseline, verdict19 = nasa_19(rows, header)

    print("== NASA-19 ==")
    print(f"Model MAE: {mae_model}")
    print(f"Baseline MAE: {mae_baseline}")
    print(f"Verdict: {verdict19}")

    answer_19 = {
        "predictions": preds,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict19,
    }
    print("ANSWER NASA-19 " + json.dumps(answer_19))

    # NASA-20
    budget, ours_list, random_list, verdict20 = nasa_20(rows, header)

    print("\n== NASA-20 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: 20")
    print(f"Ours best row: {ours_list[0]}")
    print(f"Random best rows: {random_list}")
    print(f"Verdict: {verdict20}")

    answer_20 = {
        "budget": budget,
        "repeats": 20,
        "ours": ours_list,
        "random": random_list,
        "verdict": verdict20,
    }
    print("ANSWER NASA-20 " + json.dumps(answer_20))


if __name__ == "__main__":
    main(sys.argv)

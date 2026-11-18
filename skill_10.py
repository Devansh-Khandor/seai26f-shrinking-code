#!/usr/bin/env python3
"""
Skill 10: NASA-19 & NASA-20 evaluation

Usage:
    python skill_10.py <path-to-nasa93dem.csv>
"""

import sys
import json
import random
from statistics import mean

# Import helpers
try:
    from skill_01 import read_csv, format_number
except ImportError:
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

# For NASA-19: use predict_effort from skill_06
try:
    from skill_06 import predict_effort
except ImportError:
    # Minimal implementation similar to skill_06
    from skill_02 import determine_feature_types, build_encodings, build_feature_matrix
    from sklearn.linear_model import LinearRegression
    import numpy as np

    def predict_effort(rows, feature_cols):
        # Build training and test sets
        train_rows = [r for idx, r in enumerate(rows, start=1)
                      if idx % 5 != 0 and r['Effort-'] is not None
                      and all(r[c] is not None for c in feature_cols)]
        test_rows = [(idx, r) for idx, r in enumerate(rows, start=1)
                     if idx % 5 == 0 and r['Effort-'] is not None
                     and all(r[c] is not None for c in feature_cols)]

        numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
        encodings = build_encodings(train_rows, categorical_cols)

        X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols,
                                       categorical_cols, encodings)
        y_train = np.array([r['Effort-'] for r in train_rows], dtype=float)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = {}
        for idx, r in test_rows:
            vec = []
            for col in feature_cols:
                v = r[col]
                if col in numeric_cols:
                    vec.append(float(v))
                else:
                    vec.append(encodings[col][v])
            X_test = np.array([vec], dtype=float)
            pred = model.predict(X_test)[0]
            predictions[idx] = float(pred)
        return predictions

# ----------------------------------------------------------------------
# NASA-19
# ----------------------------------------------------------------------
def nasa_19(rows):
    # Feature columns: all except Effort- and Defects-
    feature_cols = [c for c in rows[0].keys() if c not in ('Effort-', 'Defects-')]

    # Predictions from model
    predictions = predict_effort(rows, feature_cols)

    # Held‑out indices
    held_out = [idx for idx in predictions.keys()]

    # Baseline: mean of Effort- of non‑held‑out rows
    baseline_vals = [r['Effort-'] for idx, r in enumerate(rows, start=1)
                     if idx % 5 != 0 and r['Effort-'] is not None]
    baseline = mean(baseline_vals) if baseline_vals else None

    # MAE for model
    errors_model = [abs(predictions[idx] - rows[idx-1]['Effort-'])
                    for idx in held_out if rows[idx-1]['Effort-'] is not None]
    mae_model = mean(errors_model) if errors_model else None

    # MAE for baseline
    errors_base = [abs(rows[idx-1]['Effort-'] - baseline)
                   for idx in held_out if rows[idx-1]['Effort-'] is not None]
    mae_baseline = mean(errors_base) if errors_base else None

    # Verdict
    if mae_model is None or mae_baseline is None:
        verdict = "no clear difference"
    else:
        if mae_model + 1e-6 < mae_baseline:
            verdict = "better"
        elif mae_baseline + 1e-6 < mae_model:
            verdict = "worse"
        else:
            verdict = "no clear difference"

    # Results output
    print("== NASA-19 ==")
    print(f"MAE model: {format_number(mae_model)}")
    print(f"MAE baseline: {format_number(mae_baseline)}")
    print(f"Verdict: {verdict}")

    # JSON answer
    json_obj = {
        "predictions": {str(k): v for k, v in predictions.items()},
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-19 {json_str}")

# ----------------------------------------------------------------------
# NASA-20
# ----------------------------------------------------------------------
def compute_composite(row):
    """Composite score: Kloc+ / (Effort- * Defects-). Return -inf if any missing or zero."""
    k = row.get('Kloc+')
    e = row.get('Effort-')
    d = row.get('Defects-')
    if k is None or e is None or d is None or e == 0 or d == 0:
        return float('-inf')
    return k / (e * d)

def rank_tuple(row):
    """Tuple for lexicographic ranking: Kloc+ desc, Effort- asc, Defects- asc."""
    return (-row['Kloc+'], row['Effort-'], row['Defects-'])

def nasa_20(rows):
    # Budget
    budget = min(30, len(rows) // 5)
    repeats = 20
    random.seed(0)  # deterministic

    # Filter rows with complete goal data
    complete_indices = [idx for idx, r in enumerate(rows, start=1)
                        if r['Kloc+'] is not None and r['Effort-'] is not None and r['Defects-'] is not None]

    # Precompute composite scores for all
    composite_scores = {idx: compute_composite(rows[idx-1]) for idx in complete_indices}

    ours_best = []
    random_best = []

    for _ in range(repeats):
        # Our selection: top budget rows by composite
        sorted_by_composite = sorted(composite_scores.items(), key=lambda t: t[1], reverse=True)
        our_selected = [idx for idx, _ in sorted_by_composite[:budget]]

        # Best among ours (lexicographic ranking)
        best_ours = min(our_selected, key=lambda idx: rank_tuple(rows[idx-1]))
        ours_best.append(best_ours)

        # Random selection
        rand_selected = random.sample(complete_indices, budget)
        best_rand = min(rand_selected, key=lambda idx: rank_tuple(rows[idx-1]))
        random_best.append(best_rand)

    # Compute average composite of best rows
    ours_composite_vals = [composite_scores[idx] for idx in ours_best]
    rand_composite_vals = [composite_scores[idx] for idx in random_best]

    avg_ours = mean(ours_composite_vals) if ours_composite_vals else None
    avg_rand = mean(rand_composite_vals) if rand_composite_vals else None

    # Verdict
    if avg_ours is None or avg_rand is None:
        verdict = "no clear difference"
    else:
        if avg_ours + 1e-6 > avg_rand:
            verdict = "ours better"
        elif avg_rand + 1e-6 > avg_ours:
            verdict = "random better"
        else:
            verdict = "no clear difference"

    # Results output
    print("== NASA-20 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Ours best rows: {ours_best}")
    print(f"Random best rows: {random_best}")

    # JSON answer
    json_obj = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-20 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_19(rows)
    nasa_20(rows)

if __name__ == "__main__":
    main()

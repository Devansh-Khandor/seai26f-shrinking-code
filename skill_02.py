#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import json
import math
from statistics import mean
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    print("scikit-learn is required for this script.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_goal_col(col_name):
    return col_name.endswith('+') or col_name.endswith('-')


def is_input_col(col_name):
    return not col_name.endswith('X') and not is_goal_col(col_name)


def parse_int(val):
    """Parse binary 0/1, missing '?' -> 0."""
    if val == '?':
        return 0
    return int(val)


def parse_float(val):
    """Parse float or missing '?' -> None."""
    if val == '?':
        return None
    return float(val)


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Identify columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Choose a single goal column for prediction: the one that ends with '-'
    # (minimize) or '+' (maximize). If both exist, pick the one ending with '-'.
    target_col = None
    for c in goal_cols:
        if c.endswith('-'):
            target_col = c
            break
    if not target_col and goal_cols:
        target_col = goal_cols[0]
    if not target_col:
        print("No goal column found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # Build feature matrix X and target vector y (only for rows where target is present)
    X = []
    y = []
    row_indices = []  # 1-based row numbers for training data
    for idx, row in enumerate(rows, start=1):
        # Features
        X.append([parse_int(row[col]) for col in input_cols])
        # Target
        val = parse_float(row[target_col])
        if val is not None:
            y.append(val)
            row_indices.append(idx)

    # If no training data, cannot train a model
    if not X:
        print("No data with target present to train a model.", file=sys.stderr)
        sys.exit(1)

    # Train a RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_split=2,
        max_depth=None,
    )
    # We need to mask rows that have missing target; use only those in X and y
    X_train = [X[i - 1] for i in row_indices]
    model.fit(X_train, y)

    # ----------------------------------------------------------- #
    # APACHE-03
    # ----------------------------------------------------------- #
    print("== APACHE-03 ==")

    # Held‑out rows: those whose number is divisible by 5
    heldout_indices = [idx for idx in range(1, len(rows) + 1) if idx % 5 == 0]

    # Predict for held‑out rows
    predictions = {}
    mae_values = []  # actual vs predicted where actual present

    for idx in heldout_indices:
        row = rows[idx - 1]
        X_row = [parse_int(row[col]) for col in input_cols]
        pred = model.predict([X_row])[0]
        predictions[str(idx)] = round(pred, 3)

        # If actual is present, add to MAE calc
        actual = parse_float(row[target_col])
        if actual is not None:
            mae_values.append(abs(pred - actual))

    mae = round(mean(mae_values), 3) if mae_values else None

    # Basic output
    print(f"Total held‑out rows: {len(heldout_indices)}")
    print(f"MAE: {mae if mae is not None else 'N/A'}")

    # List first few predictions for visibility
    sample_preds = list(predictions.items())[:3]
    for r, p in sample_preds:
        print(f"Row {r} predicted {p}")

    # Machine‑checked answer
    ans_03 = {"predictions": predictions, "mae": mae if mae is not None else None}
    print("ANSWER APACHE-03", json.dumps(ans_03, separators=(',', ':')))

    # ----------------------------------------------------------- #
    # APACHE-04
    # ----------------------------------------------------------- #
    print("\n== APACHE-04 ==")

    # Compute predictions for all rows (even those missing target)
    all_predictions = {}
    for idx, row in enumerate(rows, start=1):
        X_row = [parse_int(row[col]) for col in input_cols]
        pred = model.predict([X_row])[0]
        all_predictions[idx] = pred

    # Find pairs that differ in exactly one setting
    pairs = []
    n_rows = len(rows)
    for i in range(1, n_rows + 1):
        for j in range(i + 1, n_rows + 1):
            # Count differences in input columns
            diff_cols = []
            for col in input_cols:
                vi = parse_int(rows[i - 1][col])
                vj = parse_int(rows[j - 1][col])
                if vi != vj:
                    diff_cols.append(col)
            if len(diff_cols) == 1:
                setting = diff_cols[0]
                diff = all_predictions[j] - all_predictions[i]
                low = diff - mae if mae is not None else None
                high = diff + mae if mae is not None else None
                pairs.append({
                    "a": i,
                    "b": j,
                    "setting": setting,
                    "difference": round(diff, 3),
                    "low": round(low, 3) if low is not None else None,
                    "high": round(high, 3) if high is not None else None
                })

    # Sort to keep deterministic output: by absolute diff desc, then diff asc
    pairs.sort(key=lambda x: (-abs(x["difference"]), x["difference"]))

    # Take at least 5 pairs (or all if fewer)
    selected_pairs = pairs[:max(5, len(pairs))]

    # Basic output
    print(f"Total pairs found: {len(pairs)}")
    for p in selected_pairs[:3]:
        print(f"Pair {p['a']} vs {p['b']} on {p['setting']}: diff={p['difference']}")

    # Machine‑checked answer
    ans_04 = {"pairs": selected_pairs}
    print("ANSWER APACHE-04", json.dumps(ans_04, separators=(',', ':')))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

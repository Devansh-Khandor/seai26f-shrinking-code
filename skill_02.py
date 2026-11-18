#!/usr/bin/env python3
"""
skill_02.py

Entry point for APACHE-03 and APACHE-04 requests.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    # In case sklearn is not available in the execution environment.
    print("scikit-learn is required for this skill.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Reusing utilities from skill_01

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_csv(path: str) -> List[Tuple[int, Dict[str, int]]]:
    """Load CSV, return list of (row_num, dict_of_columns)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=1):  # row numbers start at 1
            row = {}
            for k, v in raw.items():
                row[k] = parse_cell(v)
            rows.append((i, row))
    return rows


def identify_columns(rows):
    """Return goal_cols, input_cols as lists."""
    # All columns are in first row (assuming all rows have same keys)
    col_keys = rows[0][1].keys()
    goal_cols = [col for col in col_keys if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in col_keys if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]
    return goal_cols, input_cols


# --------------------------------------------------------------------------- #
# Utility: compute features matrix and target vector

def build_feature_matrix(rows, input_cols, target_col):
    """Build X, y from rows where target_col is not None."""
    X = []
    y = []
    for rnum, row in rows:
        if row[target_col] is None:
            continue  # skip rows with missing target
        feat = []
        missing = False
        for col in input_cols:
            val = row[col]
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        X.append(feat)
        y.append(row[target_col])
    return X, y


def predict_held_out(rows, input_cols, target_col):
    """Fit linear model on all non‑held‑out rows and predict on held‑out rows."""
    held_out = [rnum for rnum, _ in rows if rnum % 5 == 0]
    train_rows = [ (rnum, row) for rnum, row in rows if rnum % 5 != 0 ]
    X_train, y_train = build_feature_matrix(train_rows, input_cols, target_col)

    # If we have no training data, return empty predictions and zero MAE
    if not X_train:
        return {}, 0.0

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    predictions = {}
    abs_errors = []
    for rnum in held_out:
        # Find the row data
        row = next(row for rn, row in rows if rn == rnum)
        feat = []
        missing = False
        for col in input_cols:
            val = row[col]
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            # cannot predict; skip
            continue
        pred = lr.predict([feat])[0]
        predictions[rnum] = pred
        actual = row[target_col]
        if actual is not None:
            abs_errors.append(abs(pred - actual))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return predictions, mae


# --------------------------------------------------------------------------- #
# Utility: build pairs differing in one setting

def find_one_setting_pairs(rows, input_cols):
    """Yield tuples (row_a, row_b, differing_col) where rows differ in exactly one input."""
    # Pre‑filter rows that have complete data for inputs
    good_rows = [(rnum, row) for rnum, row in rows if all(row[col] is not None for col in input_cols)]
    n = len(good_rows)
    for i in range(n):
        rnum_i, row_i = good_rows[i]
        for j in range(i+1, n):
            rnum_j, row_j = good_rows[j]
            diffs = []
            for col in input_cols:
                if row_i[col] != row_j[col]:
                    diffs.append(col)
                    if len(diffs) > 1:
                        break
            if len(diffs) == 1:
                yield (rnum_i, rnum_j, diffs[0])


def estimate_pair_differences(rows, input_cols, target_col):
    """Use linear model coefficients to estimate performance difference for each pair."""
    # Build training data to fit model
    X_train, y_train = build_feature_matrix(rows, input_cols, target_col)
    if not X_train:
        return []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    coeffs = dict(zip(input_cols, lr.coef_))

    # Residual standard deviation
    preds = lr.predict(X_train)
    residuals = [pred - y for pred, y in zip(preds, y_train)]
    if residuals:
        std_res = math.sqrt(sum(r*r for r in residuals) / len(residuals))
    else:
        std_res = 0.0

    pairs = []
    for rnum_a, rnum_b, setting in find_one_setting_pairs(rows, input_cols):
        diff = coeffs.get(setting, 0.0)
        low = diff - std_res
        high = diff + std_res
        pairs.append({
            "a": rnum_a,
            "b": rnum_b,
            "setting": setting,
            "difference": diff,
            "low": low,
            "high": high
        })
        if len(pairs) >= 5:
            break
    return pairs


# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)
    # Pick Performance- as the target goal
    target_col = 'Performance-'

    # ------------------------------------------------------------------ APACHE-03
    print("== APACHE-03 ==")
    predictions, mae = predict_held_out(rows, input_cols, target_col)
    print("Predictions:")
    for rnum in sorted(predictions):
        print(f"Row {rnum}: {predictions[rnum]:.3f}")
    print(f"MAE: {mae:.3f}")

    answer_03 = {
        "predictions": {str(rnum): predictions[rnum] for rnum in predictions},
        "mae": mae
    }
    print("\nANSWER APACHE-03", json.dumps(answer_03))

    # ------------------------------------------------------------------ APACHE-04
    print("\n== APACHE-04 ==")
    pairs = estimate_pair_differences(rows, input_cols, target_col)
    if not pairs:
        print("No valid pairs found.")
    else:
        for p in pairs:
            print(f"Rows {p['a']} vs {p['b']} ({p['setting']}): "
                  f"diff={p['difference']:.3f} low={p['low']:.3f} high={p['high']:.3f}")

    answer_04 = {"pairs": pairs}
    print("\nANSWER APACHE-04", json.dumps(answer_04))


if __name__ == "__main__":
    main()

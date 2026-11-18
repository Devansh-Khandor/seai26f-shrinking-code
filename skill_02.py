#!/usr/bin/env python3
"""
skill_02.py

APACHE-03 – Predict performance for held‑out rows using 1‑NN.
APACHE-04 – Estimate the performance difference for pairs of
            configurations that differ by a single setting.
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of (row_no, dict_of_cells)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))
    return rows, reader.fieldnames


def is_goal(col):
    return col.endswith('+') or col.endswith('-')


def is_x(col):
    return col.endswith('X')


def to_float(v):
    """Convert a value to float or return None for missing."""
    if v == '?' or v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def to_num(v):
    """Convert a value to numeric 0/1 or None for missing."""
    if v == '?' or v is None:
        return None
    if v in ('0', '1'):
        return int(v)
    try:
        return int(v)
    except ValueError:
        return None


def compute_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


# ----------------------------------------------------------------------
# 1‑NN prediction
# ----------------------------------------------------------------------
def nearest_neighbor_prediction(test_row, train_rows, feature_cols, goal_col, fallback_mean):
    """Return predicted value for test_row."""
    best_dist = None
    best_perf = None
    for _, train_rec in train_rows:
        # Compute distance over available features
        dist_sum = 0.0
        cnt = 0
        for col in feature_cols:
            a = to_num(test_row[col])
            b = to_num(train_rec[col])
            if a is None or b is None:
                continue
            dist_sum += abs(a - b)
            cnt += 1
        if cnt == 0:
            dist = float('inf')
        else:
            dist = dist_sum / cnt
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_perf = to_float(train_rec[goal_col])
    if best_perf is None:
        return fallback_mean
    return best_perf


# ----------------------------------------------------------------------
# APACHE‑03
# ----------------------------------------------------------------------
def apache_03(rows, fieldnames, goal_col, held_out_rows, train_rows):
    # Identify feature columns
    feature_cols = [c for c in fieldnames if not is_goal(c) and not is_x(c)]
    # Prepare training data for predictions
    train_data = [(rn, rec) for rn, rec in train_rows]
    # Mean performance for fallback
    train_perf = [to_float(rec[goal_col]) for _, rec in train_data if to_float(rec[goal_col]) is not None]
    fallback_mean = compute_mean(train_perf)

    predictions = {}
    errors = []

    for rn, rec in held_out_rows:
        actual = to_float(rec[goal_col])
        if actual is None:
            continue
        pred = nearest_neighbor_prediction(rec, train_data, feature_cols, goal_col, fallback_mean)
        predictions[rn] = pred
        errors.append(abs(pred - actual))

    mae = compute_mean(errors) if errors else 0.0
    return predictions, mae


# ----------------------------------------------------------------------
# APACHE‑04
# ----------------------------------------------------------------------
def apache_04(rows, fieldnames, goal_col):
    # Build lookup by row number
    row_map = {rn: rec for rn, rec in rows}

    # Feature columns
    feature_cols = [c for c in fieldnames if not is_goal(c) and not is_x(c)]

    # Collect all pairs differing in exactly one setting
    pairs = []
    n_rows = len(rows)
    for i in range(n_rows):
        rn1, rec1 = rows[i]
        for j in range(i + 1, n_rows):
            rn2, rec2 = rows[j]
            diff_cols = []
            for col in feature_cols:
                v1 = rec1[col]
                v2 = rec2[col]
                if v1 == '?' or v2 == '?':
                    # Treat missing as no comparison; if either missing, we cannot say they differ
                    diff_cols = None
                    break
                if v1 != v2:
                    diff_cols.append(col)
            if diff_cols is None:
                continue
            if len(diff_cols) == 1:
                setting = diff_cols[0]
                # Use actual performance values if available
                perf1 = to_float(rec1[goal_col])
                perf2 = to_float(rec2[goal_col])
                if perf1 is None or perf2 is None:
                    continue
                diff = perf2 - perf1
                pairs.append({
                    'a': rn1,
                    'b': rn2,
                    'setting': setting,
                    'difference': diff,
                })
            # Stop if we have enough pairs
            if len(pairs) >= 10:
                break
        if len(pairs) >= 10:
            break

    # For uncertainty, use +/- 2*MAE from APACHE‑03 predictions
    # Compute MAE across held‑out rows first
    # (we recompute here for simplicity)
    train_rows = [r for r in rows if r[0] % 5 != 0]
    held_out_rows = [r for r in rows if r[0] % 5 == 0]
    predictions, mae = apache_03(rows, fieldnames, goal_col, held_out_rows, train_rows)
    low_factor = 2 * mae
    result_pairs = []
    for p in pairs[:5]:
        diff = p['difference']
        result_pairs.append({
            'a': p['a'],
            'b': p['b'],
            'setting': p['setting'],
            'difference': diff,
            'low': diff - low_factor,
            'high': diff + low_factor
        })
    return result_pairs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)

    # Identify goal and X columns
    goal_col = next((c for c in fieldnames if is_goal(c)), None)
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    # Separate held‑out rows (row number divisible by 5)
    held_out_rows = [r for r in rows if r[0] % 5 == 0]
    train_rows = [r for r in rows if r[0] % 5 != 0]

    # ------------------------------------------------------------------
    # APACHE-03
    # ------------------------------------------------------------------
    print("== APACHE-03 ==")
    predictions, mae = apache_03(rows, fieldnames, goal_col, held_out_rows, train_rows)
    print(f"Predicted Performance- for held‑out rows:")
    for rn in sorted(predictions):
        print(f"Row {rn}: {predictions[rn]:.4f}")
    print(f"Mean absolute error (MAE): {mae:.4f}")

    print("ANSWER APACHE-03", json.dumps(
        {"predictions": {str(rn): round(pred, 6) for rn, pred in predictions.items()},
         "mae": round(mae, 6)},
        separators=(",", ":")
    ))

    # ------------------------------------------------------------------
    # APACHE-04
    # ------------------------------------------------------------------
    print("\n== APACHE-04 ==")
    pairs = apache_04(rows, fieldnames, goal_col)
    print("Top pairs differing by one setting:")
    for p in pairs:
        print(f"Rows {p['a']} & {p['b']}: setting={p['setting']}, "
              f"difference={p['difference']:.4f}, "
              f"uncertainty=[{p['low']:.4f},{p['high']:.4f}]")
    print("ANSWER APACHE-04", json.dumps(
        {"pairs": [
            {
                "a": p["a"],
                "b": p["b"],
                "setting": p["setting"],
                "difference": round(p["difference"], 6),
                "low": round(p["low"], 6),
                "high": round(p["high"], 6)
            } for p in pairs
        ]},
        separators=(",", ":")
    ))


if __name__ == "__main__":
    main()

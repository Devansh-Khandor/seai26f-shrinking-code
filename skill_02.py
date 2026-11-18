#!/usr/bin/env python3
"""
Skill 02: Apache performance prediction and pairwise comparison

This script implements the two requested analyses:
- APACHE-03: Estimate Performance- for held‑out rows and compute MAE.
- APACHE-04: For all historical rows that differ by exactly one setting,
  estimate the performance difference and provide uncertainty bounds.

Usage:
    python skill_02.py <path-to-CSV>
"""

import json
import sys
import math
import ezr  # the provided library


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def read_csv(path: str):
    """Return header and all rows from the CSV file."""
    rows = list(ezr.csv(path))
    if not rows:
        raise ValueError("CSV file is empty")
    header = rows[0]
    data = rows[1:]
    return header, data


def goal_columns(header):
    """Return indices of columns that are goals (+/-)."""
    return [i for i, n in enumerate(header) if n.endswith(('+', '-'))]


def setting_columns(header):
    """Return indices of columns that are settings (not goal, not X)."""
    return [i for i, n in enumerate(header) if n[-1] != 'X' and not n.endswith(('+', '-'))]


def held_out_indices(n_rows):
    """Return list of 1‑based indices that are divisible by 5."""
    return [i for i in range(1, n_rows + 1) if i % 5 == 0]


def nearest_neighbor_predict(row, candidates, tbl, goal_idx):
    """Predict target for `row` using the nearest neighbor in `candidates`."""
    best_dist = float('inf')
    best_val = None
    for cand in candidates:
        if cand[goal_idx] == "?":
            continue
        d = ezr.xdist(tbl, row, cand)
        if d < best_dist:
            best_dist = d
            best_val = cand[goal_idx]
    return best_val


def compute_residual_std(predictions, actuals):
    """Compute standard deviation of residuals (abs(pred-actual))."""
    n = len(predictions)
    if n == 0:
        return 0.0
    residuals = [(abs(p - a)) for p, a in zip(predictions, actuals) if a != "?"]
    mean_res = sum(residuals) / len(residuals)
    var = sum((r - mean_res) ** 2 for r in residuals) / len(residuals)
    return math.sqrt(var)


def diff_in_one_setting(row_a, row_b, setting_indices):
    """Return the index of the setting where a and b differ, or None."""
    diff = [i for i in setting_indices if row_a[i] != row_b[i]]
    return diff[0] if len(diff) == 1 else None


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def apache_03(header, data):
    """APACHE-03: Predict Performance- for held‑out rows."""
    goal_idx = goal_columns(header)[0]  # only one goal column in this dataset
    settings = setting_columns(header)

    # Build full table to use ezr.xdist
    tbl = ezr.Tbl([header] + data)

    n_rows = len(data)
    held_out = held_out_indices(n_rows)
    held_out_set = set(held_out)

    # Separate training and held‑out rows, filtering out rows with missing goal
    training_rows = [row for idx, row in enumerate(data, start=1)
                     if idx not in held_out_set and row[goal_idx] != "?"]
    held_rows = [row for idx, row in enumerate(data, start=1)
                 if idx in held_out_set and row[goal_idx] != "?"]

    # Pre‑compute predictions for training rows (to compute residuals)
    train_preds = []
    train_actuals = []
    for i, row in enumerate(training_rows):
        # nearest neighbour excluding itself
        neighs = training_rows[:i] + training_rows[i+1:]
        pred = nearest_neighbor_predict(row, neighs, tbl, goal_idx)
        if pred is not None:
            train_preds.append(pred)
            train_actuals.append(row[goal_idx])

    residual_std = compute_residual_std(train_preds, train_actuals)

    # Predict held‑out rows
    predictions = {}
    errors = []
    for idx, row in zip(held_out, held_rows):
        pred = nearest_neighbor_predict(row, training_rows, tbl, goal_idx)
        if pred is None:
            continue
        predictions[str(idx)] = pred
        errors.append(abs(pred - row[goal_idx]))

    mae = sum(errors) / len(errors) if errors else 0.0

    # Output
    print("== APACHE-03 ==")
    for rn, val in sorted(predictions.items(), key=lambda kv: int(kv[0])):
        print(f"row {rn}: predicted={val:.2f}")
    print(f"MAE: {mae:.2f}")

    answer = {"predictions": predictions, "mae": mae}
    print(f"ANSWER APACHE-03 {json.dumps(answer, separators=(',', ':'))}")


def apache_04(header, data):
    """APACHE-04: Pairwise comparison of rows differing by one setting."""
    goal_idx = goal_columns(header)[0]
    settings = setting_columns(header)

    tbl = ezr.Tbl([header] + data)
    n_rows = len(data)
    held_out = held_out_indices(n_rows)
    held_out_set = set(held_out)

    # Training rows for predictions
    training_rows = [row for idx, row in enumerate(data, start=1)
                     if idx not in held_out_set and row[goal_idx] != "?"]

    # Pre‑compute predictions for all rows (including training) using nearest neighbour
    predictions_map = {}
    for i, row in enumerate(training_rows):
        neighs = training_rows[:i] + training_rows[i+1:]
        pred = nearest_neighbor_predict(row, neighs, tbl, goal_idx)
        if pred is not None:
            predictions_map[row] = pred
    # For rows with missing goal or no prediction, we skip
    # Compute residual std for uncertainty
    train_preds = list(predictions_map.values())
    train_actuals = [row[goal_idx] for row in training_rows if row in predictions_map]
    residual_std = compute_residual_std(train_preds, train_actuals)

    # Find all pairs differing by exactly one setting
    pairs = []
    for i in range(len(training_rows)):
        for j in range(i + 1, len(training_rows)):
            row_a = training_rows[i]
            row_b = training_rows[j]
            diff_col = diff_in_one_setting(row_a, row_b, settings)
            if diff_col is None:
                continue
            a_pred = predictions_map.get(row_a)
            b_pred = predictions_map.get(row_b)
            if a_pred is None or b_pred is None:
                continue
            diff = b_pred - a_pred
            low = diff - residual_std
            high = diff + residual_std
            pairs.append({
                "a": i + 1,  # 1-based
                "b": j + 1,
                "setting": header[diff_col],
                "difference": diff,
                "low": low,
                "high": high
            })

    # Sort by magnitude of difference and take top 5
    pairs.sort(key=lambda p: abs(p["difference"]), reverse=True)
    selected = pairs[:max(5, len(pairs))]

    # Output
    print("== APACHE-04 ==")
    for p in selected:
        print(f"rows {p['a']} & {p['b']} ({p['setting']}): diff={p['difference']:.2f} "
              f"low={p['low']:.2f} high={p['high']:.2f}")

    answer = {"pairs": selected}
    print(f"ANSWER APACHE-04 {json.dumps(answer, separators=(',', ':'))}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, data = read_csv(path)
    apache_03(header, data)
    apache_04(header, data)


if __name__ == "__main__":
    main()

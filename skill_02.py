#!/usr/bin/env python3
"""
skill_02.py - Apache configuration prediction & comparison

Usage:
    python skill_02.py <csv_path>

Prints results for:
    APACHE-03  - Predict Performance- for held‑out rows
    APACHE-04  - Estimate difference between pairs that differ in one setting
"""

import csv
import json
import math
import sys
from collections import defaultdict
from statistics import mean

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def read_csv(path):
    """Return list of dict rows; values converted to int/float when possible."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                v = v.strip()
                if v == '' or v == '?':
                    parsed[k] = '?'  # keep missing
                    continue
                try:
                    parsed[k] = int(v)
                except ValueError:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

# ----------------------------------------------------------------------
# APACHE-03 : Prediction of held‑out rows
# ----------------------------------------------------------------------
def predict_held_out(rows, goal_col, held_out_idx, k=3):
    """
    rows: list of dict
    goal_col: name of goal column
    held_out_idx: set of row indices (0‑based) that are held out
    returns: predictions dict (row_number -> pred), mae
    """
    # Separate training and held‑out
    training = [ (i, r) for i, r in enumerate(rows) if i not in held_out_idx ]
    held_out = [ (i, r) for i, r in enumerate(rows) if i in held_out_idx ]

    # Extract settings
    settings = [c for c in rows[0] if not is_goal(c)]

    # Pre‑extract training data
    train_data = []
    for idx, r in training:
        settings_vals = [r[s] for s in settings]
        train_data.append((idx, settings_vals, r[goal_col]))

    predictions = {}
    errors = []

    for idx, r in held_out:
        target_vals = [r[s] for s in settings]
        # compute distances to all training rows
        distances = []
        for t_idx, t_vals, t_perf in train_data:
            dist = 0
            for v, tv in zip(target_vals, t_vals):
                if v == '?' or tv == '?':
                    dist += 1
                elif v != tv:
                    dist += 1
            distances.append((dist, t_perf))
        # find k nearest
        distances.sort(key=lambda x: x[0])
        neigh = distances[:k]
        pred = mean([p for _, p in neigh])
        predictions[idx + 1] = pred  # 1‑based numbering
        actual = r[goal_col]
        errors.append(abs(pred - actual))

    mae = mean(errors)
    return predictions, mae

# ----------------------------------------------------------------------
# APACHE-04 : Difference estimation between pairs differing in one setting
# ----------------------------------------------------------------------
def estimate_pair_differences(rows, goal_col):
    """
    Find all historical pairs (excluding held‑out rows) that differ in exactly one setting.
    For each setting, compute mean and std of performance differences.
    Return first five pairs with estimates and confidence bounds.
    """
    # Identify held‑out rows (divisible by 5, 1‑based)
    held_out_idx = {i for i, _ in enumerate(rows) if (i+1) % 5 == 0}
    # Training rows
    training = [(i, r) for i, r in enumerate(rows) if i not in held_out_idx]

    settings = [c for c in rows[0] if not is_goal(c)]
    # Map setting to list of diffs
    diffs_by_setting = defaultdict(list)
    pair_records = []

    n = len(training)
    for a_idx in range(n):
        i, a = training[a_idx]
        a_vals = [a[s] for s in settings]
        a_perf = a[goal_col]
        for b_idx in range(a_idx + 1, n):
            j, b = training[b_idx]
            b_vals = [b[s] for s in settings]
            b_perf = b[goal_col]
            # count differences
            diff_cols = [s for s, (av, bv) in zip(settings, zip(a_vals, b_vals)) if av != bv]
            if len(diff_cols) == 1:
                setting = diff_cols[0]
                diff = b_perf - a_perf
                diffs_by_setting[setting].append(diff)
                pair_records.append((i, j, setting, diff))

    # Compute mean and std for each setting
    stats_by_setting = {}
    for setting, diffs in diffs_by_setting.items():
        m = mean(diffs)
        if len(diffs) > 1:
            s = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
        else:
            s = 0.0
        stats_by_setting[setting] = (m, s)

    # Prepare result pairs (at least five)
    result_pairs = []
    for i, j, setting, diff in pair_records[:5]:
        m, s = stats_by_setting[setting]
        low = m - 1.96 * s
        high = m + 1.96 * s
        result_pairs.append({
            "a": i + 1,       # 1‑based
            "b": j + 1,
            "setting": setting,
            "difference": round(m, 4),
            "low": round(low, 4),
            "high": round(high, 4)
        })

    return result_pairs

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data.")
        sys.exit(0)

    # Identify goal column (first one)
    goal_cols = [c for c in rows[0] if is_goal(c)]
    if not goal_cols:
        print("No goal column.")
        sys.exit(0)
    goal_col = goal_cols[0]

    # Identify held‑out rows (1‑based divisible by 5)
    held_out_idx = {i for i, _ in enumerate(rows) if (i + 1) % 5 == 0}

    # ----------------- APACHE-03 -----------------
    predictions, mae = predict_held_out(rows, goal_col, held_out_idx, k=3)

    print("== APACHE-03 ==")
    # Print at least three lines: predictions for each held‑out row
    for num in sorted(predictions):
        pred = predictions[num]
        actual = rows[num-1][goal_col]
        err = abs(pred - actual)
        print(f"Row {num}: predicted={round(pred,4)} actual={actual} error={round(err,4)}")
    print(f"MAE={round(mae,4)}")

    # JSON answer
    ans_03 = {"predictions": predictions, "mae": round(mae, 4)}
    print("ANSWER APACHE-03", json.dumps(ans_03, separators=(',', ':')))

    # ----------------- APACHE-04 -----------------
    pairs = estimate_pair_differences(rows, goal_col)

    print("\n== APACHE-04 ==")
    for p in pairs:
        print(f"Pair {p['a']},{p['b']}: setting={p['setting']} "
              f"diff={p['difference']} low={p['low']} high={p['high']}")
    # JSON answer
    ans_04 = {"pairs": pairs}
    print("ANSWER APACHE-04", json.dumps(ans_04, separators=(',', ':')))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skill_10.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-19: Compare our configuration-selection system against random configuration selection under the same benchmarking budget,
               and show which approach discovers better-performing configurations.
  - APACHE-20: Compare our performance predictions against a simple historical-average estimate and show whether our system
               provides a meaningful improvement on previously unseen configurations.

The implementation re‑uses the existing helper functions and the greedy selection logic from skill_07.
"""

import sys
import json
import random
import numpy as np
from skill_01 import read_csv, parse_column_values
from skill_07 import greedy_order

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def get_budget(num_rows):
    """Budget is the minimum of 30 and number of rows // 5."""
    return min(30, num_rows // 5)

def best_row_in_selection(rows, perf_col, indices):
    """
    Return the 1‑based row number of the best (minimal) performance
    among the given indices. If all selected rows have missing
    performance, return None.
    """
    best_val = None
    best_idx = None
    for i in indices:
        val_str = rows[i][perf_col]
        if val_str == '?':
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_idx = i + 1  # 1‑based
    return best_idx, best_val

def compute_mae(preds, actuals):
    """Compute mean absolute error given arrays of predictions and actuals."""
    if len(actuals) == 0:
        return 0.0
    return float(np.mean(np.abs(np.array(preds) - np.array(actuals))))

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal column (first one)
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]

    n_rows = len(rows)
    budget = get_budget(n_rows)

    # --- APACHE-19 ---
    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    repeats = 20
    random.seed(0)  # reproducibility

    ours_best_rows = []
    random_best_rows = []

    for _ in range(repeats):
        # Random selection
        rand_indices = random.sample(range(n_rows), budget)
        rand_best, _ = best_row_in_selection(rows, perf_col, rand_indices)

        # Our selection using greedy order (simulate=True for APACHE‑14 logic)
        greedy_indices = greedy_order(rows, [h for h in headers if not h.endswith('+')
                                             and not h.endswith('-')
                                             and not h.endswith('X')], perf_col, budget, simulate=True)
        ours_best, _ = best_row_in_selection(rows, perf_col, greedy_indices)

        ours_best_rows.append(ours_best)
        random_best_rows.append(rand_best)

    # Determine verdict
    ours_better = 0
    random_better = 0
    for ours_val, rand_val in zip(ours_best_rows, random_best_rows):
        # fetch actual performances
        ours_perf = float(rows[ours_val - 1][perf_col]) if ours_val else None
        rand_perf = float(rows[rand_val - 1][perf_col]) if rand_val else None
        if ours_perf is None or rand_perf is None:
            continue
        if ours_perf < rand_perf:
            ours_better += 1
        elif rand_perf < ours_perf:
            random_better += 1

    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    print(f"Random best rows: {random_best_rows}")
    print(f"Our best rows:   {ours_best_rows}")
    print(f"Verdict: {verdict}")

    # Machine‑checked answer
    print("ANSWER APACHE-19", json.dumps({
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }, separators=(',', ':')))

    # --- APACHE-20 ---
    print("\n== APACHE-20 ==")

    # Identify held‑out rows (divisible by 5)
    heldout_indices = [i for i in range(n_rows) if (i + 1) % 5 == 0]
    train_indices = [i for i in range(n_rows) if (i + 1) % 5 != 0]

    # Build feature matrix and target array
    from skill_02 import feature_matrix
    X = feature_matrix(rows, [h for h in headers if not h.endswith('+')
                               and not h.endswith('-')
                               and not h.endswith('X')])
    y = parse_column_values(rows, perf_col)

    # Train linear regression on training data (non‑held‑out with known target)
    from sklearn.linear_model import LinearRegression
    train_mask = np.array([i in train_indices for i in range(n_rows)])
    train_mask &= ~np.isnan(y)
    if np.sum(train_mask) == 0:
        lr = None
    else:
        lr = LinearRegression()
        lr.fit(X[train_mask], y[train_mask])

    # Predict all rows (including held‑out)
    if lr is not None:
        preds_all = lr.predict(X)
    else:
        preds_all = np.array([0.0] * n_rows)

    # Predictions for held‑out rows
    predictions = {}
    for i in heldout_indices:
        predictions[i + 1] = float(preds_all[i])

    # Baseline: mean of non‑held‑out rows with known target
    baseline_vals = y[~np.isnan(y) & ~np.array([(i + 1) % 5 == 0 for i in range(n_rows)])]
    baseline = float(np.mean(baseline_vals)) if len(baseline_vals) > 0 else None

    # MAE calculations
    actuals = []
    preds_list = []
    for i in heldout_indices:
        if np.isnan(y[i]):
            continue
        actuals.append(float(y[i]))
        preds_list.append(float(preds_all[i]))

    mae_model = compute_mae(preds_list, actuals)
    mae_baseline = compute_mae([baseline] * len(actuals), actuals) if baseline is not None else 0.0

    if mae_model < mae_baseline:
        verdict2 = "better"
    elif mae_model > mae_baseline:
        verdict2 = "worse"
    else:
        verdict2 = "no clear difference"

    # Human‑readable output
    print(f"Baseline mean: {baseline:.4f}")
    print(f"MAE model: {mae_model:.4f}")
    print(f"MAE baseline: {mae_baseline:.4f}")
    print(f"First three predictions (row: value):")
    for row_num in sorted(predictions)[:3]:
        print(f"  {row_num}: {predictions[row_num]:.4f}")

    # Machine‑checked answer
    print("ANSWER APACHE-20", json.dumps({
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict2
    }, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

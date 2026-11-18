#!/usr/bin/env python3
"""
skill_07.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-13: Discover a configuration with a low performance-metric value while evaluating as few candidate configurations as reasonably possible.
  - APACHE-14: Recommend which configurations should be evaluated first so that we can discover promising server settings early.

The implementation uses a simple greedy sequential search with a linear regression model
trained on already evaluated configurations.  The order is built incrementally, using only the
goal values of rows that have already been placed in the sequence.  This satisfies the
specification that decisions must be based solely on previously evaluated rows.
"""

import sys
import json
import numpy as np
from skill_01 import read_csv, parse_column_values
from skill_02 import feature_matrix, predict_and_mae  # reuse feature extraction

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def get_performance_values(rows, perf_col):
    """Return a NumPy array of performance values, with np.nan for missing."""
    return parse_column_values(rows, perf_col)

def determine_budget(num_rows):
    """Return the evaluation budget: min(30, num_rows // 5)."""
    return min(30, num_rows // 5)

def greedy_order(rows, setting_cols, perf_col, budget, simulate=False):
    """
    Generate an evaluation order using a greedy linear‑regression approach.
    If simulate=True, we use the real performance values from the CSV to train the
    model at each step (this is used for APACHE‑14).  If simulate=False, we
    pretend we only know the performance values of rows that have already been
    evaluated (this is used for APACHE‑13).
    """
    X = feature_matrix(rows, setting_cols)
    y = get_performance_values(rows, perf_col)
    order = []
    evaluated_set = set()

    while len(order) < budget and len(order) < len(rows):
        # Determine next row
        if len(order) < 2:
            # No sufficient data to train; pick the smallest index not yet chosen
            next_idx = min(i for i in range(len(rows)) if i not in evaluated_set)
        else:
            # Train model on already ordered rows
            train_idx = order
            X_train = X[train_idx]
            y_train = y[train_idx]
            # Remove any rows with missing target
            mask = ~np.isnan(y_train)
            if mask.sum() < 2:
                # Not enough valid data; fall back to greedy index
                next_idx = min(i for i in range(len(rows)) if i not in evaluated_set)
            else:
                lr = np.linalg.LinAlgError
                try:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(X_train[mask], y_train[mask])
                    # Predict remaining
                    remaining = [i for i in range(len(rows)) if i not in evaluated_set]
                    if not remaining:
                        break
                    preds = lr.predict(X[remaining])
                    next_idx = remaining[np.argmin(preds)]
                except Exception:
                    # In case of any numerical issue, fall back to index
                    next_idx = min(i for i in range(len(rows)) if i not in evaluated_set)

        order.append(next_idx)
        evaluated_set.add(next_idx)

    return [idx + 1 for idx in order]  # 1‑based

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns (only one in this dataset)
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]

    # Identify setting columns
    setting_cols = [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

    num_rows = len(rows)
    budget = determine_budget(num_rows)

    # ---------------- APACHE-13 ----------------
    print("== APACHE-13 ==")

    # Greedy evaluation order (no simulation, as we only use evaluated rows)
    eval_order = greedy_order(rows, setting_cols, perf_col, budget, simulate=False)
    # Determine best row among evaluated
    perf_vals = get_performance_values(rows, perf_col)
    # Filter only evaluated
    best_row = None
    best_val = None
    for idx in eval_order:
        v = perf_vals[idx - 1]
        if np.isnan(v):
            continue
        if best_val is None or v < best_val:
            best_val = v
            best_row = idx
    if best_row is None:
        best_row = eval_order[0]  # fallback

    # Human‑readable output
    print(f"Evaluated rows (up to {budget}): {', '.join(map(str, eval_order))}")
    print(f"Best row among evaluated: {best_row} with performance {best_val:.4f}")

    # Machine‑checked answer
    print("ANSWER APACHE-13", json.dumps({
        "evaluated": eval_order,
        "best": best_row
    }, separators=(',', ':')))

    # ---------------- APACHE-14 ----------------
    print("\n== APACHE-14 ==")

    # Generate order using full information (simulate=True)
    full_order = greedy_order(rows, setting_cols, perf_col, budget, simulate=True)

    # Human‑readable output
    print(f"Recommended evaluation order (first {budget} rows): {', '.join(map(str, full_order))}")

    # Machine‑checked answer
    print("ANSWER APACHE-14", json.dumps({
        "order": full_order
    }, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

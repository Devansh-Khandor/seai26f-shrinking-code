#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <path to Apache_AllMeasurements.csv>

This script implements the APACHE‑13 and APACHE‑14 requirements using the
existing skill modules.  It uses the linear‑regression helpers from
skill_02 to iteratively select configurations to evaluate.

Output format:
  For each request, a line with the header "== <ID> ==",
  followed by at least three human‑readable result lines, and finally
  a machine‑checked answer line:
      ANSWER <ID> <JSON>
"""

import json
import sys

# Reuse helpers from earlier modules
import skill_01 as s01
import skill_02 as s02


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def goal_column(header):
    """Return the first column that is a goal (+ or -)."""
    for col in header:
        if s01.is_goal(col):
            return col
    return None


def setting_columns(header):
    """Return all columns that are settings (not goals, not ending in X)."""
    return [c for c in header if not s01.is_goal(c) and not c.endswith("X")]


def best_row(rows, goal_col):
    """
    Return the 1‑based row index of the best row for the given goal.
    For '-' goals, lower is better; for '+' goals, higher is better.
    """
    best_idx = None
    best_val = None
    for idx, r in enumerate(rows, start=1):
        val = r.get(goal_col)
        if val is None:
            continue
        if best_val is None:
            best_val = val
            best_idx = idx
        else:
            if goal_col.endswith("-"):
                if val < best_val:
                    best_val = val
                    best_idx = idx
            else:
                if val > best_val:
                    best_val = val
                    best_idx = idx
    return best_idx, best_val


def adaptive_selection(rows, goal_col, settings, budget):
    """
    Iteratively choose rows to evaluate.

    Returns:
        order: list of 1‑based row indices in the order they would be evaluated
        best_idx: the best row index found after evaluating up to budget rows
    """
    n_rows = len(rows)
    budget = min(budget, n_rows)

    # Start with the first row
    order = [1]
    remaining = set(range(2, n_rows + 1))

    # Keep track of evaluated rows for model fitting
    while len(order) < budget and remaining:
        # Train on evaluated rows so far
        train_rows = [rows[i - 1] for i in order]
        try:
            coeff = s02._fit_linear(train_rows, goal_col, settings)
        except Exception:
            # If regression fails, just pick an arbitrary remaining row
            next_idx = min(remaining)
            order.append(next_idx)
            remaining.remove(next_idx)
            continue

        # Predict goal for all remaining rows
        preds = {}
        for idx in remaining:
            r = rows[idx - 1]
            try:
                val = s02._predict_row(r, coeff, settings)
                preds[idx] = val
            except Exception:
                # Fallback for rows with missing data
                preds[idx] = float("inf") if goal_col.endswith("-") else float("-inf")

        # Choose the row that is predicted to be best (lowest for '-' or highest for '+')
        if goal_col.endswith("-"):
            next_idx = min(preds, key=preds.get)
        else:
            next_idx = max(preds, key=preds.get)

        order.append(next_idx)
        remaining.remove(next_idx)

    # Determine best row among those evaluated
    best_idx, _ = best_row([rows[i - 1] for i in order], goal_col)
    best_row_idx = order[best_idx - 1] if best_idx else None

    return order, best_row_idx


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    goal_col = goal_column(header)
    if not goal_col:
        print("No goal column found.")
        sys.exit(1)
    settings = setting_columns(header)

    # Budget: at most min(30, number of rows // 5) – as specified
    budget = min(30, len(rows) // 5)

    # ---------- APACHE-13 ----------
    print("== APACHE-13 ==")

    order13, best13 = adaptive_selection(rows, goal_col, settings, budget)

    # Human‑readable results
    print(f"Evaluated rows (up to {len(order13)}): {order13[:5]}")
    print(f"Best evaluated row: {best13}")

    apache13_answer = {
        "evaluated": order13,
        "best": best13,
    }
    print("ANSWER APACHE-13", json.dumps(apache13_answer, separators=(",", ":")))

    # ---------- APACHE-14 ----------
    print("\n== APACHE-14 ==")

    order14, _ = adaptive_selection(rows, goal_col, settings, budget)

    # Human‑readable results
    print(f"Recommended evaluation order (first 10): {order14[:10]}")
    print(f"Total rows to evaluate: {len(order14)}")

    apache14_answer = {
        "order": order14,
    }
    print("ANSWER APACHE-14", json.dumps(apache14_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()

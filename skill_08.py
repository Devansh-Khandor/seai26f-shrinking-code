#!/usr/bin/env python3
"""
skill_08.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-15 – For a project proposal predicted to require excessive effort, identify plausible
adjustments to adjustable planning or process characteristics that might reduce its predicted effort.
NASA-16 – For a project proposal with a high predicted defect count, identify the smallest plausible
combination of changes to adjustable planning or process characteristics that could improve its
predicted quality, while explaining potential tradeoffs.
"""

import csv
import json
import sys
from itertools import combinations

# Import helper utilities from the previous skill (skill_06)
# They are defined in skill_06.py and are fully compatible.
# This avoids re‑implementing parsing, model training, and prediction logic.
from skill_06 import load_csv, train_model, get_feature_columns, predict_row, parse_value

# --------------------------------------------------------------------------- #
# Main analysis functions
# --------------------------------------------------------------------------- #

def find_adjustable_columns(all_columns):
    """Return list of columns that can be altered."""
    # Any column that is not a goal (no + or -), not a non‑input X column,
    # and not Kloc+ (the only goal that should be fixed)
    return [c for c in all_columns
            if not c.endswith('X')
            and not (c.endswith('+') or c.endswith('-'))
            and c != 'Kloc+']

def collect_unique_values(rows, adjustable_cols):
    """Build a mapping from column to the set of unique values present in the data."""
    values_by_col = {c: set() for c in adjustable_cols}
    for r in rows:
        for c in adjustable_cols:
            val = r.get(c)
            if val is not None:
                values_by_col[c].add(val)
    return values_by_col

def best_single_change(row, model, feature_cols, adjustable_cols, values_by_col, target_pred_func):
    """
    For each adjustable column, try every distinct value in the column
    and find the change that gives the best (lowest) prediction.
    Returns a tuple (col, new_val, new_pred) or None if no improvement.
    """
    best = None
    baseline_pred = target_pred_func(row)
    for col in adjustable_cols:
        current = row[col]
        for new_val in values_by_col[col]:
            if new_val == current:
                continue
            tmp = row.copy()
            tmp[col] = new_val
            new_pred = target_pred_func(tmp)
            if new_pred is None:
                continue
            if new_pred < baseline_pred and (best is None or new_pred < best[2]):
                best = (col, new_val, new_pred)
    return best

def best_multiple_changes(row, model, feature_cols, adjustable_cols, values_by_col, target_pred_func, max_changes=3):
    """
    Greedily pick up to max_changes columns that give the largest single‑column improvement.
    """
    improvements = []
    baseline_pred = target_pred_func(row)
    for col in adjustable_cols:
        current = row[col]
        best_val = None
        best_pred = baseline_pred
        for new_val in values_by_col[col]:
            if new_val == current:
                continue
            tmp = row.copy()
            tmp[col] = new_val
            pred = target_pred_func(tmp)
            if pred is None:
                continue
            if pred < best_pred:
                best_pred = pred
                best_val = new_val
        if best_val is not None:
            reduction = baseline_pred - best_pred
            improvements.append((reduction, col, best_val, best_pred))
    # Take top reductions
    improvements.sort(reverse=True, key=lambda t: t[0])
    changes = {}
    new_row = row.copy()
    new_pred = baseline_pred
    for _, col, new_val, pred_val in improvements[:max_changes]:
        changes[col] = [row[col], new_val]
        new_row[col] = new_val
        new_pred = pred_val  # since we only apply one change at a time, this is not exact
    return changes, new_row

def main(csv_path):
    # Load data
    rows, all_columns = load_csv(csv_path)

    # Separate history and held‑out
    history = [r for r in rows if r['row_num'] % 5 != 0]
    held_out = [r for r in rows if r['row_num'] % 5 == 0]

    # Train models
    effort_feature_cols = get_feature_columns(all_columns, exclude=('Effort-', 'Defects-'))
    effort_model, effort_stats = train_model(history, effort_feature_cols, 'Effort-')

    defect_feature_cols = get_feature_columns(all_columns, exclude=('Defects-', 'Effort-'))
    defect_model, defect_stats = train_model(history, defect_feature_cols, 'Defects-')

    # Helper lambdas for predictions
    effort_pred = lambda r: predict_row(effort_model, r, effort_feature_cols)
    defect_pred = lambda r: predict_row(defect_model, r, defect_feature_cols)

    # Compute predictions for held‑out rows
    effort_preds = {r['row_num']: effort_pred(r) for r in held_out}
    defect_preds = {r['row_num']: defect_pred(r) for r in held_out}

    # Identify the most extreme projects
    max_effort_row_num = max(effort_preds, key=effort_preds.get)
    max_defect_row_num = max(defect_preds, key=defect_preds.get)

    eff_project = next(r for r in held_out if r['row_num'] == max_effort_row_num)
    defect_project = next(r for r in held_out if r['row_num'] == max_defect_row_num)

    # Adjustable columns and unique values
    adjustable_cols = find_adjustable_columns(all_columns)
    values_by_col = collect_unique_values(rows, adjustable_cols)

    # --------------------------------------------------------------------------- #
    # NASA-15 – Reduce effort
    # --------------------------------------------------------------------------- #
    baseline_effort = effort_preds[max_effort_row_num]
    best_change = best_single_change(
        eff_project, effort_model, effort_feature_cols,
        adjustable_cols, values_by_col, effort_pred)

    print("== NASA-15 ==")
    print(f"Project row {max_effort_row_num} predicted effort before: {baseline_effort:.2f}")

    if best_change:
        col, new_val, new_effort = best_change
        changes_dict = {col: [eff_project[col], new_val]}
        print(f"Change {col}: {eff_project[col]} -> {new_val}")
        print(f"Predicted effort after: {new_effort:.2f}")
    else:
        changes_dict = {}
        print("No single adjustment improves effort.")
        new_effort = baseline_effort

    answer_15 = {
        "project": max_effort_row_num,
        "changes": changes_dict,
        "predicted_effort": {"before": baseline_effort, "after": new_effort}
    }
    print("ANSWER NASA-15", json.dumps(answer_15, ensure_ascii=False))

    # --------------------------------------------------------------------------- #
    # NASA-16 – Reduce defects (and show effort trade‑off)
    # --------------------------------------------------------------------------- #
    baseline_defect = defect_preds[max_defect_row_num]
    baseline_effort_before = effort_preds[max_defect_row_num]

    # Find best up to 3 changes
    changes, temp_row = best_multiple_changes(
        defect_project, defect_model, defect_feature_cols,
        adjustable_cols, values_by_col, defect_pred, max_changes=3)

    # Compute new predictions after applying changes
    new_defect = defect_pred(temp_row)
    new_effort = effort_pred(temp_row)

    print("\n== NASA-16 ==")
    print(f"Project row {max_defect_row_num} predicted defects before: {baseline_defect:.2f}")
    print(f"Predicted effort before: {baseline_effort_before:.2f}")

    if changes:
        for col, vals in changes.items():
            print(f"Change {col}: {vals[0]} -> {vals[1]}")
        print(f"Predicted defects after: {new_defect:.2f}")
        print(f"Predicted effort after: {new_effort:.2f}")
    else:
        print("No adjustments improve defect count.")
        new_defect = baseline_defect
        new_effort = baseline_effort_before

    answer_16 = {
        "project": max_defect_row_num,
        "changes": changes,
        "predicted_defects": {"before": baseline_defect, "after": new_defect},
        "predicted_effort": {"before": baseline_effort_before, "after": new_effort}
    }
    print("ANSWER NASA-16", json.dumps(answer_16, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

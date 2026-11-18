#!/usr/bin/env python3
"""
Skill 08 – NASA-15 & NASA-16

NASA-15: Identify a plausible change to reduce the predicted effort of a project proposal that is expected to need excessive effort.
NASA-16: Identify a small set of changes to reduce the predicted defect count of a project proposal, while showing the effort trade‑off.

The script is self‑contained, uses only the standard library and scikit‑learn/numpy,
and follows the output conventions described in the project brief.
"""

import sys
import csv
import json
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Utility – parsing numeric values
# ----------------------------------------------------------------------
def parse_float(v):
    """Return float(v) or None if missing or cannot be parsed."""
    if v is None or v == '':
        return None
    v = str(v).strip()
    if v == '?' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(path):
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames


# ----------------------------------------------------------------------
# Helpers – identify columns
# ----------------------------------------------------------------------
def goal_cols(fieldnames):
    return [c for c in fieldnames if c.endswith('+') or c.endswith('-')]


def excluded_cols(fieldnames):
    return [c for c in fieldnames if c.endswith('X')]


def numeric_feature_cols(fieldnames, data):
    """
    Return a list of columns that are numeric in every non‑missing value
    across the whole dataset.
    """
    candidate = [c for c in fieldnames if c not in goal_cols(fieldnames) and c not in excluded_cols(fieldnames)]
    numeric = []
    for col in candidate:
        ok = True
        for _, row in data:
            val = row.get(col)
            if val is None or val == '?' or val == '':
                continue
            if parse_float(val) is None:
                ok = False
                break
        if ok:
            numeric.append(col)
    return numeric


# ----------------------------------------------------------------------
# Build training data for a given target
# ----------------------------------------------------------------------
def build_X_y(data, target, feature_cols):
    X = []
    y = []
    indices = []
    for idx, row in data:
        t = parse_float(row.get(target))
        if t is None:
            continue
        vec = []
        missing = False
        for col in feature_cols:
            val = parse_float(row.get(col))
            if val is None:
                missing = True
                break
            vec.append(val)
        if missing:
            continue
        X.append(vec)
        y.append(t)
        indices.append(idx)
    return np.array(X, dtype=float), np.array(y, dtype=float), indices


# ----------------------------------------------------------------------
# Predict for all rows that have full feature values
# ----------------------------------------------------------------------
def predict_all(data, target, model, feature_cols):
    preds = {}
    for idx, row in data:
        vec = []
        missing = False
        for col in feature_cols:
            val = parse_float(row.get(col))
            if val is None:
                missing = True
                break
            vec.append(val)
        if missing:
            continue
        preds[idx] = float(model.predict(np.array([vec]))[0])
    return preds


# ----------------------------------------------------------------------
# Find best single change that reduces a target value
# ----------------------------------------------------------------------
def best_single_change(row_idx, data, target_name, preds, feature_cols, model):
    """
    Return (col, new_val, before, after, delta) with the largest reduction.
    If no improvement, return (None, None, before, before, 0).
    """
    row = next(r for i, r in data if i == row_idx)
    before = preds[row_idx]
    vec_orig = [parse_float(row.get(c)) for c in feature_cols]
    best = (None, None, before, before, 0.0)

    # Gather distinct values for each column (excluding current)
    col_values = defaultdict(set)
    for _, r in data:
        for c in feature_cols:
            val = r.get(c)
            fval = parse_float(val)
            if fval is not None:
                col_values[c].add(fval)
    for col in feature_cols:
        cur_val = vec_orig[feature_cols.index(col)]
        for new_val in col_values[col]:
            if new_val == cur_val:
                continue
            vec_new = list(vec_orig)
            vec_new[feature_cols.index(col)] = new_val
            new_pred = float(model.predict(np.array([vec_new]))[0])
            delta = before - new_pred
            if delta > best[4]:
                best = (col, new_val, before, new_pred, delta)
    return best


# ----------------------------------------------------------------------
# Find up to 3 changes that reduce defects
# ----------------------------------------------------------------------
def best_multi_changes(row_idx, data, target_name, preds, feature_cols, model, max_changes=3):
    """
    Greedy selection of up to max_changes columns that each reduce the
    target value the most when changed independently.
    Returns list of (col, new_val, before, after) tuples and final after value.
    """
    row = next(r for i, r in data if i == row_idx)
    vec = [parse_float(row.get(c)) for c in feature_cols]
    before = preds[row_idx]
    changes = []

    # Precompute distinct values per column
    col_values = defaultdict(set)
    for _, r in data:
        for c in feature_cols:
            val = r.get(c)
            fval = parse_float(val)
            if fval is not None:
                col_values[c].add(fval)

    remaining = feature_cols[:]
    cur_vec = list(vec)
    cur_before = before

    for _ in range(max_changes):
        best = (None, None, None, None, 0.0)
        for col in remaining:
            cur_val = cur_vec[feature_cols.index(col)]
            for new_val in col_values[col]:
                if new_val == cur_val:
                    continue
                new_vec = list(cur_vec)
                new_vec[feature_cols.index(col)] = new_val
                new_pred = float(model.predict(np.array([new_vec]))[0])
                delta = cur_before - new_pred
                if delta > best[4]:
                    best = (col, new_val, cur_before, new_pred, delta)
        if best[0] is None:
            break  # no further improvement
        # apply best change
        changes.append((best[0], best[1], best[2], best[3]))
        cur_vec[feature_cols.index(best[0])] = best[1]
        cur_before = best[3]
        remaining.remove(best[0])

    after = cur_before
    return changes, after


# ----------------------------------------------------------------------
# Main logic for NASA-15 & NASA-16
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    # Identify numeric adjustable features
    feat_cols = numeric_feature_cols(fieldnames, data)

    if not feat_cols:
        # No usable numeric features – fall back to empty results
        print("== NASA-15 ==")
        print("No numeric features available for prediction.")
        print("ANSWER NASA-15 {}")
        print("\n== NASA-16 ==")
        print("No numeric features available for prediction.")
        print("ANSWER NASA-16 {}")
        return

    # Train models for Effort- and Defects-
    X_eff, y_eff, _ = build_X_y(data, 'Effort-', feat_cols)
    X_def, y_def, _ = build_X_y(data, 'Defects-', feat_cols)

    model_eff = LinearRegression()
    if X_eff.size:
        model_eff.fit(X_eff, y_eff)
    model_def = LinearRegression()
    if X_def.size:
        model_def.fit(X_def, y_def)

    # Predict for all rows
    eff_preds = predict_all(data, 'Effort-', model_eff, feat_cols)
    def_preds = predict_all(data, 'Defects-', model_def, feat_cols)

    # ------------------------------------------------------------------
    # NASA-15: Project with highest predicted effort
    # ------------------------------------------------------------------
    if eff_preds:
        max_eff_row = max(eff_preds, key=lambda k: eff_preds[k])
    else:
        max_eff_row = data[0][0]  # fallback

    # Find best single change
    col_eff, new_val_eff, eff_before, eff_after, _ = best_single_change(
        max_eff_row, data, 'Effort-', eff_preds, feat_cols, model_eff
    )
    # Prepare changes dict
    changes_eff = {}
    if col_eff is not None:
        orig_val = next(r for i, r in data if i == max_eff_row)[col_eff]
        # keep numeric type if applicable
        if parse_float(orig_val) is not None:
            orig_val = float(orig_val)
        changes_eff[col_eff] = [orig_val, new_val_eff]

    # Output
    print("== NASA-15 ==")
    print(f"Project {max_eff_row} predicted effort before: {eff_before:.2f}")
    print(f"Project {max_eff_row} predicted effort after: {eff_after:.2f}")
    print(f"Changes: {json.dumps(changes_eff)}")

    # Machine‑checked answer
    ans15 = {
        "project": max_eff_row,
        "changes": changes_eff,
        "predicted_effort": {"before": round(eff_before, 2), "after": round(eff_after, 2)}
    }
    print(f"ANSWER NASA-15 {json.dumps(ans15, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # NASA-16: Project with highest predicted defects
    # ------------------------------------------------------------------
    if def_preds:
        max_def_row = max(def_preds, key=lambda k: def_preds[k])
    else:
        max_def_row = data[0][0]  # fallback

    # Find up to 3 best changes
    changes_def_list, def_after = best_multi_changes(
        max_def_row, data, 'Defects-', def_preds, feat_cols, model_def, max_changes=3
    )
    changes_def = {}
    for col, new_val, before_val, after_val in changes_def_list:
        orig_val = next(r for i, r in data if i == max_def_row)[col]
        if parse_float(orig_val) is not None:
            orig_val = float(orig_val)
        changes_def[col] = [orig_val, new_val]

    # Effort predictions before and after applying changes
    eff_before_all = eff_preds.get(max_def_row, None)
    if eff_before_all is None:
        eff_before_all = eff_preds.get(max_eff_row, 0.0)  # fallback
    # Build feature vector for the project
    vec = [parse_float(next(r for i, r in data if i == max_def_row).get(c)) for c in feat_cols]
    # Apply changes
    for col, new_val in changes_def.items():
        idx = feat_cols.index(col)
        vec[idx] = new_val
    eff_after_all = float(model_eff.predict(np.array([vec]))[0]) if eff_before_all is not None else None
    def_before_all = def_preds.get(max_def_row, None)
    if def_before_all is None:
        def_before_all = def_preds.get(max_def_row, 0.0)  # fallback

    # Output
    print("\n== NASA-16 ==")
    print(f"Project {max_def_row} predicted defects before: {def_before_all:.2f}")
    print(f"Project {max_def_row} predicted defects after: {def_after:.2f}")
    print(f"Project {max_def_row} predicted effort before: {eff_before_all:.2f}")
    print(f"Project {max_def_row} predicted effort after: {eff_after_all:.2f}")
    print(f"Changes: {json.dumps(changes_def)}")

    # Machine‑checked answer
    ans16 = {
        "project": max_def_row,
        "changes": changes_def,
        "predicted_defects": {"before": round(def_before_all, 2), "after": round(def_after, 2)},
        "predicted_effort": {"before": round(eff_before_all, 2), "after": round(eff_after_all, 2)}
    }
    print(f"ANSWER NASA-16 {json.dumps(ans16, separators=(',', ':'))}")


if __name__ == "__main__":
    main()

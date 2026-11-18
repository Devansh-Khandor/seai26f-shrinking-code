#!/usr/bin/env python3
"""
skill_06.py <csv path>

Implements:
    NASA-11: When the system estimates the effort required for a proposed project,
              explain the estimate using relevant historical projects and their
              recorded characteristics.

    NASA-12: When two project proposals receive different risk assessments,
              explain the main differences that contributed to those assessments
              and identify the important tradeoffs.

The script prints each request header, at least three human‑readable result lines,
and then the machine‑checked JSON answer.
"""

import csv
import json
import math
import sys

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def is_numeric(v):
    """Return True if string can be parsed to float."""
    if v == '?' or v is None:
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False

def to_float(v):
    """Convert a string to float, return None on '?'."""
    return None if v == '?' else float(v)

# ------------------------------------------------------------------
# Distance between two rows in feature space
# ------------------------------------------------------------------
def distance(row_a, row_b, feature_cols):
    """Euclidean distance with categorical treated as 0/1."""
    sq_sum = 0.0
    for col in feature_cols:
        va = row_a[col]
        vb = row_b[col]
        if va == '?' or vb == '?':
            continue
        if is_numeric(va) and is_numeric(vb):
            diff = float(va) - float(vb)
            sq_sum += diff * diff
        else:
            # categorical
            sq_sum += 0.0 if va == vb else 1.0
    return math.sqrt(sq_sum)

# ------------------------------------------------------------------
# Nearest‑neighbour search
# ------------------------------------------------------------------
def nearest_neighbors(target_row, rows, feature_cols, k=3):
    """Return list of (row_no, row_dict) nearest to target_row."""
    dists = []
    for r_no, r in rows:
        d = distance(target_row, r, feature_cols)
        dists.append((d, r_no, r))
    dists.sort(key=lambda x: x[0])
    return [(r_no, r) for _, r_no, r in dists[:k]]

# ------------------------------------------------------------------
# Prediction of a goal using nearest neighbours
# ------------------------------------------------------------------
def predict_goal(target_row, rows, feature_cols, goal_col):
    """Return mean of goal_col for nearest neighbours that have the goal."""
    neighs = nearest_neighbors(target_row, rows, feature_cols, k=3)
    vals = [to_float(n[2][goal_col]) for _, n in neighs if to_float(n[2][goal_col]) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV rows preserving raw string values
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows_raw = [row for row in reader]

    n_rows = len(rows_raw)
    # Row numbers are 1‑based
    rows_with_no = list(enumerate(rows_raw, start=1))

    # Identify goal columns and feature columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    feature_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    # Separate held‑out rows (divisible by 5) and the rest
    held_out = [(no, r) for no, r in rows_with_no if no % 5 == 0]
    non_held = [(no, r) for no, r in rows_with_no if no % 5 != 0]

    # ------------------------------------------------------------------
    # NASA-11
    # ------------------------------------------------------------------
    # Pick first held‑out row
    proj_no, proj_row = held_out[0]

    # Predict Effort- using nearest neighbours
    eff_pred = predict_goal(proj_row, non_held, feature_cols, 'Effort-')
    if eff_pred is None:
        eff_pred = 0.0  # fallback

    # Nearest neighbours for explanation
    neigh_list = nearest_neighbors(proj_row, non_held, feature_cols, k=3)
    neighbors = []
    for n_no, n_row in neigh_list:
        # Copy raw dict for output
        neighbors.append({
            "row": n_no,
            "values": n_row.copy()
        })

    nasa11_ans = {
        "project": proj_no,
        "estimate": eff_pred,
        "neighbors": neighbors
    }

    # ------------------------------------------------------------------
    # NASA-12
    # ------------------------------------------------------------------
    # For each held‑out row, compute predicted Effort- and Defects-
    predictions = {}
    for h_no, h_row in held_out:
        pred_eff = predict_goal(h_row, non_held, feature_cols, 'Effort-')
        pred_def = predict_goal(h_row, non_held, feature_cols, 'Defects-')
        # Handle missing
        if pred_eff is None:
            pred_eff = 0.0
        if pred_def is None:
            pred_def = 0.0
        # Score: higher better => lower risk => negative sum
        score = -(pred_eff + pred_def)
        predictions[h_no] = {
            "effort": pred_eff,
            "defects": pred_def,
            "score": score
        }

    # Find pair with largest score difference
    best_pair = None
    best_diff = -1
    sorted_keys = sorted(predictions.keys())
    for i in range(len(sorted_keys)):
        for j in range(i+1, len(sorted_keys)):
            a_no = sorted_keys[i]
            b_no = sorted_keys[j]
            diff = abs(predictions[a_no]["score"] - predictions[b_no]["score"])
            if diff > best_diff:
                best_diff = diff
                best_pair = (a_no, b_no)

    a_no, b_no = best_pair
    a_pred = predictions[a_no]
    b_pred = predictions[b_no]
    # Preferred is the one with higher score
    preferred = a_no if a_pred["score"] >= b_pred["score"] else b_no

    # Differences in feature values
    diff_list = []
    row_a = dict(rows_with_no[a_no-1])
    row_b = dict(rows_with_no[b_no-1])
    for col in feature_cols:
        val_a = row_a[col]
        val_b = row_b[col]
        if val_a != val_b:
            diff_list.append({"feature": col, "a": val_a, "b": val_b})
    # Ensure at least one
    if not diff_list:
        # add first differing feature if any
        for col in feature_cols:
            if row_a[col] != row_b[col]:
                diff_list.append({"feature": col, "a": row_a[col], "b": row_b[col]})
                break

    # Tradeoffs: goal columns where non‑preferred row has better value
    tradeoffs = []
    non_pref_no = b_no if preferred == a_no else a_no
    non_pref_pred = predictions[non_pref_no]
    for gcol in goal_cols:
        # lower is better for -, higher for +
        is_greater = gcol.endswith('+')
        val_non = non_pref_pred["effort"] if gcol == 'Effort-' else non_pref_pred["defects"]
        val_pref = predictions[preferred]["effort"] if gcol == 'Effort-' else predictions[preferred]["defects"]
        better = (val_non < val_pref) if not is_greater else (val_non > val_pref)
        if better:
            tradeoffs.append(gcol)

    nasa12_ans = {
        "a": a_no,
        "b": b_no,
        "preferred": preferred,
        "scores": {
            "a": a_pred["score"],
            "b": b_pred["score"]
        },
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-11
    print("== NASA-11 ==")
    print(f"Project row: {proj_no}")
    print(f"Estimated Effort-: {eff_pred:.2f}")
    print("Nearest neighbours:")
    for n in neighbors[:3]:
        print(f"  Row {n['row']}")
    print(f"ANSWER NASA-11 {json.dumps(nasa11_ans, separators=(',', ':'))}")

    # NASA-12
    print("\n== NASA-12 ==")
    print(f"Comparing held‑out rows {a_no} (score {a_pred['score']:.4f}) and {b_no} (score {b_pred['score']:.4f})")
    print(f"Preferred row: {preferred}")
    print(f"Feature differences (at least one):")
    for d in diff_list[:3]:
        print(f"  {d['feature']}: row {a_no}={d['a']}, row {b_no}={d['b']}")
    print(f"Tradeoffs where non‑preferred is better: {tradeoffs}")
    print(f"ANSWER NASA-12 {json.dumps(nasa12_ans, separators=(',', ':'))}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

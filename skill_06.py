#!/usr/bin/env python3
"""
Skill 06 – NASA-11 & NASA-12

NASA-11:
  Estimate the Effort- for a held‑out project and explain the estimate
  using the three nearest historical projects (neighbors).

NASA-12:
  Compare two projects with very different risk assessments
  (based on predicted Effort-).  Identify the main differences and
  the trade‑offs between goal columns.
"""

import sys
import csv
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_float(val):
    """Convert a CSV cell into a float, handling missing data."""
    if val is None or val == '' or val.strip() == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def read_csv(path):
    """Return list of (row_idx, row_dict) tuples and the fieldnames."""
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames

def get_goal_columns(fieldnames):
    return [c for c in fieldnames if c.endswith('+') or c.endswith('-')]

def get_feature_columns(fieldnames):
    """All columns except X columns and goal columns."""
    return [c for c in fieldnames if not c.endswith('X') and not (c.endswith('+') or c.endswith('-'))]

def get_values_dict(row):
    """Return dict of values where numeric ones are floats, others are strings."""
    out = {}
    for k, v in row.items():
        fv = parse_float(v)
        out[k] = fv if fv is not None else v
    return out

def build_X_y(data, target, exclude_set):
    """Build training X, y, indices excluding rows in exclude_set."""
    X_list = []
    y_list = []
    idx_list = []
    for idx, row in data:
        if idx in exclude_set:
            continue
        t_val = parse_float(row.get(target))
        if t_val is None:
            continue
        feat_vals = []
        missing = False
        for col in get_feature_columns(row):
            f_val = parse_float(row.get(col))
            if f_val is None:
                missing = True
                break
            feat_vals.append(f_val)
        if missing:
            continue
        X_list.append(feat_vals)
        y_list.append(t_val)
        idx_list.append(idx)
    if not X_list:
        return np.array([], dtype=float), np.array([], dtype=float), []
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float), idx_list

def nearest_neighbors(row_idx, row_feat, train_feats, train_indices, k=3):
    """Return k nearest neighbors (indices) based on Euclidean distance."""
    dists = []
    for i, feat in enumerate(train_feats):
        # compute distance over numeric columns where both have values
        diff = (row_feat - feat)**2
        d = math.sqrt(np.sum(diff))
        dists.append((d, train_indices[i]))
    dists.sort(key=lambda x: x[0])
    return [idx for _, idx in dists[:k]]

# ----------------------------------------------------------------------
# NASA-11 implementation
# ----------------------------------------------------------------------
def nasa_11(data, fieldnames):
    print("== NASA-11 ==")

    # Identify held‑out rows
    heldout = {idx for idx, _ in data if idx % 5 == 0}
    # Use all non‑held‑out rows as training
    X_train, y_train, train_idxs = build_X_y(data, 'Effort-', heldout)

    # Build feature matrix for training rows
    feature_cols = get_feature_columns(data[0][1])
    train_feats = []
    for idx in train_idxs:
        row = dict(data[idx-1][1])  # data indexed from 1
        feat_vals = []
        for col in feature_cols:
            f_val = parse_float(row.get(col))
            feat_vals.append(f_val)
        train_feats.append(np.array(feat_vals, dtype=float))
    train_feats = np.array(train_feats)

    model = LinearRegression()
    if X_train.size:
        model.fit(X_train, y_train)

    # Find the first held‑out row that can be predicted
    proj_idx = None
    proj_pred = None
    proj_feat = None
    for idx, row in data:
        if idx in heldout:
            # build feature vector
            feat_vals = []
            missing = False
            for col in feature_cols:
                f_val = parse_float(row.get(col))
                if f_val is None:
                    missing = True
                    break
                feat_vals.append(f_val)
            if missing:
                continue
            proj_idx = idx
            proj_feat = np.array(feat_vals, dtype=float)
            if X_train.size:
                proj_pred = float(model.predict(np.array([proj_feat]))[0])
            break

    if proj_idx is None:
        print("No suitable held‑out row found for NASA‑11.")
        print("ANSWER NASA-11 {}")
        return

    # Find neighbors
    neighbor_idxs = nearest_neighbors(proj_idx, proj_feat, train_feats, train_idxs, k=3)
    neighbors = []
    for n_idx in neighbor_idxs:
        # Build values dict
        n_row = dict(data[n_idx-1][1])
        neighbors.append({"row": n_idx, "values": get_values_dict(n_row)})

    # Print human readable output
    print(f"Project {proj_idx}: estimated Effort- = {proj_pred:.2f}")
    print(f"Nearest historical projects:")
    for n in neighbors:
        print(f"  Row {n['row']}: Effort- = {n['values'].get('Effort-')}")
    # Machine‑checked answer
    ans = {
        "project": proj_idx,
        "estimate": round(proj_pred, 2),
        "neighbors": neighbors
    }
    print(f"ANSWER NASA-11 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-12 implementation
# ----------------------------------------------------------------------
def nasa_12(data, fieldnames):
    print("\n== NASA-12 ==")

    # Identify held‑out rows (used only for training, but predictions for all)
    heldout = {idx for idx, _ in data if idx % 5 == 0}
    X_train, y_train, train_idxs = build_X_y(data, 'Effort-', heldout)

    # Build training features (full numeric vector)
    feature_cols = get_feature_columns(data[0][1])
    train_feats = []
    for idx in train_idxs:
        row = dict(data[idx-1][1])
        feat_vals = []
        for col in feature_cols:
            f_val = parse_float(row.get(col))
            feat_vals.append(f_val)
        train_feats.append(np.array(feat_vals, dtype=float))
    train_feats = np.array(train_feats)

    model = LinearRegression()
    if X_train.size:
        model.fit(X_train, y_train)

    # Predict Effort- for all rows where prediction is possible
    pred_scores = {}
    pred_vals = {}
    for idx, row in data:
        feat_vals = []
        missing = False
        for col in feature_cols:
            f_val = parse_float(row.get(col))
            if f_val is None:
                missing = True
                break
            feat_vals.append(f_val)
        if missing or X_train.size == 0:
            continue
        pred = float(model.predict(np.array([np.array(feat_vals, dtype=float)]))[0])
        score = -pred  # lower Effort- => higher score
        pred_scores[idx] = score
        pred_vals[idx] = pred

    # Find two projects with extreme scores
    if not pred_scores:
        print("No predictions available for NASA‑12.")
        print("ANSWER NASA-12 {}")
        return

    a_idx = max(pred_scores, key=lambda k: pred_scores[k])
    b_idx = min(pred_scores, key=lambda k: pred_scores[k])

    # Ensure they are different
    if a_idx == b_idx:
        # if only one prediction available, duplicate
        a_idx = b_idx
        # choose another by removing a and picking max again
        other_candidates = {k: v for k, v in pred_scores.items() if k != a_idx}
        if other_candidates:
            b_idx = max(other_candidates, key=lambda k: pred_scores[k])

    a_score = pred_scores[a_idx]
    b_score = pred_scores[b_idx]

    # Gather differences
    row_a = dict(data[a_idx-1][1])
    row_b = dict(data[b_idx-1][1])
    differences = []
    # iterate over all columns except X columns
    for col in fieldnames:
        if col.endswith('X'):
            continue
        a_val = row_a.get(col)
        b_val = row_b.get(col)
        if parse_float(a_val) is None and parse_float(b_val) is None:
            if a_val != b_val:
                differences.append({"feature": col, "a": a_val, "b": b_val})
        elif parse_float(a_val) is not None and parse_float(b_val) is not None:
            if parse_float(a_val) != parse_float(b_val):
                differences.append({"feature": col, "a": parse_float(a_val), "b": parse_float(b_val)})
        else:
            if a_val != b_val:
                differences.append({"feature": col, "a": a_val, "b": b_val})
        if len(differences) >= 1:
            break

    # Trade‑offs: goals where non‑preferred (b) is better
    goal_cols = get_goal_columns(fieldnames)
    tradeoffs = []
    for g in goal_cols:
        a_val = parse_float(row_a.get(g))
        b_val = parse_float(row_b.get(g))
        if a_val is None or b_val is None:
            continue
        if g.endswith('+'):
            if b_val > a_val:
                tradeoffs.append(g)
        else:  # '-'
            if b_val < a_val:
                tradeoffs.append(g)

    # Human readable output
    print(f"Project {a_idx} (score {a_score:.2f}) vs Project {b_idx} (score {b_score:.2f})")
    print(f"Preferred project: {a_idx}")
    if differences:
        diff = differences[0]
        print(f"First differing feature: {diff['feature']} -> a:{diff['a']}, b:{diff['b']}")
    if tradeoffs:
        print(f"Trade‑offs (b better on): {', '.join(tradeoffs)}")

    # Machine‑checked answer
    ans = {
        "a": a_idx,
        "b": b_idx,
        "preferred": a_idx,
        "scores": {str(a_idx): round(a_score, 2), str(b_idx): round(b_score, 2)},
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER NASA-12 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    nasa_11(data, fieldnames)
    nasa_12(data, fieldnames)

if __name__ == "__main__":
    main()

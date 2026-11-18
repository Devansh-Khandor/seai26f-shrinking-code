#!/usr/bin/env python3
"""
Skill 06: NASA-11 & NASA-12 explanations

Usage:
    python skill_06.py <path-to-nasa93dem.csv>
"""

import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Import helpers from previous skills
# ------------------------------------------------------------
try:
    from skill_01 import read_csv, format_number
except ImportError:
    # minimal reimplementation
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

try:
    from skill_02 import determine_feature_types, build_encodings, build_feature_matrix
except ImportError:
    # minimal reimplementation
    def determine_feature_types(rows, feature_cols):
        numeric_cols = []
        categorical_cols = []
        for col in feature_cols:
            is_numeric = True
            for r in rows:
                v = r[col]
                if v is None:
                    continue
                if isinstance(v, float):
                    continue
                try:
                    float(v)
                except Exception:
                    is_numeric = False
                    break
            if is_numeric:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        return numeric_cols, categorical_cols

    def build_encodings(rows, categorical_cols):
        encodings = {}
        for col in categorical_cols:
            mapping = {}
            code = 0
            for r in rows:
                v = r[col]
                if v is None:
                    continue
                if v not in mapping:
                    mapping[v] = code
                    code += 1
            encodings[col] = mapping
        return encodings

    def build_feature_matrix(rows, feature_cols, numeric_cols, categorical_cols, encodings):
        X = []
        for r in rows:
            vec = []
            for col in feature_cols:
                v = r[col]
                if v is None:
                    vec.append(0.0)
                    continue
                if col in numeric_cols:
                    vec.append(float(v))
                else:
                    vec.append(encodings[col][v])
            X.append(vec)
        return np.array(X, dtype=float)

# ------------------------------------------------------------
# Helper: predictions
# ------------------------------------------------------------
def predict_effort(rows, feature_cols):
    """Return dict {row_idx: prediction} for held‑out rows."""
    train_rows = [r for idx, r in enumerate(rows, start=1)
                  if idx % 5 != 0 and r['Effort-'] is not None
                  and all(r[c] is not None for c in feature_cols)]
    test_rows = [ (idx, r) for idx, r in enumerate(rows, start=1)
                  if idx % 5 == 0 and r['Effort-'] is not None
                  and all(r[c] is not None for c in feature_cols) ]

    numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
    encodings = build_encodings(train_rows, categorical_cols)

    X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols,
                                   categorical_cols, encodings)
    y_train = np.array([r['Effort-'] for r in train_rows], dtype=float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = {}
    for idx, r in test_rows:
        vec = []
        for col in feature_cols:
            v = r[col]
            if col in numeric_cols:
                vec.append(float(v))
            else:
                vec.append(encodings[col][v])
        X_test = np.array([vec], dtype=float)
        pred = model.predict(X_test)[0]
        predictions[idx] = float(pred)
    return predictions

def predict_defects(rows, feature_cols):
    """Return dict {row_idx: prediction} for held‑out rows."""
    train_rows = [r for idx, r in enumerate(rows, start=1)
                  if idx % 5 != 0 and r['Defects-'] is not None
                  and all(r[c] is not None for c in feature_cols)]
    test_rows = [ (idx, r) for idx, r in enumerate(rows, start=1)
                  if idx % 5 == 0 and r['Defects-'] is not None
                  and all(r[c] is not None for c in feature_cols) ]

    numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
    encodings = build_encodings(train_rows, categorical_cols)

    X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols,
                                   categorical_cols, encodings)
    y_train = np.array([r['Defects-'] for r in train_rows], dtype=float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = {}
    for idx, r in test_rows:
        vec = []
        for col in feature_cols:
            v = r[col]
            if col in numeric_cols:
                vec.append(float(v))
            else:
                vec.append(encodings[col][v])
        X_test = np.array([vec], dtype=float)
        pred = model.predict(X_test)[0]
        predictions[idx] = float(pred)
    return predictions

# ------------------------------------------------------------
# NASA-11: estimate & neighbors
# ------------------------------------------------------------
def nasa_11(rows):
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if c not in ('Effort-', 'Defects-')]

    # get predictions for all held‑out rows
    eff_pred = predict_effort(rows, feature_cols)

    # choose the first held‑out row (smallest index)
    held_indices = sorted([i for i in eff_pred.keys()])
    if not held_indices:
        print("== NASA-11 ==")
        print("No held‑out rows with sufficient data.")
        print("ANSWER NASA-11 {}")
        return
    proj = held_indices[0]
    estimate = eff_pred[proj]

    # prepare training data for similarity search
    train_rows = [ (i, r) for i, r in enumerate(rows, start=1)
                  if i not in eff_pred and all(r[c] is not None for c in feature_cols) ]

    numeric_cols, categorical_cols = determine_feature_types(
        [r for _, r in train_rows], feature_cols)
    encodings = build_encodings([r for _, r in train_rows], categorical_cols)

    # vector for the held‑out row
    rproj = rows[proj-1]
    vec_proj = []
    for col in feature_cols:
        v = rproj[col]
        if col in numeric_cols:
            vec_proj.append(float(v))
        else:
            vec_proj.append(encodings[col][v])

    # compute distances
    neighbors = []
    for idx, r in train_rows:
        vec = []
        for col in feature_cols:
            v = r[col]
            if col in numeric_cols:
                vec.append(float(v))
            else:
                vec.append(encodings[col][v])
        dist = np.linalg.norm(np.array(vec) - np.array(vec_proj))
        neighbors.append((dist, idx, r))

    neighbors.sort(key=lambda t: t[0])
    top_neighbors = neighbors[:3]

    # Build output
    print("== NASA-11 ==")
    print(f"Project row: {proj}")
    print(f"Estimated Effort-: {format_number(estimate)}")
    print("Neighbors:")
    for _, idx, r in top_neighbors:
        vals = {k: (None if v is None else v) for k, v in r.items()}
        print(f"  Row {idx}: {vals}")

    # JSON answer
    neighbors_json = []
    for _, idx, r in top_neighbors:
        vals = {k: (None if v is None else v) for k, v in r.items()}
        neighbors_json.append({"row": idx, "values": vals})
    answer = {
        "project": proj,
        "estimate": estimate,
        "neighbors": neighbors_json
    }
    print(f"ANSWER NASA-11 {json.dumps(answer, separators=(',', ':'))}")

# ------------------------------------------------------------
# NASA-12: risk comparison
# ------------------------------------------------------------
def nasa_12(rows):
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if c not in ('Effort-', 'Defects-')]

    eff_pred = predict_effort(rows, feature_cols)
    def_pred = predict_defects(rows, feature_cols)

    # choose two held‑out rows with different risk (Effort prediction)
    if not eff_pred:
        print("== NASA-12 ==")
        print("No held‑out rows with sufficient data.")
        print("ANSWER NASA-12 {}")
        return

    sorted_eff = sorted(eff_pred.items(), key=lambda t: t[1])
    low_row, low_score = sorted_eff[0]
    high_row, high_score = sorted_eff[-1]

    # determine preferred: lower risk
    if low_score <= high_score:
        preferred = low_row
        a_row, b_row = low_row, high_row
        a_score, b_score = low_score, high_score
    else:
        preferred = high_row
        a_row, b_row = high_row, low_row
        a_score, b_score = high_score, low_score

    # differences
    diffs = []
    for f in feature_cols:
        va = rows[a_row-1][f]
        vb = rows[b_row-1][f]
        if va is None or vb is None:
            continue
        # numeric or categorical
        if isinstance(va, float) and isinstance(vb, float):
            diff = abs(va - vb)
        else:
            diff = 0 if va == vb else 1
        diffs.append((diff, f, va, vb))

    diffs.sort(key=lambda t: t[0], reverse=True)
    top_diffs = diffs[:3]
    differences = [{"feature": f, "a": va, "b": vb} for _, f, va, vb in top_diffs]

    # tradeoffs: goal columns where non-preferred has better value
    goal_cols = [c for c in all_cols if (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]
    tradeoffs = []
    for g in goal_cols:
        va = rows[a_row-1].get(g)
        vb = rows[b_row-1].get(g)
        if va is None or vb is None:
            continue
        better = False
        if g.endswith('+'):  # maximize
            better = va > vb
        elif g.endswith('-'):  # minimize
            better = va < vb
        if better:
            tradeoffs.append(g)

    # Print section
    print("== NASA-12 ==")
    print(f"Row a: {a_row}")
    print(f"Row b: {b_row}")
    print(f"Preferred: {preferred}")
    print(f"Score a: {format_number(a_score)}")
    print(f"Score b: {format_number(b_score)}")
    print("Differences:")
    for d in differences:
        print(f"  {d['feature']}: a={d['a']} b={d['b']}")
    print("Tradeoffs:")
    for t in tradeoffs:
        print(f"  {t}")

    # JSON answer
    answer = {
        "a": a_row,
        "b": b_row,
        "preferred": preferred,
        "scores": {"a": a_score, "b": b_score},
        "differences": differences,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER NASA-12 {json.dumps(answer, separators=(',', ':'))}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_11(rows)
    nasa_12(rows)

if __name__ == "__main__":
    main()

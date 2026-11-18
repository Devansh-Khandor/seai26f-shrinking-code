#!/usr/bin/env python3
"""
skill_06.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-11 – When the system estimates the effort required for a proposed project,
explain the estimate using relevant historical projects and their recorded characteristics.
NASA-12 – When two project proposals receive different risk assessments,
explain the main differences that contributed to those assessments and identify the important tradeoffs.

The script prints the requested sections and a machine‑checked JSON answer for each request.
"""

import sys
import csv
import json
import math
from statistics import mean, median
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_value(v):
    """Parse a cell value; return float or None for missing."""
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def get_feature_columns(all_columns, exclude=None):
    """Return column names to be used as features."""
    exclude = set(exclude or [])
    features = [c for c in all_columns if not c.endswith('X')
                and c not in exclude and c not in ('idX', 'YearX', 'MonthsX')]
    return features

def train_model(rows, feature_cols, target_col):
    """
    Train linear regression on rows with non‑None target.
    Return the fitted pipeline and min/max stats for the target column.
    """
    X, y = [], []
    for r in rows:
        if r[target_col] is None:
            continue
        x = [r[c] for c in feature_cols]
        X.append(x)
        y.append(r[target_col])

    if not X:
        return None, (None, None)

    X = np.array(X)
    y = np.array(y)
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        LinearRegression()
    )
    pipeline.fit(X, y)

    # compute training target stats
    y_pred = pipeline.predict(X)
    resid_std = np.sqrt(np.mean((y - y_pred) ** 2))
    return pipeline, (min(y), max(y), resid_std)

def predict_row(model, r, feature_cols):
    """Predict target for a single row."""
    x = np.array([r[c] for c in feature_cols]).reshape(1, -1)
    return float(model.predict(x)[0])

def normalize(v, vmin, vmax, best_first=True):
    """Normalize a value to [0,1]. If best_first is True, 1 is best; else 0 is best."""
    if vmax == vmin or vmin is None or vmax is None:
        return 0.5
    if best_first:
        return (v - vmin) / (vmax - vmin)
    else:
        return (vmax - v) / (vmax - vmin)

def load_csv(csv_path):
    """Read CSV and return list of rows (dict) with parsed numeric values."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                parsed[col] = parse_value(raw[col]) if not col.endswith('X') else None
            parsed['row_num'] = idx
            rows.append(parsed)
    return rows, all_columns

# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #

def nasa_11(rows, all_columns):
    """
    Estimate effort for a held‑out project (first divisible by 5).
    Explain the estimate using the 3 nearest historical projects.
    """
    # Separate held‑out and history
    held_out = [r for r in rows if r['row_num'] % 5 == 0]
    history = [r for r in rows if r['row_num'] % 5 != 0]
    if not held_out or not history:
        return None

    target_row = held_out[0]
    # Train model on history for Effort-
    feature_cols = get_feature_columns(all_columns, exclude=('Effort-', 'Defects-'))
    effort_model, effort_stats = train_model(history, feature_cols, 'Effort-')
    if effort_model is None:
        estimate = None
    else:
        estimate = predict_row(effort_model, target_row, feature_cols)

    # Find nearest neighbors (based on Euclidean distance on features)
    feature_names = [c for c in all_columns if not c.endswith('X')
                     and not c.endswith('+') and not c.endswith('-')
                     and c not in ('idX', 'YearX', 'MonthsX')]
    neighbors = []
    for h in history:
        # compute distance over overlapping features
        sq = 0.0
        count = 0
        for f in feature_names:
            v1 = target_row.get(f)
            v2 = h.get(f)
            if v1 is not None and v2 is not None:
                sq += (v1 - v2) ** 2
                count += 1
        if count == 0:
            continue
        dist = math.sqrt(sq / count)
        neighbors.append((dist, h))

    neighbors.sort(key=lambda x: x[0])
    top3 = neighbors[:3]
    neighbor_info = []
    for _, nbr in top3:
        neighbor_info.append({
            "row": nbr['row_num'],
            "values": {k: v for k, v in nbr.items() if k != 'row_num'}
        })

    return {
        "project": target_row['row_num'],
        "estimate": estimate,
        "neighbors": neighbor_info
    }

def nasa_12(rows, all_columns):
    """
    Compare two held‑out proposals with different risk assessments.
    Explain differences and tradeoffs.
    """
    # Separate held‑out and history
    held_out = [r for r in rows if r['row_num'] % 5 == 0]
    history = [r for r in rows if r['row_num'] % 5 != 0]
    if not held_out or not history:
        return None

    # Train models for Effort- and Defects-
    feature_excl_effort = get_feature_columns(all_columns, exclude=('Effort-', 'Defects-'))
    feature_excl_defects = get_feature_columns(all_columns, exclude=('Defects-', 'Effort-'))

    eff_model, eff_stats = train_model(history, feature_excl_effort, 'Effort-')
    def_model, def_stats = train_model(history, feature_excl_defects, 'Defects-')

    # Compute normalized scores and risk
    eff_min, eff_max, _ = eff_stats
    def_min, def_max, _ = def_stats

    def normalized(v, mn, mx):
        return normalize(v, mn, mx, best_first=True)

    risk_scores = {}
    for r in held_out:
        if r['Effort-'] is None or r['Defects-'] is None:
            continue
        eff_norm = normalized(r['Effort-'], eff_min, eff_max)
        def_norm = normalized(r['Defects-'], def_min, def_max)
        risk_scores[r['row_num']] = eff_norm + def_norm

    if not risk_scores:
        return None

    # Pick highest and lowest risk rows
    a_row_num = max(risk_scores, key=risk_scores.get)
    b_row_num = min(risk_scores, key=risk_scores.get)
    a_score = risk_scores[a_row_num]
    b_score = risk_scores[b_row_num]

    a_row = next(r for r in held_out if r['row_num'] == a_row_num)
    b_row = next(r for r in held_out if r['row_num'] == b_row_num)

    preferred = a_row_num if a_score > b_score else b_row_num
    # Differences: all columns except goals and X
    diff_list = []
    for col in all_columns:
        if col.endswith('X') or col.endswith('+') or col.endswith('-'):
            continue
        val_a = a_row.get(col)
        val_b = b_row.get(col)
        if val_a != val_b:
            diff_list.append({
                "feature": col,
                "a": val_a,
                "b": val_b
            })
    if not diff_list:
        # Force at least one difference if data identical (unlikely)
        diff_list.append({"feature": "none", "a": None, "b": None})

    # Tradeoffs: goal columns where non-preferred row is better
    tradeoffs = []
    for col in all_columns:
        if col.endswith('+') or col.endswith('-'):
            val_a = a_row.get(col)
            val_b = b_row.get(col)
            if val_a is None or val_b is None:
                continue
            # For '-' (minimize): lower is better
            if col.endswith('-'):
                if val_b < val_a:
                    tradeoffs.append(col)
            else:
                if val_b > val_a:
                    tradeoffs.append(col)

    return {
        "a": a_row_num,
        "b": b_row_num,
        "preferred": preferred,
        "scores": {"a": a_score, "b": b_score},
        "differences": diff_list,
        "tradeoffs": tradeoffs
    }

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, all_columns = load_csv(csv_path)

    # NASA-11
    print("== NASA-11 ==")
    result_11 = nasa_11(rows, all_columns)
    if result_11:
        proj = result_11['project']
        est = result_11['estimate']
        neighs = [n['row'] for n in result_11['neighbors']]
        print(f"Selected held‑out project: {proj}")
        print(f"Estimated Effort-: {est:.2f}" if est is not None else "Estimated Effort-: None")
        print(f"Top 3 similar historical projects: {neighs}")
    else:
        print("No held‑out or historical data available.")
    print("ANSWER NASA-11", json.dumps(result_11, ensure_ascii=False))

    # NASA-12
    print("\n== NASA-12 ==")
    result_12 = nasa_12(rows, all_columns)
    if result_12:
        a = result_12['a']
        b = result_12['b']
        pref = result_12['preferred']
        print(f"Comparing projects {a} vs {b}")
        print(f"Risk scores: a={result_12['scores']['a']:.3f}, b={result_12['scores']['b']:.3f}")
        print(f"Preferred: {pref}")
        # Show at least one difference
        diff = result_12['differences'][0]
        print(f"Difference example: {diff['feature']} a={diff['a']} b={diff['b']}")
        print(f"Trade‑offs where non‑preferred is better: {result_12['tradeoffs']}")
    else:
        print("Insufficient data for risk comparison.")
    print("ANSWER NASA-12", json.dumps(result_12, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_04.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-07 – As new project proposals arrive, assess each proposal using its available
characteristics and alert me when its predicted effort or defects appear unusually high.
NASA-08 – As additional completed-project records become available,
update our historical summaries and project estimates so that future assessments reflect
the latest information.

The script prints the requested sections and a machine‑checked JSON answer
for each request.
"""

import sys
import csv
import json
import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from statistics import mean, median
from math import isnan, floor

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

def compute_stats(vals):
    """Return mean, min, max for a list of numeric values (or None if empty)."""
    if not vals:
        return None, None, None
    return mean(vals), min(vals), max(vals)

def get_goal_columns(all_columns):
    """Return list of column names that are goals (end with '+' or '-')."""
    return [c for c in all_columns if (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

def get_feature_columns(all_columns, exclude=None):
    """Return column names to be used as features."""
    exclude = set(exclude or [])
    features = [c for c in all_columns if not c.endswith('X')
                and c not in exclude and c not in ('idX', 'YearX', 'MonthsX')]
    return features

def train_model(rows, feature_cols, target_col):
    """Train linear regression on rows with non‑None target; return model and training target stats."""
    X, y = [], []
    for r in rows:
        if r[target_col] is None:
            continue
        x = [r[c] for c in feature_cols]
        X.append(x)
        y.append(r[target_col])
    if not X:
        return None, (None, None)  # no data
    X = np.array(X)
    y = np.array(y)
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        LinearRegression()
    )
    pipeline.fit(X, y)
    # compute training target mean and std
    y_pred = pipeline.predict(X)
    residuals = y - y_pred
    resid_std = np.sqrt(np.mean(residuals ** 2))
    return pipeline, (np.mean(y), resid_std)

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

def compute_shortlist(rows, goals, top_n=5):
    """Return list of row numbers that are best according to normalized scores."""
    # compute min/max per goal
    mins, maxs = {}, {}
    for col, direction in goals:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = None

    scored = []
    for r in rows:
        # skip rows missing any goal value
        if any(r[col] is None for col, _ in goals):
            continue
        score = 0.0
        for col, direction in goals:
            v = r[col]
            best_first = (direction == '+')
            norm = normalize(v, mins[col], maxs[col], best_first=best_first)
            score += norm
        scored.append((r['row_num'], score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [rn for rn, _ in scored[:top_n]]

def compute_summary(rows, goal_cols):
    """Return dict mapping goal name to stats: n, mean, min, max."""
    summary = {}
    for col in goal_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        n = len(vals)
        if n == 0:
            summary[col] = {"n": 0, "mean": None, "min": None, "max": None}
        else:
            m, mi, ma = compute_stats(vals)
            summary[col] = {"n": n, "mean": m, "min": mi, "max": ma}
    return summary

# --------------------------------------------------------------------------- #
# Main analysis functions
# --------------------------------------------------------------------------- #

def nasa_07(rows, all_columns, arrival_indices, budget):
    """
    For each arrival row, train models on all prior rows (history + earlier arrivals),
    predict Effort- and Defects-, and alert when prediction is unusually high.
    Returns list of assessment dicts.
    """
    goal_cols = get_goal_columns(all_columns)
    # Map target to other goal for feature exclusion
    feature_exclusions = {'Effort-': ('Effort-', 'Defects-'),
                          'Defects-': ('Defects-', 'Effort-')}

    assessments = []
    for idx, row_num in enumerate(arrival_indices[:budget]):
        # training set: all rows before this arrival
        train_rows = [r for r in rows if r['row_num'] < row_num]
        arrival_row = next(r for r in rows if r['row_num'] == row_num)

        preds = {}
        alerts = []

        for target in ('Effort-', 'Defects-'):
            target_dir = '+' if target.endswith('+') else '-'
            excl = feature_exclusions.get(target, ())
            feature_cols = get_feature_columns(all_columns, exclude=excl)
            model, (t_mean, t_std) = train_model(train_rows, feature_cols, target)
            if model is None:
                pred_val = None
            else:
                pred_val = predict_row(model, arrival_row, feature_cols)
                # alert threshold: mean + 2*std
                if t_std is None or t_std == 0:
                    threshold = t_mean
                else:
                    threshold = t_mean + 2 * t_std
                alerts.append(pred_val > threshold)
            preds[target] = pred_val

        # if either alert True, set alert True
        alert_flag = any(alerts)
        assessments.append({
            "row": row_num,
            "predicted_effort": preds['Effort-'],
            "predicted_defects": preds['Defects-'],
            "alert": alert_flag
        })

        # print per-row output
        print(f"Row {row_num} predicted Effort- {preds['Effort-']:.2f} "
              f"Defects- {preds['Defects-']:.2f} alert: {alert_flag}")

    # summary
    alert_count = sum(1 for a in assessments if a['alert'])
    print(f"Total arrivals assessed: {len(assessments)}")
    print(f"Number of alerts: {alert_count}")

    return assessments

def nasa_08(rows, all_columns, arrival_indices):
    """
    Provide checkpoints after m = 0, 9, 19 arrivals.
    Each checkpoint includes summary of goal columns and 5 best rows seen so far.
    """
    goal_cols = get_goal_columns(all_columns)
    checkpoints = []
    total_arrivals = len(arrival_indices)
    for m in [0, 9, 19]:
        if m > total_arrivals:
            continue
        seen_rows = rows[:len(rows)-total_arrivals + m]
        summary = compute_summary(seen_rows, goal_cols)
        # compute shortlist
        goals = [(c, '+' if c.endswith('+') else '-') for c in goal_cols]
        shortlist = compute_shortlist(seen_rows, goals, top_n=5)
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })
        # print checkpoint
        print(f"\nCheckpoint after {m} arrivals:")
        for col, stats in summary.items():
            print(f"  {col}: n={stats['n']}, mean={stats['mean']:.2f}, "
                  f"min={stats['min']:.2f}, max={stats['max']:.2f}")
        print(f"  Top 5 rows: {shortlist}")

    return checkpoints

def load_rows(csv_path):
    """Read CSV and return list of parsed rows with row numbers."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                if col.endswith('X'):  # non-input
                    parsed[col] = None
                else:
                    parsed[col] = parse_value(raw[col])
            parsed['row_num'] = idx
            rows.append(parsed)
    return rows, all_columns

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, all_columns = load_rows(csv_path)
    total_rows = len(rows)

    # Determine arrivals: last round(0.2 * n) rows
    arrivals_count = round(0.2 * total_rows)
    arrival_indices = [r['row_num'] for r in rows[-arrivals_count:]]

    # Budget: at most min(30, number_of_rows // 5) rows may be evaluated
    budget = min(30, total_rows // 5)

    # NASA-07
    print("== NASA-07 ==")
    assessments = nasa_07(rows, all_columns, arrival_indices, budget)
    print("ANSWER NASA-07", json.dumps({"assessments": assessments}, ensure_ascii=False))

    # NASA-08
    print("\n== NASA-08 ==")
    checkpoints = nasa_08(rows, all_columns, arrival_indices)
    print("ANSWER NASA-08", json.dumps({"checkpoints": checkpoints}, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_08.py

Implements NASA-15 and NASA-16 analyses on the NASA software project dataset.

NASA-15: For a project proposal predicted to require excessive effort,
         identify plausible changes to adjustable planning or process characteristics
         that might reduce its predicted effort.

NASA-16: For a project proposal with a high predicted defect count,
         identify the smallest plausible combination of changes to adjustable
         planning or process characteristics that could improve its predicted
         quality, while explaining potential trade‑offs.

Usage:
    python skill_08.py <csv_path>
"""

import sys
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """Read CSV and return rows (list of dicts) and list of numeric column names."""
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        numeric_cols = []
        for col in reader.fieldnames:
            try:
                for r in rows:
                    val = r[col]
                    if val != '':
                        float(val.replace(',', ''))
                        numeric_cols.append(col)
                        break
            except (ValueError, TypeError):
                continue
    return rows, numeric_cols


def to_numpy(rows, cols):
    """Convert selected columns of rows to a numpy array, NaN for missing."""
    arr = []
    for r in rows:
        row = []
        for c in cols:
            val = r[c]
            if val == '':
                row.append(np.nan)
            else:
                try:
                    row.append(float(val.replace(',', '')))
                except ValueError:
                    row.append(np.nan)
        arr.append(row)
    return np.array(arr, dtype=float)


def train_models(rows, numeric_cols):
    """
    Train linear regression models for Effort- and Defects- using all numeric
    features except the two targets.
    """
    # Identify completed projects
    completed = [r for r in rows if r['Effort-'] != '' and r['Defects-'] != '']
    if not completed:
        return None, None, None, None

    feature_cols = [c for c in numeric_cols if c not in ('Effort-', 'Defects-')]
    X = to_numpy(completed, feature_cols)
    y_eff = to_numpy(completed, ['Effort-']).flatten()
    y_def = to_numpy(completed, ['Defects-']).flatten()

    model_eff = LinearRegression(fit_intercept=True).fit(X, y_eff)
    model_def = LinearRegression(fit_intercept=True).fit(X, y_def)

    return model_eff, model_def, X, feature_cols


def predict(row, model, feature_cols):
    """Predict target for a single proposal row."""
    vec = np.array([float(row[c]) if row[c] != '' else np.nan for c in feature_cols], dtype=float)
    if np.isnan(vec).any():
        return None
    return model.predict(vec.reshape(1, -1))[0]


def find_proposals(rows, numeric_cols):
    """Return proposals (rows with missing Effort- or Defects-) that have full numeric features."""
    proposals = []
    for r in rows:
        # Must have at least one missing target
        if r['Effort-'] == '' or r['Defects-'] == '':
            # Ensure all numeric features are present
            missing = any(r[c] == '' for c in numeric_cols if c not in ('Effort-', 'Defects-'))
            if not missing:
                proposals.append(r)
    return proposals


def adjust_feature(vec, idx, pct):
    """Return a copy of vec with feature at idx decreased by pct (as a fraction)."""
    new = np.array(vec, copy=True)
    if new[idx] > 0:
        new[idx] *= (1.0 - pct)
    return new


def best_adjustments_for_effort(proposal, vec, model_eff, model_def, feature_cols,
                               eff_pred, def_pred, adjustable_indices):
    """
    Explore reducing each adjustable feature by 10% and record effort reduction.
    Return top 3 adjustments sorted by greatest effort decrease.
    """
    results = []
    for idx in adjustable_indices:
        new_vec = adjust_feature(vec, idx, 0.10)
        new_eff = model_eff.predict(new_vec.reshape(1, -1))[0]
        eff_delta = eff_pred - new_eff
        if eff_delta > 0:
            results.append((eff_delta, feature_cols[idx], new_eff))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:3]


def best_adjustments_for_defects(proposal, vec, model_eff, model_def, feature_cols,
                                 eff_pred, def_pred, adjustable_indices):
    """
    Explore reducing each adjustable feature by 10% and record defect reduction.
    Only keep adjustments that reduce defects and increase effort by <5% relative
    to the original predicted effort.
    Return top 3 adjustments sorted by greatest defect decrease.
    """
    results = []
    for idx in adjustable_indices:
        new_vec = adjust_feature(vec, idx, 0.10)
        new_def = model_def.predict(new_vec.reshape(1, -1))[0]
        def_delta = def_pred - new_def
        new_eff = model_eff.predict(new_vec.reshape(1, -1))[0]
        eff_increase = (new_eff - eff_pred) / eff_pred if eff_pred > 0 else 0
        if def_delta > 0 and eff_increase <= 0.05:
            results.append((def_delta, feature_cols[idx], new_def, new_eff))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:3]


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # Train models
    model_eff, model_def, X_all, feature_cols = train_models(rows, numeric_cols)
    if model_eff is None:
        print("Insufficient completed data to train models.")
        sys.exit(0)

    # Compute medians for thresholding
    median_eff = np.nanmedian(to_numpy(rows, ['Effort-']).flatten())
    median_def = np.nanmedian(to_numpy(rows, ['Defects-']).flatten())
    eff_threshold = median_eff * 1.5
    def_threshold = median_def * 1.5

    # Identify adjustable features
    adjustable_candidates = ['time', 'pvol', 'acap', 'pcap', 'pcon',
                            'apex', 'plex', 'ltex', 'tool', 'sced']
    adjustable_features = [c for c in adjustable_candidates if c in feature_cols]
    adjustable_indices = [feature_cols.index(c) for c in adjustable_features]

    proposals = find_proposals(rows, numeric_cols)

    # NASA-15: Excessive effort
    print("== NASA-15 ==")
    found_effort = False
    for r in proposals:
        vec = np.array([float(r[c]) if r[c] != '' else np.nan for c in feature_cols], dtype=float)
        if np.isnan(vec).any():
            continue
        eff_pred = model_eff.predict(vec.reshape(1, -1))[0]
        def_pred = model_def.predict(vec.reshape(1, -1))[0]
        if eff_pred > eff_threshold:
            found_effort = True
            print(f"Proposal ID {r['idX']}: Predicted effort {eff_pred:.2f} person‑months")
            print(f"  Suggested adjustments (10% reduction):")
            best_eff = best_adjustments_for_effort(r, vec, model_eff, model_def,
                                                  feature_cols, eff_pred, def_pred,
                                                  adjustable_indices)
            if best_eff:
                for delta, feat, new_eff in best_eff:
                    print(f"    Reduce {feat} by 10% → effort {new_eff:.2f} (Δ={delta:.2f})")
            else:
                print("    No single‑feature adjustment yields effort reduction.")
            print()
    if not found_effort:
        print("No proposals exceed the excessive effort threshold.")
        print()

    # NASA-16: High defect count
    print("== NASA-16 ==")
    found_defect = False
    for r in proposals:
        vec = np.array([float(r[c]) if r[c] != '' else np.nan for c in feature_cols], dtype=float)
        if np.isnan(vec).any():
            continue
        eff_pred = model_eff.predict(vec.reshape(1, -1))[0]
        def_pred = model_def.predict(vec.reshape(1, -1))[0]
        if def_pred > def_threshold:
            found_defect = True
            print(f"Proposal ID {r['idX']}: Predicted defects {def_pred:.2f}")
            print(f"  Suggested adjustments (10% reduction):")
            best_def = best_adjustments_for_defects(r, vec, model_eff, model_def,
                                                    feature_cols, eff_pred, def_pred,
                                                    adjustable_indices)
            if best_def:
                for delta, feat, new_def, new_eff in best_def:
                    eff_inc_pct = (new_eff - eff_pred) / eff_pred if eff_pred > 0 else 0
                    print(f"    Reduce {feat} by 10% → defects {new_def:.2f} (Δ={delta:.2f}), "
                          f"effort ↑{eff_inc_pct*100:.1f}%")
            else:
                print("    No feasible adjustment reduces defects without >5% effort increase.")
            print()
    if not found_defect:
        print("No proposals exceed the high defect threshold.")
        print()


if __name__ == "__main__":
    main()

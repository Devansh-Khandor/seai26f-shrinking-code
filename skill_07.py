#!/usr/bin/env python3
"""
skill_07.py

Implements NASA-13 and NASA-14 analyses on the NASA software project dataset.
NASA-13: Identify promising combinations of project characteristics that historically delivered more software
         with lower development effort and fewer defects.
NASA-14: When detailed project evaluations are expensive, recommend which project proposals should be evaluated first
         to discover promising alternatives with a limited evaluation budget.

Usage:
    python skill_07.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import Counter

# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path):
    """Read CSV and return rows (dict) and list of numeric column names."""
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
    """Return (n_samples, n_features) array of floats, NaN for missing."""
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

def train_linear(X, y):
    """Train and return a LinearRegression model."""
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    return model

def predict_with_model(model, X):
    """Return predictions for X."""
    return model.predict(X)

def median_of(col):
    """Return median ignoring NaNs."""
    return np.nanmedian(col)

# ---------------------------------------------------------------------------
# NASA-13 helpers

def identify_promising_projects(rows, numeric_cols):
    """
    Return indices of projects that are in the top 25% by size (Kloc+)
    but in the bottom 25% by Effort- and Defects-.
    """
    size_arr = np.array([float(r['Kloc+']) if r['Kloc+'] != '' else np.nan for r in rows])
    eff_arr  = np.array([float(r['Effort-']) if r['Effort-'] != '' else np.nan for r in rows])
    def_arr  = np.array([float(r['Defects-']) if r['Defects-'] != '' else np.nan for r in rows])

    size_thr = np.nanpercentile(size_arr, 75)
    eff_thr  = np.nanpercentile(eff_arr, 25)
    def_thr  = np.nanpercentile(def_arr, 25)

    promising = []
    for i, (s, e, d) in enumerate(zip(size_arr, eff_arr, def_arr)):
        if np.isnan(s) or np.isnan(e) or np.isnan(d):
            continue
        if s >= size_thr and e <= eff_thr and d <= def_thr:
            promising.append(i)
    return promising

def get_feature_key(row, feature_list):
    """Return a tuple of feature values for the given row."""
    return tuple(row[f] for f in feature_list)

def analyze_combinations(rows, promising_indices, feature_list, top_n=3):
    """
    Count combinations of the specified categorical features among promising projects.
    Return top_n most frequent combinations with their stats.
    """
    counter = Counter()
    stats = {}
    for idx in promising_indices:
        row = rows[idx]
        key = get_feature_key(row, feature_list)
        counter[key] += 1

    for key, cnt in counter.most_common(top_n):
        # compute mean effort and defects for this combination
        effort_vals = []
        defect_vals = []
        for idx in promising_indices:
            row = rows[idx]
            if get_feature_key(row, feature_list) == key:
                effort_vals.append(float(row['Effort-']))
                defect_vals.append(float(row['Defects-']))
        stats[key] = (np.mean(effort_vals), np.mean(defect_vals), cnt)
    return stats

# ---------------------------------------------------------------------------
# NASA-14 helpers

def build_models(rows, numeric_cols):
    """Return trained models for Effort-, Defects- and Kloc+."""
    # Identify completed projects
    completed = [r for r in rows if r['Effort-'] != '' and r['Defects-'] != '']
    if not completed:
        return None, None, None

    feature_cols = [c for c in numeric_cols if c not in ('Effort-', 'Defects-', 'Kloc+')]

    X = to_numpy(completed, feature_cols)
    y_eff = to_numpy(completed, ['Effort-']).flatten()
    y_def = to_numpy(completed, ['Defects-']).flatten()
    y_loc = to_numpy(completed, ['Kloc+']).flatten()

    model_eff = train_linear(X, y_eff)
    model_def = train_linear(X, y_def)
    model_kloc = train_linear(X, y_loc)

    return model_eff, model_def, model_kloc

def find_proposals(rows, numeric_cols):
    """Return proposals (rows with missing Effort- or Defects-) that have full numeric features."""
    feature_cols = [c for c in numeric_cols if c not in ('Effort-', 'Defects-', 'Kloc+')]
    proposals = []
    for r in rows:
        if r['Effort-'] == '' or r['Defects-'] == '':
            missing = False
            for c in feature_cols:
                if r[c] == '':
                    missing = True
                    break
            if not missing:
                proposals.append(r)
    return proposals

def score_proposal(row, model_eff, model_def, model_kloc, feature_cols,
                   med_eff, med_def, med_kloc):
    """Return a numeric score indicating how promising the proposal is."""
    X = np.array([[float(row[c]) for c in feature_cols]], dtype=float)

    pred_eff = predict_with_model(model_eff, X)[0]
    pred_def = predict_with_model(model_def, X)[0]
    pred_kloc = predict_with_model(model_kloc, X)[0]

    # Normalized values
    norm_eff = pred_eff / med_eff if med_eff != 0 else 0
    norm_def = pred_def / med_def if med_def != 0 else 0
    norm_kloc = pred_kloc / med_kloc if med_kloc != 0 else 0

    # Higher size, lower effort & defects => higher score
    score = norm_kloc - 0.5 * (norm_eff + norm_def)
    return score

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, numeric_cols = parse_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # ---------- NASA-13 ----------
    print("== NASA-13 ==")
    promising_idx = identify_promising_projects(rows, numeric_cols)
    if not promising_idx:
        print("No projects meet the promising criteria.")
    else:
        # Select categorical features to form combinations
        categorical = ['centerX', 'team', 'pcap', 'cplx', 'tool']
        # Ensure they exist
        categorical = [c for c in categorical if c in rows[0]]
        combo_stats = analyze_combinations(rows, promising_idx, categorical, top_n=3)

        for i, (combo, (eff, defs, cnt)) in enumerate(combo_stats.items(), 1):
            combo_str = ", ".join(f"{c}={v}" for c, v in zip(categorical, combo))
            print(f"{i}. {combo_str} – {cnt} projects")
            print(f"   Avg Effort: {eff:.2f} person‑months, Avg Defects: {defs:.2f}")
    print()

    # ---------- NASA-14 ----------
    print("== NASA-14 ==")
    model_eff, model_def, model_kloc = build_models(rows, numeric_cols)
    if model_eff is None:
        print("Insufficient completed data to train models.")
        sys.exit(0)

    # Compute medians for normalization
    completed = [r for r in rows if r['Effort-'] != '' and r['Defects-'] != '']
    X_all = to_numpy(completed, [c for c in numeric_cols if c not in ('Effort-', 'Defects-', 'Kloc+')])
    med_eff = np.nanmedian(to_numpy(completed, ['Effort-']).flatten())
    med_def = np.nanmedian(to_numpy(completed, ['Defects-']).flatten())
    med_kloc = np.nanmedian(to_numpy(completed, ['Kloc+']).flatten())

    proposals = find_proposals(rows, numeric_cols)
    if not proposals:
        print("No proposals available for evaluation.")
        sys.exit(0)

    feature_cols = [c for c in numeric_cols if c not in ('Effort-', 'Defects-', 'Kloc+')]
    scored = []
    for p in proposals:
        try:
            s = score_proposal(p, model_eff, model_def, model_kloc, feature_cols,
                               med_eff, med_def, med_kloc)
            scored.append((s, p))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)

    # Recommend top 3 proposals to evaluate first
    if not scored:
        print("Unable to score proposals.")
    else:
        print("Top proposals to evaluate first (promising combinations):")
        for rank, (s, p) in enumerate(scored[:3], 1):
            print(f"{rank}. ID {p['idX']} – Score {s:.3f}")
    print()

if __name__ == "__main__":
    main()

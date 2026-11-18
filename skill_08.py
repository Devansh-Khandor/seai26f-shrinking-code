#!/usr/bin/env python3
"""
skill_08.py <csv path>

Implements:
    NASA-15: For a project proposal predicted to require excessive effort, identify
             plausible changes to adjustable planning or process characteristics
             that might reduce its predicted effort.

    NASA-16: For a project proposal with a high predicted defect count, identify
             the smallest plausible combination of changes to adjustable
             planning or process characteristics that could improve its predicted
             quality, while explaining potential tradeoffs.

The script prints the header for each request, at least three human‑readable
result lines, and then the machine‑checked JSON answer.
"""

import csv
import json
import math
import sys
from statistics import mean

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def to_float(v: str):
    """Convert string to float or None if missing."""
    return None if v == '?' else float(v)

def is_numeric_col(col: str, rows):
    """Return True if all non‑missing values in col can be parsed as float."""
    for r in rows:
        val = r.get(col, '?')
        if val == '?':
            continue
        try:
            float(val)
        except ValueError:
            return False
    return True

def build_mappings(rows, feature_cols):
    """Return dict mapping each categorical feature to value→int."""
    mappings = {}
    for col in feature_cols:
        # Skip numeric columns
        if is_numeric_col(col, rows):
            continue
        mapping = {}
        idx = 0
        for r in rows:
            val = r[col]
            if val == '?':
                continue
            if val not in mapping:
                mapping[val] = idx
                idx += 1
        if mapping:
            mappings[col] = mapping
    return mappings

def encode_row(row, feature_cols, mappings):
    """Return list of encoded numeric values for the row."""
    out = []
    for col in feature_cols:
        val = row[col]
        if val == '?':
            out.append(None)
            continue
        if col in mappings:
            out.append(mappings[col][val])
        else:
            out.append(float(val))
    return out

def compute_means(encoded_rows):
    """Return list of means per column over encoded_rows (ignore None)."""
    n_cols = len(encoded_rows[0])
    sums = [0.0] * n_cols
    counts = [0] * n_cols
    for r in encoded_rows:
        for i, v in enumerate(r):
            if v is not None:
                sums[i] += v
                counts[i] += 1
    means = [s / c if c else 0.0 for s, c in zip(sums, counts)]
    return means

def replace_missing(encoded_rows, means):
    """Return new matrix with None replaced by column mean."""
    new = []
    for r in encoded_rows:
        new.append([v if v is not None else m for v, m in zip(r, means)])
    return new

def normal_equation(X, y):
    """Solve (X^T X) beta = X^T y for beta using Gaussian elimination."""
    n_rows = len(X)
    n_feats = len(X[0])
    # Build augmented matrix
    M = [row[:] + [y[i]] for i, row in enumerate(X)]
    for i in range(n_feats):
        # Pivot
        max_row = max(range(i, n_rows), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Singular matrix")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i, n_feats + 1):
            M[i][j] /= pivot
        for k in range(n_rows):
            if k == i:
                continue
            factor = M[k][i]
            for j in range(i, n_feats + 1):
                M[k][j] -= factor * M[i][j]
    return [M[i][n_feats] for i in range(n_feats)]

def predict(coeffs, vec):
    """Return dot product of coeffs and vec (vec includes intercept)."""
    return sum(c * v for c, v in zip(coeffs, vec))

# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------
def main(csv_path):
    # Load all rows
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        all_rows = [row for row in reader]

    n_rows = len(all_rows)

    # Identify goal and non‑goal columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    feature_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    # Adjustable columns: all feature columns (Kloc+ is goal, so excluded)
    adjustable_cols = [c for c in feature_cols if c != 'Kloc+']

    # Build categorical mappings
    cat_mappings = build_mappings(all_rows, feature_cols)

    # Encode all rows
    encoded_all = [encode_row(r, feature_cols, cat_mappings) for r in all_rows]

    # Means for imputation
    feature_means = compute_means(encoded_all)

    # Split held‑out rows (divisible by 5)
    held_out_indices = [i for i in range(n_rows) if (i + 1) % 5 == 0]
    non_held_indices = [i for i in range(n_rows) if (i + 1) % 5 != 0]

    # Prepare training data for Effort and Defects
    def prepare_training(target):
        X = []
        y = []
        for i in non_held_indices:
            row = all_rows[i]
            val = to_float(row[target])
            if val is None:
                continue
            vec = encoded_all[i][:]
            # Replace missing
            vec = [v if v is not None else feature_means[j] for j, v in enumerate(vec)]
            vec = [1.0] + vec  # intercept
            X.append(vec)
            y.append(val)
        return X, y

    effort_X, effort_y = prepare_training('Effort-')
    defect_X, defect_y = prepare_training('Defects-')

    # Train linear models
    effort_coeffs = normal_equation(effort_X, effort_y) if effort_X else [0.0]
    defect_coeffs = normal_equation(defect_X, defect_y) if defect_X else [0.0]

    # Predict for held‑out rows
    predictions = {}
    for idx in held_out_indices:
        vec = encoded_all[idx][:]
        vec = [v if v is not None else feature_means[j] for j, v in enumerate(vec)]
        vec = [1.0] + vec
        eff_pred = predict(effort_coeffs, vec)
        def_pred = predict(defect_coeffs, vec)
        predictions[idx + 1] = {
            'effort_before': eff_pred,
            'defects_before': def_pred,
            'vec': vec
        }

    # ------------------------------------------------------------------
    # NASA‑15: Excessive effort
    # ------------------------------------------------------------------
    # Find held‑out row with maximum predicted Effort
    max_eff_row_no = max(predictions, key=lambda r: predictions[r]['effort_before'])
    max_eff_info = predictions[max_eff_row_no]
    effort_before = max_eff_info['effort_before']
    vec = max_eff_info['vec']

    # For each adjustable column, compute best change
    best_change = None
    best_effort_after = effort_before

    # Pre‑compute min/max values for numeric columns
    numeric_values = {}
    for col in adjustable_cols:
        if is_numeric_col(col, all_rows):
            vals = [to_float(r[col]) for r in all_rows if r[col] != '?']
            numeric_values[col] = (min(vals), max(vals))

    # Pre‑compute mean Effort per category for categorical columns (using training set)
    cat_mean_effort = {}
    for col in adjustable_cols:
        if col not in cat_mappings:
            continue
        # Count Effort per category using training rows
        cat_vals = {}
        for idx in non_held_indices:
            r = all_rows[idx]
            val = to_float(r['Effort-'])
            if val is None:
                continue
            cat_val = r[col]
            if cat_val == '?':
                continue
            cat_vals.setdefault(cat_val, []).append(val)
        if cat_vals:
            cat_mean_effort[col] = {k: mean(vs) for k, vs in cat_vals.items()}

    for col in adjustable_cols:
        # Determine current value
        cur_raw = all_rows[max_eff_row_no - 1][col]
        cur_val = cur_raw if cur_raw != '?' else None

        # Determine favorable new value
        if col in cat_mappings:
            # Categorical
            if col not in cat_mean_effort:
                continue
            # Category with lowest mean Effort
            fav_cat = min(cat_mean_effort[col].items(), key=lambda kv: kv[1])[0]
            if cur_val == fav_cat:
                continue  # no change
            new_val = fav_cat
        else:
            # Numeric
            mn, mx = numeric_values[col]
            coeff_index = feature_cols.index(col)
            coeff = effort_coeffs[1 + coeff_index]  # intercept is at 0
            if coeff > 0:
                new_val = mn
            elif coeff < 0:
                new_val = mx
            else:
                continue
            if cur_val is not None and float(cur_val) == new_val:
                continue  # already optimal
        # Build modified vector
        new_vec = vec[:]
        # Encode new value
        if col in cat_mappings:
            encoded_new = cat_mappings[col][new_val]
        else:
            encoded_new = float(new_val)
        # Replace in vector (index offset by 1 due to intercept)
        new_vec[1 + feature_cols.index(col)] = encoded_new
        effort_after = predict(effort_coeffs, new_vec)
        if effort_after < best_effort_after:
            best_effort_after = effort_after
            best_change = {
                col: [cur_val, new_val]
            }

    if best_change is None:
        # Fallback: pick first column that actually reduces effort
        best_change = {}

    # ------------------------------------------------------------------
    # NASA‑16: High defect count
    # ------------------------------------------------------------------
    max_def_row_no = max(predictions, key=lambda r: predictions[r]['defects_before'])
    max_def_info = predictions[max_def_row_no]
    defect_before = max_def_info['defects_before']
    effort_before_def = max_def_info['effort_before']
    vec_def = max_def_info['vec']

    # Pre‑compute mean defects per category for categorical columns
    cat_mean_defect = {}
    for col in adjustable_cols:
        if col not in cat_mappings:
            continue
        cat_vals = {}
        for idx in non_held_indices:
            r = all_rows[idx]
            val = to_float(r['Defects-'])
            if val is None:
                continue
            cat_val = r[col]
            if cat_val == '?':
                continue
            cat_vals.setdefault(cat_val, []).append(val)
        if cat_vals:
            cat_mean_defect[col] = {k: mean(vs) for k, vs in cat_vals.items()}

    # Evaluate each candidate change
    change_candidates = []
    for col in adjustable_cols:
        cur_raw = all_rows[max_def_row_no - 1][col]
        cur_val = cur_raw if cur_raw != '?' else None

        if col in cat_mappings:
            # Categorical
            if col not in cat_mean_defect:
                continue
            fav_cat = min(cat_mean_defect[col].items(), key=lambda kv: kv[1])[0]
            if cur_val == fav_cat:
                continue
            new_val = fav_cat
        else:
            # Numeric
            mn, mx = numeric_values[col]
            coeff_index = feature_cols.index(col)
            coeff = defect_coeffs[1 + coeff_index]
            if coeff > 0:
                new_val = mn
            elif coeff < 0:
                new_val = mx
            else:
                continue
            if cur_val is not None and float(cur_val) == new_val:
                continue
        # Build modified vector
        new_vec = vec_def[:]
        if col in cat_mappings:
            encoded_new = cat_mappings[col][new_val]
        else:
            encoded_new = float(new_val)
        new_vec[1 + feature_cols.index(col)] = encoded_new

        defect_after = predict(defect_coeffs, new_vec)
        effort_after = predict(effort_coeffs, new_vec)
        defect_improve = defect_before - defect_after
        effort_cost = effort_after - effort_before_def
        change_candidates.append((defect_improve, effort_cost, col, cur_val, new_val, defect_after, effort_after))

    # Choose up to 3 changes with positive defect improvement
    change_candidates.sort(key=lambda x: (-x[0], x[1]))  # max improvement, then least effort cost
    chosen_changes = []
    for cand in change_candidates:
        if cand[0] <= 0:
            continue
        chosen_changes.append({
            'col': cand[2],
            'cur': cand[3],
            'new': cand[4],
            'defect_after': cand[5],
            'effort_after': cand[6]
        })
        if len(chosen_changes) >= 3:
            break

    # Compute final after predictions if changes applied
    final_vec = vec_def[:]
    for ch in chosen_changes:
        col = ch['col']
        if col in cat_mappings:
            encoded_new = cat_mappings[col][ch['new']]
        else:
            encoded_new = float(ch['new'])
        final_vec[1 + feature_cols.index(col)] = encoded_new
    defect_after_final = predict(defect_coeffs, final_vec)
    effort_after_final = predict(effort_coeffs, final_vec)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA‑15
    print("== NASA-15 ==")
    print(f"Row {max_eff_row_no}: predicted Effort- before = {effort_before:.2f}, after change = {best_effort_after:.2f}")
    if best_change:
        for col, pair in best_change.items():
            cur, new = pair
            print(f"Suggested change: {col} from {cur} to {new}")
    else:
        print("No beneficial change identified.")
    print(f"ANSWER NASA-15 {json.dumps({'project': max_eff_row_no, 'changes': best_change, 'predicted_effort': {'before': round(effort_before, 2), 'after': round(best_effort_after, 2)}})}")

    # NASA‑16
    print("\n== NASA-16 ==")
    print(f"Row {max_def_row_no}: predicted Defects- before = {defect_before:.2f}, after changes = {defect_after_final:.2f}")
    print(f"Predicted Effort- before = {effort_before_def:.2f}, after changes = {effort_after_final:.2f}")
    if chosen_changes:
        for ch in chosen_changes:
            print(f"Change: {ch['col']} from {ch['cur']} to {ch['new']}")
    else:
        print("No beneficial changes identified.")
    changes_dict = {ch['col']: [ch['cur'], ch['new']] for ch in chosen_changes}
    print(f"ANSWER NASA-16 {json.dumps({'project': max_def_row_no, 'changes': changes_dict, 'predicted_defects': {'before': round(defect_before, 2), 'after': round(defect_after_final, 2)}, 'predicted_effort': {'before': round(effort_before_def, 2), 'after': round(effort_after_final, 2)}})}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

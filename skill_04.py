#!/usr/bin/env python3
"""
skill_04.py <csv path>

Implements:
    NASA-07: As new project proposals arrive, assess each proposal using its
             available characteristics and alert the manager when its
             predicted effort or defects appear unusually high.

    NASA-08: As additional completed-project records become available,
             update our historical summaries and project estimates so that
             future assessments reflect the latest information.

The script prints a header for each request, at least three human‑readable
lines, and a machine‑checked JSON answer.

The JSON answers are exactly in the format required by the test harness.
"""

import csv
import json
import math
import sys

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def to_float(v):
    """Convert a string to float; return None if missing."""
    return None if v == '?' else float(v)

def build_categorical_mappings(rows, columns):
    """Return mapping dict: column -> value -> int."""
    mappings = {}
    for col in columns:
        # Determine if column is numeric
        numeric = False
        for r in rows:
            val = r[col]
            if val == '?':
                continue
            try:
                float(val)
                numeric = True
                break
            except ValueError:
                pass
        if numeric:
            continue
        # Build mapping
        mapping = {}
        idx = 0
        for r in rows:
            val = r[col]
            if val == '?':
                continue
            if val not in mapping:
                mapping[val] = idx
                idx += 1
        mappings[col] = mapping
    return mappings

def encode_row(row, columns, mappings):
    """Encode one row as list of floats, using mappings for categorical."""
    out = []
    for col in columns:
        val = row[col]
        if val == '?':
            out.append(None)
            continue
        if col in mappings:
            out.append(mappings[col][val])
        else:
            out.append(float(val))
    return out

def compute_means(rows, columns):
    """Compute mean for each column over rows; None treated as missing."""
    sums = [0.0] * len(columns)
    counts = [0] * len(columns)
    for r in rows:
        for i, v in enumerate(r):
            if v is not None:
                sums[i] += v
                counts[i] += 1
    return [s / c if c else 0.0 for s, c in zip(sums, counts)]

def replace_missing_with_mean(rows, means):
    """Return new list of rows with missing values replaced by column means."""
    new = []
    for r in rows:
        new.append([v if v is not None else m for v, m in zip(r, means)])
    return new

def solve_linear_system(A, b):
    """Solve A x = b with Gaussian elimination (partial pivot)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Singular matrix")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot
        for k in range(n):
            if k == i:
                continue
            factor = M[k][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]

def normal_equation(X, y):
    """Solve (X^T X) beta = X^T y."""
    n_rows = len(X)
    n_feats = len(X[0])
    XtX = [[0.0] * n_feats for _ in range(n_feats)]
    XtY = [0.0] * n_feats
    for i in range(n_rows):
        xi = X[i]
        yi = y[i]
        for a in range(n_feats):
            XtY[a] += xi[a] * yi
            for b in range(n_feats):
                XtX[a][b] += xi[a] * xi[b]
    return solve_linear_system(XtX, XtY)

# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------
def main(csv_path):
    # Load all rows
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        all_rows_dicts = list(reader)

    total_rows = len(all_rows_dicts)

    # Determine goal columns (+ or -) and feature columns (exclude X and goals)
    goal_columns = [c for c in header if c.endswith('+') or c.endswith('-')]
    feature_columns = [c for c in header
                       if not c.endswith('X') and c not in goal_columns]

    # Determine arrivals: last round(0.2 * N) rows
    arrivals_count = round(0.2 * total_rows)
    if arrivals_count > total_rows:
        arrivals_count = total_rows
    history_rows_dicts = all_rows_dicts[:total_rows - arrivals_count]
    arrivals_rows_dicts = all_rows_dicts[total_rows - arrivals_count:]

    # Build categorical mappings using all rows
    cat_mappings = build_categorical_mappings(all_rows_dicts, feature_columns)

    # Encode all rows
    encoded_all = [encode_row(r, feature_columns, cat_mappings) for r in all_rows_dicts]

    # Means for imputation
    feature_means = compute_means(encoded_all, feature_columns)

    # Helper: encode a single row dict
    def encode_dict(r):
        return encode_row(r, feature_columns, cat_mappings)

    # ------------------------------------------------------------------
    # NASA-07
    # ------------------------------------------------------------------
    assessments = []

    for idx, arrival_row_dict in enumerate(arrivals_rows_dicts):
        # Training set: history + earlier arrivals
        training_dicts = history_rows_dicts + arrivals_rows_dicts[:idx]
        training_encoded = [encode_dict(r) for r in training_dicts]
        # Impute missing
        training_filled = replace_missing_with_mean(training_encoded, feature_means)
        # Build X and y for Effort-
        effort_y = []
        effort_X = []
        for r_dict, r_enc in zip(training_dicts, training_filled):
            val = to_float(r_dict.get('Effort-', '?'))
            if val is None:
                continue
            effort_y.append(val)
            effort_X.append([1.0] + r_enc)
        if effort_X:
            effort_coef = normal_equation(effort_X, effort_y)
        else:
            effort_coef = [0.0]
        # Build X and y for Defects-
        defect_y = []
        defect_X = []
        for r_dict, r_enc in zip(training_dicts, training_filled):
            val = to_float(r_dict.get('Defects-', '?'))
            if val is None:
                continue
            defect_y.append(val)
            defect_X.append([1.0] + r_enc)
        if defect_X:
            defect_coef = normal_equation(defect_X, defect_y)
        else:
            defect_coef = [0.0]

        # Predict on arrival
        arrival_enc = encode_dict(arrival_row_dict)
        arrival_filled = [v if v is not None else feature_means[i]
                          for i, v in enumerate(arrival_enc)]
        arrival_vec = [1.0] + arrival_filled
        pred_effort = sum(c * vx for c, vx in zip(effort_coef, arrival_vec))
        pred_defect = sum(c * vx for c, vx in zip(defect_coef, arrival_vec))

        # Compute predictions for training rows
        pred_effort_train = []
        pred_defect_train = []
        for r_enc in training_filled:
            vec = [1.0] + r_enc
            pred_effort_train.append(sum(c * vx for c, vx in zip(effort_coef, vec)))
            pred_defect_train.append(sum(c * vx for c, vx in zip(defect_coef, vec)))
        def mean_std(values):
            n = len(values)
            if n == 0:
                return 0.0, 0.0
            m = sum(values) / n
            var = sum((x - m) ** 2 for x in values) / n
            return m, math.sqrt(var)
        mean_eff, std_eff = mean_std(pred_effort_train)
        mean_def, std_def = mean_std(pred_defect_train)
        alert = (pred_effort > mean_eff + 2 * std_eff) or \
                (pred_defect > mean_def + 2 * std_def)
        # Row number is 1‑based in original data
        row_no = total_rows - arrivals_count + idx + 1
        assessments.append({
            "row": row_no,
            "predicted_effort": pred_effort,
            "predicted_defects": pred_defect,
            "alert": alert
        })

    # ------------------------------------------------------------------
    # NASA-08
    # ------------------------------------------------------------------
    checkpoints = []
    for m in [0, 9, 19]:
        # Subset rows up to m arrivals
        subset_dicts = history_rows_dicts + arrivals_rows_dicts[:m]
        subset_encoded = [encode_dict(r) for r in subset_dicts]
        # Goal stats
        summary = {}
        for g in goal_columns:
            vals = []
            for r_dict in subset_dicts:
                val = to_float(r_dict.get(g, '?'))
                if val is not None:
                    vals.append(val)
            if vals:
                n = len(vals)
                mean_val = sum(vals) / n
                min_val = min(vals)
                max_val = max(vals)
                summary[g] = {"n": n, "mean": mean_val, "min": min_val, "max": max_val}
            else:
                summary[g] = {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        # Shortlist: top 5 rows by joint score
        # Compute per‑goal min/max for scoring
        goal_minmax = {}
        for g in goal_columns:
            vals = [v for v in [to_float(r.get(g, '?')) for r in subset_dicts] if v is not None]
            if vals:
                goal_minmax[g] = (min(vals), max(vals))
            else:
                goal_minmax[g] = (0.0, 0.0)
        scores = []
        for idx, r_dict in enumerate(subset_dicts, start=1):
            # check all goal values present
            if any(to_float(r_dict.get(g, '?')) is None for g in goal_columns):
                continue
            norm_total = 0.0
            for g in goal_columns:
                v = to_float(r_dict.get(g))
                mn, mx = goal_minmax[g]
                if mx == mn:
                    ns = 1.0
                else:
                    if g.endswith('+'):
                        ns = (v - mn) / (mx - mn)
                    else:
                        ns = (mx - v) / (mx - mn)
                norm_total += ns
            scores.append(( -norm_total, idx))  # negative for descending
        scores.sort()
        shortlist = [idx for _, idx in scores[:5]]
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-07
    print("== NASA-07 ==")
    # Human‑readable lines
    print(f"Number of new proposals assessed: {len(assessments)}")
    print("Example assessments (first 3):")
    for a in assessments[:3]:
        print(f"  Row {a['row']}: Effort={a['predicted_effort']:.2f}, Defects={a['predicted_defects']:.2f}, alert={a['alert']}")
    print(f"ANSWER NASA-07 {json.dumps({'assessments': assessments})}")

    # NASA-08
    print("\n== NASA-08 ==")
    print(f"Number of checkpoints: {len(checkpoints)}")
    print("Example checkpoints (first 2):")
    for cp in checkpoints[:2]:
        print(f"  After {cp['after']} arrivals: shortlist={cp['shortlist']}")
    print(f"ANSWER NASA-08 {json.dumps({'checkpoints': checkpoints})}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

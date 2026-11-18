#!/usr/bin/env python3
"""
skill_10.py <csv path>

Implements:
    NASA-19: Compare our project‑effort estimation system against a simple
             historical‑average estimate and show whether it makes meaningfully
             better predictions on previously unseen projects.

    NASA-20: Compare our project‑selection approach with randomly selecting
             proposals under the same evaluation budget, and show whether our
             system identifies more desirable project outcomes.

The script prints, for each request, a header line,
human‑readable diagnostic lines, and a machine‑checked JSON answer.
"""

import csv
import json
import math
import random
import sys

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def is_numeric(v: str) -> bool:
    """Return True if v can be parsed as float."""
    if v == '?' or v is None:
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False


def to_float(v: str):
    """Convert string to float; return None for '?'."""
    return None if v == '?' else float(v)


def build_categorical_mappings(rows, columns):
    """Return mapping dict: column -> value -> int."""
    mappings = {}
    for col in columns:
        # Skip numeric columns
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


def encode_row(row, columns, mappings):
    """Return list of numeric values for a row, None for missing."""
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


def compute_means(encoded_rows):
    """Compute mean per column over all rows (ignore None)."""
    n_cols = len(encoded_rows[0])
    sums = [0.0] * n_cols
    counts = [0] * n_cols
    for row in encoded_rows:
        for i, v in enumerate(row):
            if v is not None:
                sums[i] += v
                counts[i] += 1
    means = [s / c if c else 0.0 for s, c in zip(sums, counts)]
    return means


def replace_missing_with_mean(rows, means):
    """Replace None with column mean."""
    new = []
    for row in rows:
        new.append([v if v is not None else m for v, m in zip(row, means)])
    return new


def normal_equation(X, y):
    """Solve (X^T X) beta = X^T y via Gaussian elimination."""
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
    """Dot product of coeffs and feature vector (incl. intercept)."""
    return sum(c * v for c, v in zip(coeffs, vec))


# ------------------------------------------------------------------
# Joint score helpers for NASA-20
# ------------------------------------------------------------------
def compute_joint_score(row, goal_cols, goal_minmax):
    """Return joint score for a row; None if any goal missing."""
    score = 0.0
    for g in goal_cols:
        val = to_float(row[g])
        if val is None:
            return None
        mn, mx = goal_minmax[g]
        if mx == mn:
            norm = 1.0
        else:
            if g.endswith('+'):
                norm = (val - mn) / (mx - mn)
            else:
                norm = (mx - val) / (mx - mn)
        score += norm
    return score


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    n_rows = len(rows)
    all_row_nums = list(range(1, n_rows + 1))

    # Identify goal columns and feature columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    feature_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    # ---------- NASA-19 ----------
    # Held‑out rows: numbers divisible by 5
    held_out = [idx for idx in all_row_nums if idx % 5 == 0]
    training = [idx for idx in all_row_nums if idx % 5 != 0]

    # Build categorical mappings from training rows
    train_rows = [rows[i - 1] for i in training]
    cat_mappings = build_categorical_mappings(train_rows, feature_cols)

    # Encode training rows
    encoded_train = [encode_row(r, feature_cols, cat_mappings) for r in train_rows]

    # Target y: Effort- of training rows (skip missing)
    y = []
    X = []
    for r, enc in zip(train_rows, encoded_train):
        val = to_float(r['Effort-'])
        if val is None:
            continue
        y.append(val)
        X.append(enc)

    # Impute missing with column means
    if X:
        col_means = compute_means(X)
        X_filled = replace_missing_with_mean(X, col_means)
    else:
        X_filled = []

    # Add intercept
    X_filled = [[1.0] + row for row in X_filled]

    # Train linear regression
    if X_filled:
        coeffs = normal_equation(X_filled, y)
    else:
        coeffs = [0.0]

    # Baseline: mean Effort- over training rows
    baseline_mean = sum(y) / len(y) if y else 0.0

    # Predict on held‑out rows
    preds = {}
    mae_model = 0.0
    mae_base = 0.0
    count = 0

    for idx in held_out:
        row = rows[idx - 1]
        enc = encode_row(row, feature_cols, cat_mappings)
        # impute
        imputed = [v if v is not None else col_means[i] for i, v in enumerate(enc)]
        vec = [1.0] + imputed
        pred = predict(coeffs, vec)
        preds[str(idx)] = round(pred, 5)

        actual = to_float(row['Effort-'])
        if actual is not None:
            mae_model += abs(pred - actual)
            mae_base += abs(baseline_mean - actual)
            count += 1

    mae_model /= count if count else 0.0
    mae_base /= count if count else 0.0

    if mae_model < mae_base - 1e-6:
        verdict19 = "better"
    elif mae_model > mae_base + 1e-6:
        verdict19 = "worse"
    else:
        verdict19 = "no clear difference"

    # ---------- NASA-20 ----------
    # Compute joint scores for all rows (only rows with all goals)
    # First compute min/max per goal across all rows that have the goal
    goal_minmax = {}
    for g in goal_cols:
        vals = [to_float(r[g]) for r in rows if to_float(r[g]) is not None]
        if vals:
            goal_minmax[g] = (min(vals), max(vals))
        else:
            goal_minmax[g] = (0.0, 0.0)

    scores = {}
    for idx, r in enumerate(rows, start=1):
        score = compute_joint_score(r, goal_cols, goal_minmax)
        if score is not None:
            scores[idx] = score

    # Order of evaluation (our method): rows sorted by score descending
    eval_order = sorted(scores.keys(), key=lambda i: -scores[i])

    budget = min(30, n_rows // 5)
    repeats = 20

    ours_best_rows = []
    random_best_rows = []
    ours_best_scores = []
    random_best_scores = []

    random.seed(0)

    for _ in range(repeats):
        # Ours: first budget rows of eval_order
        eval_rows = eval_order[:budget]
        # Find best score among eval_rows
        best_ours = None
        best_score_ours = -math.inf
        for i in eval_rows:
            s = scores[i]
            if s > best_score_ours:
                best_score_ours = s
                best_ours = i
        ours_best_rows.append(best_ours)
        ours_best_scores.append(best_score_ours)

        # Random: sample budget rows
        rand_rows = random.sample(all_row_nums, budget)
        best_rand = None
        best_score_rand = -math.inf
        for i in rand_rows:
            if i not in scores:
                continue
            s = scores[i]
            if s > best_score_rand:
                best_score_rand = s
                best_rand = i
        random_best_rows.append(best_rand)
        random_best_scores.append(best_score_rand)

    # Verdict for NASA-20
    better_count = sum(1 for a, b in zip(ours_best_scores, random_best_scores) if a > b)
    worse_count = sum(1 for a, b in zip(ours_best_scores, random_best_scores) if a < b)

    if better_count > worse_count:
        verdict20 = "ours better"
    elif worse_count > better_count:
        verdict20 = "random better"
    else:
        verdict20 = "no clear difference"

    # ---------- Output ----------
    # NASA-19
    print("== NASA-19 ==")
    print("Predicted Effort- for held‑out rows:")
    for rn in sorted(preds, key=int):
        print(f"  Row {rn}: {preds[rn]:.5f}")
    print(f"Baseline average Effort-: {baseline_mean:.5f}")
    print(f"MAE of model: {mae_model:.5f}")
    print(f"MAE of baseline: {mae_base:.5f}")
    print(f"ANSWER NASA-19 {json.dumps({'predictions': preds, 'baseline': round(baseline_mean,5), 'mae_model': round(mae_model,5), 'mae_baseline': round(mae_base,5), 'verdict': verdict19})}")

    # NASA-20
    print("\n== NASA-20 ==")
    print(f"Evaluation budget: {budget}")
    print("Best row (ours) each repeat:")
    for i, r in enumerate(ours_best_rows, start=1):
        print(f"  Repeat {i}: row {r} score {ours_best_scores[i-1]:.5f}")
    print("Best row (random) each repeat:")
    for i, r in enumerate(random_best_rows, start=1):
        print(f"  Repeat {i}: row {r} score {random_best_scores[i-1]:.5f}")
    print(f"ANSWER NASA-20 {json.dumps({'budget': budget, 'repeats': repeats, 'ours': ours_best_rows, 'random': random_best_rows, 'verdict': verdict20})}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

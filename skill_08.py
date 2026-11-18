#!/usr/bin/env python3
"""
Skill 08 – NASA-15 & NASA-16
"""
import csv
import json
import sys
from typing import List, Dict, Any, Tuple

# Re‑use goal parsing helper
from skill_01 import parse_value

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output

# ----------------------------------------------------------------------
# Helper: load rows with string values for non‑goal columns
# ----------------------------------------------------------------------
def load_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for col, val in row.items():
                if col in GOALS:
                    parsed[col] = parse_value(val)
                else:
                    parsed[col] = val
            rows.append(parsed)
    return rows


# ----------------------------------------------------------------------
# Determine adjustable (changeable) columns
# ----------------------------------------------------------------------
def adjustable_columns(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    sample = rows[0]
    cols = []
    for col in sample.keys():
        if col in GOALS:
            continue
        if col.endswith("X"):
            continue
        if col == "Kloc+":
            continue
        cols.append(col)
    return cols


# ----------------------------------------------------------------------
# Multivariate linear regression
# ----------------------------------------------------------------------
def train_multilinear_regression(rows: List[Dict[str, Any]],
                                 target_col: str,
                                 feature_cols: List[str]) -> List[float]:
    """Return weights [intercept, w1, w2, ...]"""
    X = []
    Y = []

    for r in rows:
        # target must exist
        target = r.get(target_col)
        if target is None:
            continue
        # all features must exist
        feat_vals = []
        ok = True
        for col in feature_cols:
            val_raw = r.get(col, "")
            val = parse_value(val_raw)
            if val is None:
                ok = False
                break
            feat_vals.append(val)
        if not ok:
            continue
        X.append([1.0] + feat_vals)
        Y.append(target)

    p = len(feature_cols) + 1
    if not X:
        return [0.0] * p

    # Build XtX and XtY
    XtX = [[0.0] * p for _ in range(p)]
    XtY = [0.0] * p
    for i in range(len(X)):
        xi = X[i]
        yi = Y[i]
        for a in range(p):
            for b in range(p):
                XtX[a][b] += xi[a] * xi[b]
            XtY[a] += xi[a] * yi

    # Solve linear system XtX * w = XtY
    w = solve_linear_system(XtX, XtY)
    return w


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b via Gaussian elimination. Assumes full rank."""
    n = len(b)
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Forward elimination
    for i in range(n):
        # Pivot selection
        piv_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv_row] = M[piv_row], M[i]
        piv = M[i][i]
        if abs(piv) < 1e-12:
            # Singular, return zeros
            return [0.0] * n
        # Normalize pivot row
        for k in range(i, n + 1):
            M[i][k] /= piv
        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = M[r][i]
            for k in range(i, n + 1):
                M[r][k] -= factor * M[i][k]
    # Solution
    return [M[i][n] for i in range(n)]


def predict(row: Dict[str, Any], weights: List[float], feature_cols: List[str]) -> float:
    intercept = weights[0]
    total = intercept
    for idx, col in enumerate(feature_cols):
        val_raw = row.get(col, "")
        val = parse_value(val_raw)
        if val is None:
            # missing value shouldn't happen in predictions
            continue
        total += weights[idx + 1] * val
    return total


# ----------------------------------------------------------------------
# Find best changes to reduce effort
# ----------------------------------------------------------------------
def find_effort_changes(rows: List[Dict[str, Any]],
                        feature_cols: List[str],
                        weights: List[float],
                        target_row: Dict[str, Any],
                        target_idx: int) -> Tuple[Dict[str, Tuple[Any, Any]], float, float]:
    """Return (changes_dict, before_pred, after_pred)."""
    # Current predicted effort
    before = predict(target_row, weights, feature_cols)

    # Collect unique raw values per column
    unique_vals: Dict[str, List[str]] = {}
    for col in feature_cols:
        unique_vals[col] = []

    for r in rows:
        for col in feature_cols:
            val = r.get(col, "")
            if val not in ("", "?"):
                unique_vals[col].append(val)

    # For each column find the value that gives lowest predicted effort
    improvements = {}
    for col in feature_cols:
        best_val = None
        best_pred = None
        for val in set(unique_vals[col]):
            if val == target_row.get(col, ""):
                continue  # no change
            # create a copy of target_row with changed value
            new_row = target_row.copy()
            new_row[col] = val
            pred = predict(new_row, weights, feature_cols)
            if best_pred is None or pred < best_pred:
                best_pred = pred
                best_val = val
        if best_val is not None and best_pred < before - 1e-6:
            improvements[col] = (best_val, best_pred)

    # Pick up to 3 columns with largest improvement
    best_changes = {}
    after = before
    for col, (new_val, new_pred) in sorted(improvements.items(),
                                            key=lambda x: x[1][1] - before):
        best_changes[col] = (target_row.get(col, ""), new_val)
        after = new_pred
        if len(best_changes) >= 3:
            break

    return best_changes, before, after


# ----------------------------------------------------------------------
# Find best changes to reduce defects
# ----------------------------------------------------------------------
def find_defects_changes(rows: List[Dict[str, Any]],
                         feature_cols: List[str],
                         effort_weights: List[float],
                         defect_weights: List[float],
                         target_row: Dict[str, Any],
                         target_idx: int) -> Tuple[Dict[str, Tuple[Any, Any]],
                                                    float, float, float, float]:
    """Return (changes_dict, before_defects, after_defects, before_effort, after_effort)."""
    before_defects = predict(target_row, defect_weights, feature_cols)
    before_effort = predict(target_row, effort_weights, feature_cols)

    unique_vals: Dict[str, List[str]] = {}
    for col in feature_cols:
        unique_vals[col] = []

    for r in rows:
        for col in feature_cols:
            val = r.get(col, "")
            if val not in ("", "?"):
                unique_vals[col].append(val)

    improvements = {}
    for col in feature_cols:
        best_val = None
        best_pred = None
        for val in set(unique_vals[col]):
            if val == target_row.get(col, ""):
                continue
            new_row = target_row.copy()
            new_row[col] = val
            new_def = predict(new_row, defect_weights, feature_cols)
            if best_pred is None or new_def < best_pred:
                best_pred = new_def
                best_val = val
        if best_val is not None and best_pred < before_defects - 1e-6:
            improvements[col] = (best_val, best_pred)

    # Pick the single column with largest defects reduction
    if improvements:
        col, (new_val, new_pred) = max(improvements.items(),
                                       key=lambda x: before_defects - x[1][1])
        changes = {col: (target_row.get(col, ""), new_val)}
        after_defects = new_pred
        after_effort = predict({**target_row, col: new_val}, effort_weights, feature_cols)
    else:
        changes = {}
        after_defects = before_defects
        after_effort = before_effort

    return changes, before_defects, after_defects, before_effort, after_effort


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_rows(csv_path)
    if not rows:
        print("No data.")
        return

    feature_cols = adjustable_columns(rows)

    # Train regression models once
    effort_weights = train_multilinear_regression(rows, "Effort-", feature_cols)
    defect_weights = train_multilinear_regression(rows, "Defects-", feature_cols)

    # Predict effort for all rows to find the worst
    predicted_efforts = [predict(r, effort_weights, feature_cols) for r in rows]
    worst_effort_idx = max(range(len(rows)), key=lambda i: predicted_efforts[i] if predicted_efforts[i] is not None else float("-inf"))
    worst_effort_row = rows[worst_effort_idx]

    # NASA-15
    changes_effort, before_eff, after_eff = find_effort_changes(
        rows, feature_cols, effort_weights, worst_effort_row, worst_effort_idx
    )

    print("== NASA-15 ==")
    print(f"Project row {worst_effort_idx + 1} predicted Effort- = {before_eff:.3f}")
    if changes_effort:
        for col, (cur, new) in changes_effort.items():
            print(f"Change {col}: {cur} → {new}")
        print(f"After changes predicted Effort- = {after_eff:.3f}")
    else:
        print("No improvement found.")
    answer_15 = {
        "project": worst_effort_idx + 1,
        "changes": {k: [v[0], v[1]] for k, v in changes_effort.items()},
        "predicted_effort": {"before": before_eff, "after": after_eff},
    }
    print("ANSWER NASA-15 " + json.dumps(answer_15, separators=(",", ":")))

    # Find the worst defect prediction
    predicted_defects = [predict(r, defect_weights, feature_cols) for r in rows]
    worst_defect_idx = max(range(len(rows)), key=lambda i: predicted_defects[i] if predicted_defects[i] is not None else float("-inf"))
    worst_defect_row = rows[worst_defect_idx]

    changes_defects, before_def, after_def, before_eff2, after_eff2 = find_defects_changes(
        rows, feature_cols, effort_weights, defect_weights, worst_defect_row, worst_defect_idx
    )

    print("\n== NASA-16 ==")
    print(f"Project row {worst_defect_idx + 1} predicted Defects- = {before_def:.3f}")
    if changes_defects:
        for col, (cur, new) in changes_defects.items():
            print(f"Change {col}: {cur} → {new}")
        print(f"After changes predicted Defects- = {after_def:.3f}")
        print(f"Effort trade‑off: before {before_eff2:.3f} -> after {after_eff2:.3f}")
    else:
        print("No improvement found.")
    answer_16 = {
        "project": worst_defect_idx + 1,
        "changes": {k: [v[0], v[1]] for k, v in changes_defects.items()},
        "predicted_defects": {"before": before_def, "after": after_def},
        "predicted_effort": {"before": before_eff2, "after": after_eff2},
    }
    print("ANSWER NASA-16 " + json.dumps(answer_16, separators=(",", ":")))


if __name__ == "__main__":
    main()

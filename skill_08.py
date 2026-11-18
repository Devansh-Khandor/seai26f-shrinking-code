#!/usr/bin/env python3
"""
skill_08.py

NASA-15: For a project proposal predicted to require excessive effort,
identify plausible changes to adjustable planning or process characteristics that might reduce its predicted effort.

NASA-16: For a project proposal with a high predicted defect count,
identify the smallest plausible combination of changes to adjustable planning or process characteristics
that could improve its predicted quality, while explaining potential trade‑offs.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import json
import sys
import math
import itertools
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    return rows, header

def to_float(val: str) -> float:
    """Convert string to float; return NaN if missing or not convertible."""
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")

def is_numeric(val: str) -> bool:
    """True if val can be parsed as float and is not '?'."""
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False

# ------------------------------------------------------------------
# Machine‑learning helpers (from skill_02 / skill_04)
# ------------------------------------------------------------------
def prepare_feature_dict(row: Dict[str, str], feature_names: List[str]) -> Dict[str, Any]:
    """Convert a row dict to a feature dict suitable for the pipeline."""
    d: Dict[str, Any] = {}
    for col in feature_names:
        val = row[col]
        if val.strip() == "?":
            d[col] = None  # will be handled by imputer / encoder
        else:
            if is_numeric(val):
                d[col] = float(val)
            else:
                d[col] = val
    return d

def build_pipeline(feature_rows: List[Dict[str, Any]], feature_names: List[str]):
    """Create and return a scikit‑learn pipeline for regression."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    # Determine numeric vs categorical columns
    numeric_cols: Set[str] = set()
    categorical_cols: Set[str] = set()

    for col in feature_names:
        found_num = False
        for row in feature_rows:
            val = row.get(col, None)
            if val is None:
                continue
            if isinstance(val, float) or isinstance(val, int):
                found_num = True
                break
            if isinstance(val, str) and val.strip() != "?":
                try:
                    float(val)
                    found_num = True
                    break
                except ValueError:
                    pass
        if found_num:
            numeric_cols.add(col)
        else:
            categorical_cols.add(col)

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ]
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                          ("model", LinearRegression())])
    return pipe

# ------------------------------------------------------------------
# Core analysis helpers
# ------------------------------------------------------------------
def get_adjustable_columns(header: List[str]) -> List[str]:
    """Return columns that can be changed: non‑goal, non‑X, not Kloc+."""
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    x_cols = {c for c in header if c.endswith("X")}
    return [c for c in header if c not in goal_cols and c not in x_cols and c != "Kloc+"]

def compute_predictions(rows: List[Dict[str, str]],
                        feature_names: List[str],
                        effort_pipe,
                        defects_pipe) -> Tuple[List[float], List[float]]:
    """Return lists of predicted Effort- and Defects- for each row."""
    eff_preds = []
    def_preds = []
    for r in rows:
        feat = prepare_feature_dict(r, feature_names)
        try:
            eff = effort_pipe.predict([feat])[0]
            eff_preds.append(eff)
        except Exception:
            eff_preds.append(float("nan"))
        try:
            dft = defects_pipe.predict([feat])[0]
            def_preds.append(dft)
        except Exception:
            def_preds.append(float("nan"))
    return eff_preds, def_preds

def round_val(v: float) -> float:
    return round(v, 4)

def find_max_row(preds: List[float]) -> int:
    """Return index (1‑based) of row with maximum non‑nan value."""
    max_val = -float("inf")
    idx = -1
    for i, v in enumerate(preds, start=1):
        if math.isnan(v):
            continue
        if v > max_val:
            max_val = v
            idx = i
    return idx

# ------------------------------------------------------------------
# NASA‑15 helpers
# ------------------------------------------------------------------
def best_single_change_effort(row_idx: int,
                             rows: List[Dict[str, str]],
                             feature_names: List[str],
                             adjustable: List[str],
                             effort_pipe,
                             all_unique_vals: Dict[str, Set[str]]) -> Tuple[Dict[str, Tuple[str, str]], float, float]:
    """Find a single column change that reduces predicted effort most."""
    target_row = rows[row_idx-1]
    current_eff = effort_pipe.predict([prepare_feature_dict(target_row, feature_names)])[0]
    best_col = None
    best_new = None
    best_eff = current_eff

    for col in adjustable:
        cur_val = target_row[col]
        for val in all_unique_vals[col]:
            if val == cur_val:
                continue
            # build modified row
            mod_row = target_row.copy()
            mod_row[col] = val
            pred_eff = effort_pipe.predict([prepare_feature_dict(mod_row, feature_names)])[0]
            if pred_eff < best_eff:
                best_eff = pred_eff
                best_col = col
                best_new = val

    changes = {}
    if best_col:
        changes[best_col] = [cur_val, best_new]
    return changes, round_val(current_eff), round_val(best_eff)

# ------------------------------------------------------------------
# NASA‑16 helpers
# ------------------------------------------------------------------
def best_change_defects(row_idx: int,
                        rows: List[Dict[str, str]],
                        feature_names: List[str],
                        adjustable: List[str],
                        effort_pipe,
                        defects_pipe,
                        all_unique_vals: Dict[str, Set[str]]) -> Tuple[Dict[str, Tuple[str, str]], float, float, float, float]:
    """
    Find smallest combination (up to 3) of changes that reduces predicted defects most.
    Return changes dict, before/after defects, before/after effort.
    """
    target_row = rows[row_idx-1]
    current_def = defects_pipe.predict([prepare_feature_dict(target_row, feature_names)])[0]
    current_eff = effort_pipe.predict([prepare_feature_dict(target_row, feature_names)])[0]

    # For each column compute best new value that reduces defects
    best_for_col = {}
    for col in adjustable:
        cur_val = target_row[col]
        best_val = None
        best_def = current_def
        for val in all_unique_vals[col]:
            if val == cur_val:
                continue
            mod_row = target_row.copy()
            mod_row[col] = val
            pred_def = defects_pipe.predict([prepare_feature_dict(mod_row, feature_names)])[0]
            if pred_def < best_def:
                best_def = pred_def
                best_val = val
        if best_val is not None:
            best_for_col[col] = (best_val, best_def)

    # If no single column improves, return empty
    if not best_for_col:
        return {}, round_val(current_def), round_val(current_def), round_val(current_eff), round_val(current_eff)

    # Try combinations of size 1, 2, 3
    best_combo = None
    best_def_after = current_def
    for size in [1, 2, 3]:
        if size > len(best_for_col):
            continue
        for cols in itertools.combinations(best_for_col.keys(), size):
            mod_row = target_row.copy()
            for c in cols:
                mod_row[c] = best_for_col[c][0]
            pred_def = defects_pipe.predict([prepare_feature_dict(mod_row, feature_names)])[0]
            if pred_def < best_def_after:
                best_def_after = pred_def
                best_combo = cols
        if best_combo:
            break  # prefer smallest size

    if not best_combo:
        # No improvement
        return {}, round_val(current_def), round_val(current_def), round_val(current_eff), round_val(current_eff)

    # Build changes dict
    changes = {}
    for c in best_combo:
        changes[c] = [target_row[c], best_for_col[c][0]]

    # Compute effort after changes
    mod_row = target_row.copy()
    for c in best_combo:
        mod_row[c] = best_for_col[c][0]
    eff_after = effort_pipe.predict([prepare_feature_dict(mod_row, feature_names)])[0]

    return changes, round_val(current_def), round_val(best_def_after), round_val(current_eff), round_val(eff_after)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # Feature columns: all non‑goal, non‑X
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    x_cols = {c for c in header if c.endswith("X")}
    feature_names = [c for c in header if c not in goal_cols and c not in x_cols]

    adjustable = get_adjustable_columns(header)

    # Build pipelines
    effort_feature_rows = [prepare_feature_dict(r, feature_names) for r in rows]
    defects_feature_rows = effort_feature_rows  # same features

    effort_pipe = build_pipeline(effort_feature_rows, feature_names)
    defects_pipe = build_pipeline(defects_feature_rows, feature_names)

    # Train on all rows (including those with missing goal values; model will ignore NaN predictions)
    effort_pipe.fit(effort_feature_rows, [to_float(r["Effort-"]) for r in rows])
    defects_pipe.fit(defects_feature_rows, [to_float(r["Defects-"]) for r in rows])

    # Compute predictions
    eff_preds, def_preds = compute_predictions(rows, feature_names, effort_pipe, defects_pipe)

    # Gather unique values per adjustable column
    all_unique_vals: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        for col in adjustable:
            val = r[col]
            if val.strip() != "?":
                all_unique_vals[col].add(val)

    # NASA‑15: find row with highest predicted effort
    proj_eff_idx = find_max_row(eff_preds)
    changes_eff, before_eff, after_eff = best_single_change_effort(
        proj_eff_idx, rows, feature_names, adjustable, effort_pipe, all_unique_vals)

    print("== NASA-15 ==")
    print(f"Project: {proj_eff_idx}")
    print(f"Changes: {json.dumps(changes_eff)}")
    print(f"Predicted Effort before: {before_eff}, after: {after_eff}")
    answer_15 = {
        "project": proj_eff_idx,
        "changes": changes_eff,
        "predicted_effort": {"before": before_eff, "after": after_eff}
    }
    print("ANSWER NASA-15 " + json.dumps(answer_15))

    # NASA‑16: find row with highest predicted defects
    proj_def_idx = find_max_row(def_preds)
    changes_def, before_def, after_def, before_eff_def, after_eff_def = best_change_defects(
        proj_def_idx, rows, feature_names, adjustable, effort_pipe, defects_pipe, all_unique_vals)

    print("\n== NASA-16 ==")
    print(f"Project: {proj_def_idx}")
    print(f"Changes: {json.dumps(changes_def)}")
    print(f"Predicted Defects before: {before_def}, after: {after_def}")
    print(f"Predicted Effort before: {before_eff_def}, after: {after_eff_def}")
    answer_16 = {
        "project": proj_def_idx,
        "changes": changes_def,
        "predicted_defects": {"before": before_def, "after": after_def},
        "predicted_effort": {"before": before_eff_def, "after": after_eff_def}
    }
    print("ANSWER NASA-16 " + json.dumps(answer_16))

if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
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


def prepare_feature_dict(row: Dict[str, str], feature_names: List[str]) -> Dict[str, Any]:
    """Convert a row dict to a feature dict suitable for the pipeline."""
    d: Dict[str, Any] = {}
    for col in feature_names:
        val = row[col]
        if val.strip() == "?":
            d[col] = None  # will be handled by imputer / encoder
        else:
            # Decide if numeric or categorical
            if is_numeric(val):
                d[col] = float(val)
            else:
                d[col] = val
    return d


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute sample mean and sample std dev of a list of floats."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return (mean, math.sqrt(var))


# ------------------------------------------------------------------
# NASA-07: arrivals assessment
# ------------------------------------------------------------------
def nasa_07(rows: List[Dict[str, str]],
            header: List[str],
            history_count: int,
            arrivals: List[Dict[str, str]]) -> Dict[str, Any]:
    # Goal columns and X columns
    GOAL_COLUMNS = ["Kloc+", "Effort-", "Defects-"]
    X_COLUMNS = {c for c in header if c.endswith("X")}

    # Feature columns: all except GOAL_COLUMNS and X_COLUMNS
    feature_names = [c for c in header if c not in GOAL_COLUMNS and c not in X_COLUMNS]

    # Training rows for models: history only
    history_rows = rows[:history_count]

    # Build and train Effort- model
    effort_feature_rows = [prepare_feature_dict(r, feature_names) for r in history_rows]
    effort_targets = [to_float(r["Effort-"]) for r in history_rows]
    effort_pipe = build_pipeline(effort_feature_rows, feature_names)
    effort_pipe.fit(effort_feature_rows, effort_targets)

    # Build and train Defects- model
    defects_feature_rows = [prepare_feature_dict(r, feature_names) for r in history_rows]
    defects_targets = [to_float(r["Defects-"]) for r in history_rows]
    defects_pipe = build_pipeline(defects_feature_rows, feature_names)
    defects_pipe.fit(defects_feature_rows, defects_targets)

    # Compute historical mean and std for alert threshold
    effort_vals = [v for v in effort_targets if not math.isnan(v)]
    defects_vals = [v for v in defects_targets if not math.isnan(v)]
    effort_mean, effort_std = compute_mean_std(effort_vals)
    defects_mean, defects_std = compute_mean_std(defects_vals)

    assessments = []
    for i, row in enumerate(arrivals, start=history_count + 1):
        feat_dict = prepare_feature_dict(row, feature_names)
        pred_eff = effort_pipe.predict([feat_dict])[0]
        pred_def = defects_pipe.predict([feat_dict])[0]
        pred_eff_r = round(pred_eff, 4)
        pred_def_r = round(pred_def, 4)
        # Determine alert
        alert_eff = pred_eff_r > effort_mean + 2 * effort_std
        alert_def = pred_def_r > defects_mean + 2 * defects_std
        alert = alert_eff or alert_def
        assessments.append({
            "row": i,
            "predicted_effort": pred_eff_r,
            "predicted_defects": pred_def_r,
            "alert": alert
        })
    return {"assessments": assessments}


# ------------------------------------------------------------------
# NASA-08: checkpoints and shortlist
# ------------------------------------------------------------------
def nasa_08(rows: List[Dict[str, str]],
            header: List[str],
            history_count: int,
            arrivals: List[Dict[str, str]]) -> Dict[str, Any]:
    GOAL_COLUMNS = ["Kloc+", "Effort-", "Defects-"]
    # Determine checkpoints
    arrivals_count = len(arrivals)
    checkpoints_m = [0, 9, 19]
    checkpoints = []
    for m in checkpoints_m:
        if m > arrivals_count:
            continue
        data_rows = rows[:history_count + m]
        summary = {}
        for col in GOAL_COLUMNS:
            vals = [to_float(r[col]) for r in data_rows]
            vals = [v for v in vals if not math.isnan(v)]
            n = len(vals)
            if n == 0:
                mean = 0.0
                mn = 0.0
                mx = 0.0
            else:
                mean = round(sum(vals) / n, 4)
                mn = round(min(vals), 4)
                mx = round(max(vals), 4)
            summary[col] = {"n": n, "mean": mean, "min": mn, "max": mx}
        checkpoints.append({"after": m, "summary": summary})

    # Shortlist: best 5 rows using actual goal values
    all_rows = rows
    valid_rows = []
    for idx, row in enumerate(all_rows, start=1):
        vals = []
        skip = False
        for col in GOAL_COLUMNS:
            v = to_float(row[col])
            if math.isnan(v):
                skip = True
                break
            vals.append(v)
        if skip:
            continue
        # Ranking tuple: (-Kloc+, Effort-, Defects-)
        ranking = (-vals[0], vals[1], vals[2])
        valid_rows.append((ranking, idx))
    valid_rows.sort()
    shortlist = [idx for _, idx in valid_rows[:5]]
    return {"checkpoints": checkpoints, "shortlist": shortlist}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    csv_path = argv[1]
    rows, header = parse_csv(csv_path)
    n_rows = len(rows)
    arrivals_count = round(0.2 * n_rows)
    history_count = n_rows - arrivals_count
    arrivals = rows[history_count:]

    # NASA-07
    assessment_res = nasa_07(rows, header, history_count, arrivals)
    print("== NASA-07 ==")
    # print at least three lines
    if not assessment_res["assessments"]:
        print("No arrivals to assess.")
    else:
        for a in assessment_res["assessments"][:3]:
            print(f"Row {a['row']}: Effort-={a['predicted_effort']}, Defects-={a['predicted_defects']}, alert={a['alert']}")
    print("ANSWER NASA-07 " + json.dumps(assessment_res))

    # NASA-08
    res08 = nasa_08(rows, header, history_count, arrivals)
    print("\n== NASA-08 ==")
    # print at least three lines
    for ckpt in res08["checkpoints"][:3]:
        print(f"Checkpoint after {ckpt['after']} arrivals:")
        for g, stats in ckpt["summary"].items():
            print(f"{g}: n={stats['n']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
    # ensure at least three lines
    if len(res08["checkpoints"]) < 3:
        print("No additional data.")
    print("ANSWER NASA-08 " + json.dumps(res08))


if __name__ == "__main__":
    main(sys.argv)

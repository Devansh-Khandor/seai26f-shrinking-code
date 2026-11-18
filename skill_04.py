#!/usr/bin/env python3
"""
Skill 04: NASA-07 & NASA-08 dynamic analysis

Usage:
    python skill_04.py <path-to-nasa93dem.csv>
"""

import sys
import json
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# Import helpers from previous skills
try:
    from skill_01 import read_csv, format_number, compute_stats
    from skill_02 import determine_feature_types, build_encodings, build_feature_matrix
except ImportError:
    # Fallback minimal implementations
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

    def compute_stats(values):
        n = len(values)
        if n == 0:
            return {"mean": None, "median": None, "min": None, "max": None}
        sorted_vals = sorted(values)
        m = sum(values) / n
        mid = n // 2
        if n % 2 == 1:
            med = sorted_vals[mid]
        else:
            med = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return {"mean": m, "median": med, "min": sorted_vals[0], "max": sorted_vals[-1]}

    # Reimplement helpers used by skill_02
    def determine_feature_types(rows, feature_cols):
        numeric_cols = []
        categorical_cols = []
        for col in feature_cols:
            is_numeric = True
            for r in rows:
                v = r[col]
                if v is None:
                    continue
                if isinstance(v, float):
                    continue
                try:
                    float(v)
                except Exception:
                    is_numeric = False
                    break
            if is_numeric:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        return numeric_cols, categorical_cols

    def build_encodings(rows, categorical_cols):
        encodings = {}
        for col in categorical_cols:
            mapping = {}
            code = 0
            for r in rows:
                v = r[col]
                if v is None:
                    continue
                if v not in mapping:
                    mapping[v] = code
                    code += 1
            encodings[col] = mapping
        return encodings

    def build_feature_matrix(rows, feature_cols, numeric_cols, categorical_cols, encodings):
        X = []
        for r in rows:
            vec = []
            for col in feature_cols:
                v = r[col]
                if v is None:
                    vec.append(0.0)
                    continue
                if col in numeric_cols:
                    vec.append(float(v))
                else:
                    vec.append(encodings[col][v])
            X.append(vec)
        return np.array(X, dtype=float)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def compute_std(values):
    """Sample standard deviation with Bessel's correction."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))

def goal_columns(rows):
    """Return list of goal column names (end with + or - and not X)."""
    cols = list(rows[0].keys())
    return [c for c in cols if (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

# ----------------------------------------------------------------------
# NASA-07: arrival assessment
# ----------------------------------------------------------------------
def nasa_07(rows):
    # Determine arrivals
    total_rows = len(rows)
    arrivals_count = int(round(0.2 * total_rows))
    if arrivals_count > total_rows:
        arrivals_count = total_rows
    history_rows = rows[:total_rows - arrivals_count]
    arrivals_rows = rows[total_rows - arrivals_count:]

    # Precompute feature columns (exclude goal columns)
    all_cols = list(rows[0].keys())
    goal_cols = goal_columns(rows)
    feature_cols = [c for c in all_cols if c not in goal_cols + ['MonthsX']]

    assessments = []

    for i, arrival in enumerate(arrivals_rows):
        # Training data: history + earlier arrivals
        train_rows = history_rows + arrivals_rows[:i]
        # Ensure we only keep rows with complete feature data
        train_rows = [r for r in train_rows if all(r[c] is not None for c in feature_cols)]

        if not train_rows:
            # Cannot train; skip
            assessments.append({
                "row": total_rows - arrivals_count + i + 1,
                "predicted_effort": None,
                "predicted_defects": None,
                "alert": False
            })
            continue

        # Determine feature types and encodings
        numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
        encodings = build_encodings(train_rows, categorical_cols)

        # Build training matrices
        X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols, categorical_cols, encodings)
        y_effort_train = np.array([r['Effort-'] for r in train_rows if r['Effort-'] is not None], dtype=float)
        y_defects_train = np.array([r['Defects-'] for r in train_rows if r['Defects-'] is not None], dtype=float)

        # Train models
        model_effort = LinearRegression()
        model_effort.fit(X_train, y_effort_train)
        model_defects = LinearRegression()
        model_defects.fit(X_train, y_defects_train)

        # Compute thresholds from training data
        effort_values = [r['Effort-'] for r in train_rows if r['Effort-'] is not None]
        defects_values = [r['Defects-'] for r in train_rows if r['Defects-'] is not None]
        eff_mean = sum(effort_values) / len(effort_values) if effort_values else 0
        eff_std = compute_std(effort_values)
        def_mean = sum(defects_values) / len(defects_values) if defects_values else 0
        def_std = compute_std(defects_values)

        eff_threshold = eff_mean + 2 * eff_std
        def_threshold = def_mean + 2 * def_std

        # Prepare arrival feature vector
        if any(arrival[c] is None for c in feature_cols):
            # Missing feature data: skip prediction
            assessments.append({
                "row": total_rows - arrivals_count + i + 1,
                "predicted_effort": None,
                "predicted_defects": None,
                "alert": False
            })
            continue

        vec = []
        for col in feature_cols:
            v = arrival[col]
            if col in numeric_cols:
                vec.append(float(v))
            else:
                vec.append(encodings[col][v])
        X_test = np.array([vec], dtype=float)

        pred_effort = model_effort.predict(X_test)[0]
        pred_defects = model_defects.predict(X_test)[0]

        alert = (pred_effort > eff_threshold) or (pred_defects > def_threshold)

        assessments.append({
            "row": total_rows - arrivals_count + i + 1,
            "predicted_effort": float(pred_effort),
            "predicted_defects": float(pred_defects),
            "alert": alert
        })

    # Print results
    print("== NASA-07 ==")
    for a in assessments:
        rnum = a["row"]
        pe = a["predicted_effort"]
        pd = a["predicted_defects"]
        print(f"Row {rnum}: Effort- predicted={format_number(pe)}, Defects- predicted={format_number(pd)}, alert={str(a['alert']).lower()}")

    # JSON answer line
    json_obj = {"assessments": assessments}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-07 {json_str}")

# ----------------------------------------------------------------------
# NASA-08: checkpoints and shortlist
# ----------------------------------------------------------------------
def nasa_08(rows):
    total_rows = len(rows)
    arrivals_count = int(round(0.2 * total_rows))
    if arrivals_count > total_rows:
        arrivals_count = total_rows
    history_rows = rows[:total_rows - arrivals_count]
    arrivals_rows = rows[total_rows - arrivals_count:]

    goal_cols = goal_columns(rows)
    checkpoints = [0, 9, 19]
    checkpoints = [m for m in checkpoints if m <= arrivals_count]
    if 0 not in checkpoints:
        checkpoints = [0] + checkpoints
    # Ensure uniqueness
    checkpoints = sorted(set(checkpoints))

    # For each checkpoint, build data up to that point
    results = []

    for m in checkpoints:
        seen_rows = history_rows + arrivals_rows[:m]
        # Statistics for each goal column
        summary = {}
        for g in goal_cols:
            vals = [r[g] for r in seen_rows if r[g] is not None]
            if vals:
                stats = compute_stats(vals)
                summary[g] = {
                    "n": len(vals),
                    "mean": stats["mean"],
                    "min": stats["min"],
                    "max": stats["max"]
                }
            else:
                summary[g] = {"n": 0, "mean": None, "min": None, "max": None}

        # Shortlist: top 5 rows by ranking
        # Ranking: Kloc+ desc, Effort- asc, Defects- asc
        # Need to ensure rows have these values
        ranking_rows = [r for r in seen_rows if r['Kloc+'] is not None and r['Effort-'] is not None and r['Defects-'] is not None]
        ranking_rows.sort(key=lambda r: (-r['Kloc+'], r['Effort-'], r['Defects-']))
        shortlist = [rows.index(r) + 1 for r in ranking_rows[:5]]

        results.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # Print results
    print("== NASA-08 ==")
    for chk in results:
        print(f"After {chk['after']} arrivals:")
        for g in sorted(chk['summary']):
            s = chk['summary'][g]
            print(f"{g}: n={s['n']}, mean={format_number(s['mean'])}, min={format_number(s['min'])}, max={format_number(s['max'])}")
        print(f"Shortlist: {chk['shortlist']}")

    # JSON answer line
    json_obj = {"checkpoints": results}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-08 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_07(rows)
    nasa_08(rows)

if __name__ == "__main__":
    main()

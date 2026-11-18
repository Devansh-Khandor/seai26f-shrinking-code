#!/usr/bin/env python3
"""
skill_04.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-07: As new project proposals arrive, assess each proposal using its
             available characteristics and alert when its predicted effort or
             defects appear unusually high.
    NASA-08: As additional completed-project records become available, update
             our historical summaries and project estimates so that future
             assessments reflect the latest information.

Usage:
    python skill_04.py <csv path>
"""

import json
import sys
import statistics
import math

import skill_02  # reuse helper functions and pipelines


# --------------------------------------------------------------------------- #
# Helper functions for NASA-07
# --------------------------------------------------------------------------- #
def percentile(values, p=90):
    """Return the p‑th percentile (0‑100) of a numeric list."""
    if not values:
        return None
    sorted_vals = sorted(values)
    k = math.ceil(p / 100 * len(sorted_vals)) - 1
    k = max(0, min(k, len(sorted_vals) - 1))
    return sorted_vals[k]


def build_feature_matrix(rows, feature_names):
    """Return a numpy array of shape (n_rows, n_features)."""
    data = []
    for r in rows:
        row_vals = []
        for col in feature_names:
            v = r[col]
            if v == "?":
                row_vals.append(float("nan"))
            else:
                row_vals.append(v)
        data.append(row_vals)
    return skill_02.build_feature_matrix(rows, feature_names)  # use skill_02 implementation


def train_and_predict_row(training_rows, target_row, goal, feature_names, numeric_indices, categorical_indices):
    """Train on training_rows and predict `goal` for target_row. Return float."""
    # Build X_train and y_train with only rows that have the target present
    X_train = []
    y_train = []
    for r in training_rows:
        val = skill_02.parse_float(r[goal])
        if val is None:
            continue
        X_train.append(r)
        y_train.append(val)

    if not X_train:
        return float("nan")

    # Build feature matrix for training
    X_train_matrix = build_feature_matrix(X_train, feature_names)
    # Separate numeric and categorical columns
    X_train_numeric = X_train_matrix[:, numeric_indices].astype(float, copy=False)
    X_train_categorical = X_train_matrix[:, categorical_indices].astype(str, copy=False)
    X_train_combined = X_train_matrix.copy()
    X_train_combined[:, numeric_indices] = X_train_numeric
    X_train_combined[:, categorical_indices] = X_train_categorical

    # Build pipeline
    model = skill_02.build_pipeline(numeric_indices, categorical_indices)
    model.fit(X_train_combined, y_train)

    # Prepare target row
    X_test = build_feature_matrix([target_row], feature_names)
    X_test_numeric = X_test[:, numeric_indices].astype(float, copy=False)
    X_test_categorical = X_test[:, categorical_indices].astype(str, copy=False)
    X_test_combined = X_test.copy()
    X_test_combined[:, numeric_indices] = X_test_numeric
    X_test_combined[:, categorical_indices] = X_test_categorical

    pred = model.predict(X_test_combined)[0]
    return float(pred)


# --------------------------------------------------------------------------- #
# Helper functions for NASA-08
# --------------------------------------------------------------------------- #
def summary_stats(rows, goal):
    vals = [skill_02.parse_float(r[goal]) for r in rows if skill_02.parse_float(r[goal]) is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
    }


def shortlist_top(rows, top_n=5):
    ranked = []
    for idx, r in enumerate(rows):
        k = skill_02.parse_float(r["Kloc+"])
        e = skill_02.parse_float(r["Effort-"])
        d = skill_02.parse_float(r["Defects-"])
        if k is None or e is None or d is None:
            continue
        ranked.append((-k, e, d, idx + 1))  # 1‑based row number
    ranked.sort()
    return [t[3] for t in ranked[:top_n]]


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    all_rows = skill_02.load_csv(csv_path)
    n_rows = len(all_rows)

    # Determine arrivals (last round(0.2*n) rows)
    arrivals_count = round(n_rows * 0.2)
    history_rows = all_rows[: n_rows - arrivals_count]
    arrival_rows = all_rows[n_rows - arrivals_count :]

    # ------------------------------------------------------------ #
    # NASA-07
    # ------------------------------------------------------------ #
    print("== NASA-07 ==")

    # Historical goal values for thresholding
    historic_effort = [skill_02.parse_float(r["Effort-"]) for r in history_rows if skill_02.parse_float(r["Effort-"]) is not None]
    historic_defects = [skill_02.parse_float(r["Defects-"]) for r in history_rows if skill_02.parse_float(r["Defects-"]) is not None]
    eff_threshold = percentile(historic_effort, 90)
    def_threshold = percentile(historic_defects, 90)

    # Feature info
    feature_names, numeric_indices, categorical_indices = skill_02.identify_features(all_rows, ["Kloc+", "Effort-", "Defects-"])

    training_rows = list(history_rows)  # copy
    assessments = []

    for idx, arr_row in enumerate(arrival_rows):
        row_num = len(history_rows) + idx + 1

        pred_effort = train_and_predict_row(training_rows, arr_row, "Effort-", feature_names, numeric_indices, categorical_indices)
        pred_defects = train_and_predict_row(training_rows, arr_row, "Defects-", feature_names, numeric_indices, categorical_indices)

        # Determine alert
        alert = False
        if not math.isnan(pred_effort) and eff_threshold is not None and pred_effort > eff_threshold:
            alert = True
        if not math.isnan(pred_defects) and def_threshold is not None and pred_defects > def_threshold:
            alert = True

        assessments.append({
            "row": row_num,
            "predicted_effort": pred_effort,
            "predicted_defects": pred_defects,
            "alert": alert,
        })

        # Update training set for next arrival
        training_rows.append(arr_row)

    # Simple result lines
    print(f"Number of arrivals: {len(arrival_rows)}")
    if assessments:
        first = assessments[0]
        print(f"First prediction: row={first['row']}, effort={first['predicted_effort']:.2f}, defects={first['predicted_defects']:.2f}")
        print(f"Alerts detected: {sum(a['alert'] for a in assessments)}")
    else:
        print("No arrivals processed")

    answer_07 = {"assessments": assessments}
    print(f"ANSWER NASA-07 {json.dumps(answer_07, separators=(',', ':'))}")

    # ------------------------------------------------------------ #
    # NASA-08
    # ------------------------------------------------------------ #
    print("\n== NASA-08 ==")

    checkpoints = [0, 9, 19]
    checkpoint_info = []

    for m in checkpoints:
        if m > arrivals_count:
            continue
        rows_upto = history_rows + arrival_rows[:m]
        summary = {
            "Kloc+": summary_stats(rows_upto, "Kloc+"),
            "Effort-": summary_stats(rows_upto, "Effort-"),
            "Defects-": summary_stats(rows_upto, "Defects-"),
        }
        shortlist = shortlist_top(rows_upto, top_n=5)

        checkpoint_info.append({"after": m, "summary": summary, "shortlist": shortlist})

        # Print some output for the checkpoint
        print(f"Checkpoint after {m} arrivals:")
        print(f"  Kloc+ n={summary['Kloc+']['n']}, mean={summary['Kloc+']['mean']:.2f}")
        print(f"  Effort- n={summary['Effort-']['n']}, mean={summary['Effort-']['mean']:.2f}")
        print(f"  Defects- n={summary['Defects-']['n']}, mean={summary['Defects-']['mean']:.2f}")
        print(f"  Shortlist top 5 rows: {shortlist}")

    answer_08 = {"checkpoints": checkpoint_info}
    print(f"ANSWER NASA-08 {json.dumps(answer_08, separators=(',', ':'))}")


if __name__ == "__main__":
    main()

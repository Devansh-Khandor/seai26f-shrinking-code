#!/usr/bin/env python3
"""
Skill 09 – Outlier detection and anomaly flagging.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import json
import sys
import math
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Import helpers from earlier skill
# --------------------------------------------------------------------------- #
try:
    from skill_01 import parse_float, compute_stats  # type: ignore
except Exception:
    # Fallback definitions if import fails (unlikely in contest env)
    def parse_float(s: str):
        s = s.strip()
        if s == '' or s == '?':
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def compute_stats(values: List[float]) -> Dict[str, float]:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = sum(sorted_vals) / n
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        return {"mean": mean, "median": median, "min": sorted_vals[0], "max": sorted_vals[-1]}

# --------------------------------------------------------------------------- #
# Reuse machine learning utilities
# --------------------------------------------------------------------------- #
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np

# --------------------------------------------------------------------------- #
# Helper to get input columns (exclude goal and X)
# --------------------------------------------------------------------------- #
def get_input_columns(row_keys: List[str]) -> List[str]:
    return [c for c in row_keys if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))]

# --------------------------------------------------------------------------- #
# CAR-17: Unusual input combinations
# --------------------------------------------------------------------------- #
def car_17(data: List[Dict[str, Optional[float]]]) -> Tuple[List[int], Dict[int, float]]:
    input_cols = get_input_columns(list(data[0].keys()))

    # Compute mean & std per input column
    means = {}
    stds = {}
    for col in input_cols:
        vals = [row[col] for row in data if row[col] is not None]
        if not vals:
            means[col] = 0.0
            stds[col] = 1.0
        else:
            means[col] = sum(vals) / len(vals)
            # sample std
            mean_val = means[col]
            var = sum((x - mean_val) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
            stds[col] = math.sqrt(var) if var > 0 else 1.0

    # Compute unusualness score per row
    scores: Dict[int, float] = {}
    for idx, row in enumerate(data, start=1):
        if any(row[col] is None for col in input_cols):
            continue  # skip rows with missing inputs
        z2_sum = 0.0
        for col in input_cols:
            val = row[col]
            z = (val - means[col]) / stds[col]
            z2_sum += z ** 2
        scores[idx] = z2_sum

    if not scores:
        return [], {}

    # Determine threshold: mean + 2 * std of scores
    score_vals = list(scores.values())
    mean_score = sum(score_vals) / len(score_vals)
    var_score = sum((s - mean_score) ** 2 for s in score_vals) / (len(score_vals) - 1) if len(score_vals) > 1 else 0.0
    std_score = math.sqrt(var_score) if var_score > 0 else 0.0
    threshold = mean_score + 2 * std_score

    flagged = [row for row, sc in scores.items() if sc > threshold]
    flagged.sort(key=lambda r: scores[r], reverse=True)
    # Limit to 40
    flagged = flagged[:40]

    return flagged, scores

# --------------------------------------------------------------------------- #
# CAR-18: Anomalous target values
# --------------------------------------------------------------------------- #
def car_18(data: List[Dict[str, Optional[float]]]) -> List[Dict]:
    input_cols = get_input_columns(list(data[0].keys()))
    target_cols = ["Mpg+", "Lbs-"]

    # Build training data: rows with all input cols present (for model) and target not missing
    train_rows = []
    for row in data:
        if any(row[col] is None for col in input_cols + target_cols):
            continue
        train_rows.append(row)

    # If not enough data, return empty
    if len(train_rows) < 2:
        return []

    flagged_entries: List[Dict] = []

    for tcol in target_cols:
        # Prepare X, y
        X = np.array([[row[col] for col in input_cols] for row in train_rows])
        y = np.array([row[tcol] for row in train_rows], dtype=float)

        # Impute missing inputs if any
        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(X)

        model = LinearRegression()
        model.fit(X_imp, y)

        # Predict for all rows that have target value (even if inputs missing)
        for idx, row in enumerate(data, start=1):
            if row[tcol] is None:
                continue
            # Need inputs to predict
            if any(row[col] is None for col in input_cols):
                continue
            x_vec = np.array([[row[col] for col in input_cols]])
            x_imp = imputer.transform(x_vec)
            predicted = model.predict(x_imp)[0]
            actual = row[tcol]
            residual = abs(actual - predicted)

            # Compute threshold per target after collecting all residuals
            # For efficiency, compute residuals list first
            # We'll store them temporarily
            # We'll compute threshold after gathering all residuals
            # To keep logic simple, we will compute residuals first
            flagged_entries.append({"row": idx, "column": tcol, "actual": actual, "expected": predicted, "residual": residual})

    # Now compute thresholds per column
    # Separate residuals per column
    residuals_by_col: Dict[str, List[float]] = {"Mpg+": [], "Lbs-": []}
    for entry in flagged_entries:
        residuals_by_col[entry["column"]].append(entry["residual"])

    # Determine which entries exceed threshold
    final_flags: List[Dict] = []
    for entry in flagged_entries:
        col = entry["column"]
        res = entry["residual"]
        res_list = residuals_by_col[col]
        mean_res = sum(res_list) / len(res_list)
        var_res = sum((r - mean_res) ** 2 for r in res_list) / (len(res_list) - 1) if len(res_list) > 1 else 0.0
        std_res = math.sqrt(var_res) if var_res > 0 else 0.0
        threshold = mean_res + 2 * std_res
        if res > threshold:
            final_flags.append({
                "row": entry["row"],
                "column": entry["column"],
                "actual": round(entry["actual"], 6),
                "expected": round(entry["expected"], 6)
            })
    # Limit to 40
    final_flags = final_flags[:40]
    return final_flags

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Read data
    data: List[Dict[str, Optional[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    # CAR-17
    flagged_17, scores_17 = car_17(data)
    print("== CAR-17 ==")
    print(f"Flagged rows (unusual input combinations): {flagged_17}")
    if flagged_17:
        # show score of first flagged row
        first = flagged_17[0]
        print(f"Score for row {first}: {round(scores_17[first], 6)}")
    else:
        print("No flagged rows found.")

    ans_car_17 = json.dumps(
        {"flagged": flagged_17, "scores": {str(k): round(v, 6) for k, v in scores_17.items()}},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-17 {ans_car_17}")

    # CAR-18
    flagged_18 = car_18(data)
    print("\n== CAR-18 ==")
    if flagged_18:
        print(f"Flagged {len(flagged_18)} rows with anomalous target values.")
        for f in flagged_18[:3]:
            print(f"Row {f['row']}, column {f['column']}: actual={f['actual']}, expected={f['expected']}")
    else:
        print("No anomalous target values detected.")

    ans_car_18 = json.dumps(
        {"flagged": flagged_18},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-18 {ans_car_18}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

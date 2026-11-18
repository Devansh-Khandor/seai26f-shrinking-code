#!/usr/bin/env python3
"""
skill_06.py: NASA-11 and NASA-12 analyses.
Usage:  python skill_06.py <csv path>
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Import helper functions from earlier skills
# ----------------------------------------------------------------------
try:
    from skill_01 import parse_float, median, std_dev
except Exception:
    # Fallback implementations if skill_01 is not importable
    def parse_float(v):
        try:
            return float(v)
        except Exception:
            return None

    def median(lst):
        n = len(lst)
        if n == 0:
            return None
        s = sorted(lst)
        mid = n // 2
        if n % 2:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0

    def std_dev(values, mean):
        if len(values) == 0:
            return 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(var)

try:
    from skill_02 import train_linear, predict
except Exception:
    # Simple linear regression if skill_02 is not importable
    def train_linear(X, y):
        n = len(X)
        if n == 0:
            return 0.0, 0.0
        a = sum(x[0] * x[0] for x in X)
        b = sum(x[0] * x[1] for x in X)
        c = sum(x[1] * x[0] for x in X)
        d = sum(x[1] * x[1] for x in X)
        e = sum(x[0] * yi for x, yi in zip(X, y))
        f = sum(x[1] * yi for x, yi in zip(X, y))
        det = a * d - b * c
        if abs(det) < 1e-12:
            return sum(y) / n, 0.0
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det
        beta0 = inv_a * e + inv_b * f
        beta1 = inv_c * e + inv_d * f
        return beta0, beta1

    def predict(beta0, beta1, x):
        return beta0 + beta1 * x

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def compute_mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(values) / n
    var = sum((v - mean_val) ** 2 for v in values) / n
    return mean_val, math.sqrt(var)

# ----------------------------------------------------------------------
# NASA-07 style assessment (for risk alerts)
# ----------------------------------------------------------------------
def nasa07_assessments(rows, arrivals, budget):
    """
    rows: list of dicts with 'row_num'
    arrivals: list of last rows (history + first arrivals)
    budget: max number to evaluate
    Returns list of dicts with keys: row, predicted_effort, predicted_defects, alert
    """
    assessments = []

    # Separate training sets
    training_rows = [r for r in rows if r not in arrivals]
    training_effort_rows = [r for r in training_rows
                           if parse_float(r.get("Kloc+")) is not None
                           and parse_float(r.get("Effort-")) is not None]
    training_defect_rows = [r for r in training_rows
                            if parse_float(r.get("Kloc+")) is not None
                            and parse_float(r.get("Defects-")) is not None]
    training_effort_targets = [parse_float(r.get("Effort-")) for r in training_effort_rows]
    training_defect_targets = [parse_float(r.get("Defects-")) for r in training_defect_rows]

    processed = 0
    for row in arrivals:
        if processed >= budget:
            break

        row_num = row["row_num"]
        k = parse_float(row.get("Kloc+"))

        # Predict Effort-
        if training_effort_rows:
            X_eff = [[1.0, parse_float(r.get("Kloc+"))] for r in training_effort_rows]
            y_eff = [parse_float(r.get("Effort-")) for r in training_effort_rows]
            beta0_eff, beta1_eff = train_linear(X_eff, y_eff)
            if k is None:
                pred_eff = sum(y_eff) / len(y_eff)
            else:
                pred_eff = predict(beta0_eff, beta1_eff, k)
        else:
            pred_eff = 0.0

        # Predict Defects-
        if training_defect_rows:
            X_def = [[1.0, parse_float(r.get("Kloc+"))] for r in training_defect_rows]
            y_def = [parse_float(r.get("Defects-")) for r in training_defect_rows]
            beta0_def, beta1_def = train_linear(X_def, y_def)
            if k is None:
                pred_def = sum(y_def) / len(y_def)
            else:
                pred_def = predict(beta0_def, beta1_def, k)
        else:
            pred_def = 0.0

        # Compute thresholds from current training set
        mean_eff, std_eff = compute_mean_std(training_effort_targets)
        mean_def, std_def = compute_mean_std(training_defect_targets)

        # Alert if unusually high
        alert = False
        if std_eff > 0:
            if pred_eff > mean_eff + 2 * std_eff:
                alert = True
        else:
            if pred_eff > mean_eff:
                alert = True
        if std_def > 0:
            if pred_def > mean_def + 2 * std_def:
                alert = True
        else:
            if pred_def > mean_def:
                alert = True

        assessments.append({
            "row": row_num,
            "predicted_effort": round(pred_eff, 4),
            "predicted_defects": round(pred_def, 4),
            "alert": alert
        })

        # Update training sets
        if k is not None:
            if parse_float(row.get("Effort-")) is not None:
                training_effort_rows.append(row)
                training_effort_targets.append(parse_float(row.get("Effort-")))
            if parse_float(row.get("Defects-")) is not None:
                training_defect_rows.append(row)
                training_defect_targets.append(parse_float(row.get("Defects-")))

        processed += 1

    return assessments

# ----------------------------------------------------------------------
# NASA-11: estimate Effort- for a held‑out row
# ----------------------------------------------------------------------
def nasa11_estimate(rows, held_out_nums):
    """
    Return dict with project, estimate, neighbors (list of dicts)
    """
    # Pick first held‑out row
    proj_row_num = min(held_out_nums)
    proj_row = next(r for r in rows if r["row_num"] == proj_row_num)
    # Determine numeric columns (excluding goals and X)
    numeric_cols = set()
    for r in rows:
        for col, val in r.items():
            if col in ("row_num",):
                continue
            if col.endswith('+') or col.endswith('-') or col.endswith('X'):
                continue
            if parse_float(val) is not None:
                numeric_cols.add(col)
    numeric_cols = sorted(numeric_cols)

    # Compute distances to all non‑held‑out rows
    distances = []
    for r in rows:
        if r["row_num"] in held_out_nums:
            continue
        dist_sq = 0.0
        valid = True
        for col in numeric_cols:
            v1 = parse_float(proj_row.get(col))
            v2 = parse_float(r.get(col))
            if v1 is None or v2 is None:
                continue
            diff = v1 - v2
            dist_sq += diff * diff
        if not valid:
            continue
        distances.append((dist_sq, r))

    # Take 3 nearest neighbors
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:3]

    # Estimate Effort- as weighted average by inverse distance
    est = 0.0
    weight_sum = 0.0
    for d, r in neighbors:
        eff = parse_float(r.get("Effort-"))
        if eff is None:
            continue
        weight = 1.0 / (d + 1e-6)
        est += weight * eff
        weight_sum += weight
    if weight_sum == 0:
        est = 0.0
    else:
        est = est / weight_sum
    est = round(est, 4)

    # Prepare neighbor details (include all columns as raw strings)
    neighbor_list = []
    for _, r in neighbors:
        n_dict = {}
        for col, val in r.items():
            n_dict[col] = val
        neighbor_list.append(n_dict)

    return {
        "project": proj_row_num,
        "estimate": est,
        "neighbors": neighbor_list
    }

# ----------------------------------------------------------------------
# NASA-12: compare two proposals with different risk assessments
# ----------------------------------------------------------------------
def nasa12_compare(rows, arrivals, assessments):
    """
    Return dict with a, b, preferred, scores, differences, tradeoffs
    """
    # Find first alert True and first alert False
    row_a = None
    row_b = None
    for a in assessments:
        if a["alert"]:
            row_a = a
            break
    for b in assessments:
        if not b["alert"]:
            row_b = b
            break
    if row_a is None or row_b is None:
        return None  # cannot compare

    # Prefer the less risky (alert False)
    preferred = row_b["row"] if not row_a["alert"] else row_a["row"]

    # Scores: use predicted_effort + predicted_defects as risk score (higher worse)
    score_a = row_a["predicted_effort"] + row_a["predicted_defects"]
    score_b = row_b["predicted_effort"] + row_b["predicted_defects"]

    # Differences: all columns (raw) that differ
    def get_row_dict(row_num):
        return next(r for r in rows if r["row_num"] == row_num)

    row_a_raw = get_row_dict(row_a["row"])
    row_b_raw = get_row_dict(row_b["row"])

    diffs = []
    for col in row_a_raw.keys():
        if col == "row_num":
            continue
        val_a = row_a_raw.get(col)
        val_b = row_b_raw.get(col)
        if val_a != val_b:
            diffs.append({"feature": col, "a": val_a, "b": val_b})
    if not diffs:
        diffs.append({"feature": "none", "a": "", "b": ""})

    # Tradeoffs: goal columns where the non‑preferred row has better value
    # goal columns: end with '+' (higher better), '-' (lower better)
    tradeoffs = []
    for col in row_a_raw.keys():
        if col.endswith('+'):
            # higher better
            v_a = parse_float(row_a_raw.get(col))
            v_b = parse_float(row_b_raw.get(col))
            if v_a is None or v_b is None:
                continue
            # if non‑preferred (row_a if preferred is row_b, etc.)
            if preferred == row_b["row"]:
                # preferred is row_b, so non-preferred is row_a
                if v_a > v_b:
                    tradeoffs.append(col)
            else:
                if v_b > v_a:
                    tradeoffs.append(col)
        elif col.endswith('-'):
            # lower better
            v_a = parse_float(row_a_raw.get(col))
            v_b = parse_float(row_b_raw.get(col))
            if v_a is None or v_b is None:
                continue
            if preferred == row_b["row"]:
                if v_a < v_b:
                    tradeoffs.append(col)
            else:
                if v_b < v_a:
                    tradeoffs.append(col)

    # Remove duplicates
    tradeoffs = list(dict.fromkeys(tradeoffs))

    return {
        "a": row_a["row"],
        "b": row_b["row"],
        "preferred": preferred,
        "scores": {str(row_a["row"]): round(score_a, 4),
                   str(row_b["row"]): round(score_b, 4)},
        "differences": diffs,
        "tradeoffs": tradeoffs
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    # Read CSV and keep row numbers
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            r = dict(r)  # copy
            r["row_num"] = idx
            rows.append(r)

    # Identify held‑out rows
    held_out_nums = {r["row_num"] for r in rows if r["row_num"] % 5 == 0}

    # Arrivals and budget
    n_rows = len(rows)
    arrival_count = round(0.2 * n_rows)
    arrivals = rows[-arrival_count:] if arrival_count > 0 else []
    budget = min(30, n_rows // 5)

    # ------------------------------------------------------------------
    # NASA-11
    # ------------------------------------------------------------------
    print("== NASA-11 ==")
    nasa11_res = nasa11_estimate(rows, held_out_nums)
    # Print at least three lines
    print(f"Project row {nasa11_res['project']}")
    print(f"Estimated Effort-: {nasa11_res['estimate']}")
    print(f"Neighbors: {[n['row_num'] for n in nasa11_res['neighbors']]}")
    nasa11_json = {
        "project": nasa11_res["project"],
        "estimate": nasa11_res["estimate"],
        "neighbors": nasa11_res["neighbors"]
    }
    print(f"ANSWER NASA-11 {json.dumps(nasa11_json)}")

    # ------------------------------------------------------------------
    # NASA-12
    # ------------------------------------------------------------------
    # Run assessments for arrivals
    assessments = nasa07_assessments(rows, arrivals, budget)
    nasa12_res = nasa12_compare(rows, arrivals, assessments)
    print("\n== NASA-12 ==")
    if nasa12_res:
        print(f"Row a: {nasa12_res['a']}")
        print(f"Row b: {nasa12_res['b']}")
        print(f"Preferred: {nasa12_res['preferred']}")
        print(f"Scores: {nasa12_res['scores']}")
        print(f"Differences: {nasa12_res['differences']}")
        print(f"Tradeoffs: {nasa12_res['tradeoffs']}")
        print(f"ANSWER NASA-12 {json.dumps(nasa12_res)}")
    else:
        print("No suitable pair of proposals with different risk assessments found.")
        print("ANSWER NASA-12 {}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

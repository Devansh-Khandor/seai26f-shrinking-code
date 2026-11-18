#!/usr/bin/env python3
"""
skill_04.py: NASA-07 and NASA-08 analyses.
Usage:  python skill_04.py <csv path>
"""

import csv
import json
import math
import sys

# Import helper functions from earlier skills
from skill_01 import parse_float, median, std_dev
from skill_02 import train_linear, predict


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def compute_stats(values):
    """Return (n, mean, min, max) for a list of floats."""
    n = len(values)
    if n == 0:
        return 0, None, None, None
    mean_val = sum(values) / n
    return n, mean_val, min(values), max(values)


def compute_mean_std(values):
    """Return mean and std for a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(values) / n
    var = sum((v - mean_val) ** 2 for v in values) / n
    return mean_val, math.sqrt(var)


def compute_combined_score(row, means, stds):
    """Compute combined normalized score for a row."""
    # Handle missing values safely
    k = parse_float(row.get("Kloc+"))
    e = parse_float(row.get("Effort-"))
    d = parse_float(row.get("Defects-"))
    if k is None or e is None or d is None:
        return None

    # Normalize
    std_k = stds["Kloc+"] if stds["Kloc+"] > 0 else 1.0
    std_e = stds["Effort-"] if stds["Effort-"] > 0 else 1.0
    std_d = stds["Defects-"] if stds["Defects-"] > 0 else 1.0

    z_k = (k - means["Kloc+"]) / std_k
    z_e = (means["Effort-"] - e) / std_e      # lower is better
    z_d = (means["Defects-"] - d) / std_d

    return z_k + z_e + z_d


# ----------------------------------------------------------------------
# NASA-07: Predict and alert on arrivals
# ----------------------------------------------------------------------
def nasa07(rows, arrivals, budget):
    assessments = []

    # Training data starts with history rows
    training_rows = [r for r in rows if r not in arrivals]

    # Keep track of training sets for regression
    training_effort_rows = [r for r in training_rows if parse_float(r.get("Kloc+")) is not None and parse_float(r.get("Effort-")) is not None]
    training_defect_rows = [r for r in training_rows if parse_float(r.get("Kloc+")) is not None and parse_float(r.get("Defects-")) is not None]

    # Also keep target values for thresholds
    training_effort_targets = [parse_float(r.get("Effort-")) for r in training_effort_rows]
    training_defect_targets = [parse_float(r.get("Defects-")) for r in training_defect_rows]

    processed = 0
    for row in arrivals:
        if processed >= budget:
            break

        row_num = row["row_num"]
        k = parse_float(row["Kloc+"])

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

        # Update training sets with this row for next predictions
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
# NASA-08: Checkpoints and best rows
# ----------------------------------------------------------------------
def nasa08(rows, arrivals, checkpoint_ms):
    checkpoints = []

    # Determine mapping of row numbers to rows
    row_map = {r["row_num"]: r for r in rows}

    for m in checkpoint_ms:
        # Rows up to m arrivals (history + first m arrivals)
        seen_rows = rows[: len(rows) - len(arrivals)]  # history
        seen_rows += arrivals[:m]

        # Compute stats for each goal
        goal_cols = ["Kloc+", "Effort-", "Defects-"]
        summary = {}
        for col in goal_cols:
            vals = [parse_float(r.get(col)) for r in seen_rows]
            vals = [v for v in vals if v is not None]
            n, mean_val, min_val, max_val = compute_stats(vals)
            summary[col] = {
                "n": n,
                "mean": round(mean_val, 4) if mean_val is not None else None,
                "min": round(min_val, 4) if min_val is not None else None,
                "max": round(max_val, 4) if max_val is not None else None
            }

        # Prepare means and stds for normalization
        means = {}
        stds = {}
        for col in goal_cols:
            vals = [parse_float(r.get(col)) for r in seen_rows]
            vals = [v for v in vals if v is not None]
            mean_val, std_val = compute_mean_std(vals)
            means[col] = mean_val
            stds[col] = std_val

        # Compute combined scores
        scored_rows = []
        for r in seen_rows:
            score = compute_combined_score(r, means, stds)
            if score is not None:
                scored_rows.append((score, r["row_num"]))

        scored_rows.sort(reverse=True)
        shortlist = [num for _, num in scored_rows[:5]]

        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    return checkpoints


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            r["row_num"] = idx
            rows.append(r)

    n = len(rows)
    arrival_count = round(0.2 * n)
    arrivals = rows[-arrival_count:] if arrival_count > 0 else []
    history = rows[:-arrival_count] if arrival_count > 0 else rows

    budget = min(30, n // 5)

    # NASA-07
    assessments = nasa07(history + arrivals, arrivals, budget)
    print("== NASA-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: effort {a['predicted_effort']} defects {a['predicted_defects']} alert {a['alert']}")
    print(f"ANSWER NASA-07 {json.dumps({'assessments': assessments})}")

    # NASA-08
    checkpoint_ms = [0, 9, 19]
    checkpoints = nasa08(history + arrivals, arrivals, checkpoint_ms)
    print("\n== NASA-08 ==")
    for cp in checkpoints:
        print(f"Checkpoint after {cp['after']} arrivals:")
        for g, stats in cp['summary'].items():
            print(f"  {g}: n={stats['n']} mean={stats['mean']} min={stats['min']} max={stats['max']}")
        print(f"  Shortlist: {cp['shortlist']}")
    print(f"ANSWER NASA-08 {json.dumps({'checkpoints': checkpoints})}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])

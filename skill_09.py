#!/usr/bin/env python3
"""
skill_09.py

Usage:
    python skill_09.py <path to Apache_AllMeasurements.csv>
"""

import json
import math
import sys

# Reuse helpers from earlier modules
import skill_01 as s01
import skill_02 as s02


def goal_column(header):
    """Return the first column that is a goal (+ or -)."""
    for col in header:
        if s01.is_goal(col):
            return col
    return None


def setting_columns(header):
    """Return all columns that are settings (not goals, not ending in X)."""
    return [c for c in header if not s01.is_goal(c) and not c.endswith("X")]


def hamming_distance(r1, r2, settings):
    """Hamming distance over settings, treating None as a difference."""
    dist = 0
    for s in settings:
        v1 = r1.get(s)
        v2 = r2.get(s)
        if v1 != v2:
            dist += 1
    return dist


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    std = math.sqrt(sum((x - m) ** 2 for x in values) / len(values))
    return m, std


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    goal_col = goal_column(header)
    if not goal_col:
        print("No goal column found.")
        sys.exit(1)

    settings = setting_columns(header)

    # ---------------- APACHE-17 ----------------
    print("== APACHE-17 ==")

    # Fit linear model using all rows that have valid data
    try:
        coeff = s02._fit_linear(rows, goal_col, settings)
    except Exception:
        coeff = None

    # Predict expected values and compute residuals
    residuals = []
    predictions = {}
    for idx, r in enumerate(rows, start=1):
        if r[goal_col] is None:
            continue
        if coeff is not None:
            pred = s02._predict_row(r, coeff, settings)
        else:
            pred = None
        predictions[idx] = pred
        if pred is not None:
            residuals.append(abs(pred - r[goal_col]))

    if residuals:
        mean_err, std_err = mean_std(residuals)
        threshold = mean_err + 2 * std_err if std_err > 0 else mean_err
    else:
        threshold = 0

    flagged17 = []
    for idx, r in enumerate(rows, start=1):
        if r[goal_col] is None or idx not in predictions or predictions[idx] is None:
            continue
        err = abs(predictions[idx] - r[goal_col])
        if err > threshold:
            flagged17.append({
                "row": idx,
                "column": goal_col,
                "actual": r[goal_col],
                "expected": predictions[idx]
            })
        if len(flagged17) >= 20:
            break

    # Human‑readable output
    print(f"Threshold for flagging: {threshold:.2f}")
    print(f"Flagged rows (up to 20):")
    for f in flagged17:
        print(f"  Row {f['row']}: {f['column']} actual={f['actual']}, expected={f['expected']:.2f}")

    apache17_answer = {"flagged": flagged17}
    print("ANSWER APACHE-17", json.dumps(apache17_answer, separators=(",", ":")))

    # ---------------- APACHE-18 ----------------
    print("\n== APACHE-18 ==")

    # Compute distance to nearest neighbor for each row
    distances = {}
    for i, r1 in enumerate(rows, start=1):
        best_dist = None
        for j, r2 in enumerate(rows, start=1):
            if i == j:
                continue
            d = hamming_distance(r1, r2, settings)
            if best_dist is None or d < best_dist:
                best_dist = d
        distances[i] = best_dist if best_dist is not None else 0

    # Statistics of distances
    dist_values = list(distances.values())
    mean_d, std_d = mean_std(dist_values)
    # Score is z‑score of distance
    scores = {}
    for idx, d in distances.items():
        score = (d - mean_d) / std_d if std_d > 0 else 0.0
        scores[idx] = score

    # Determine poor threshold for the goal (to exclude obviously poor rows)
    goal_vals = [r[goal_col] for r in rows if r[goal_col] is not None]
    mean_g, std_g = mean_std(goal_vals)
    if goal_col.endswith('-'):
        poor_thresh = mean_g + 1.96 * std_g
        is_poor = lambda v: v is not None and v > poor_thresh
    else:
        poor_thresh = mean_g - 1.96 * std_g
        is_poor = lambda v: v is not None and v < poor_thresh

    flagged18 = []
    for idx, score in scores.items():
        if score > 2.0:
            r = rows[idx - 1]
            if not is_poor(r[goal_col]):
                flagged18.append(idx)
        if len(flagged18) >= 20:
            break

    # Human‑readable output
    print(f"Mean distance: {mean_d:.2f}, std: {std_d:.2f}")
    print(f"Score threshold: 2.0")
    print(f"Flagged rows (up to 20) with unusual setting combinations:")
    for idx in flagged18:
        print(f"  Row {idx}: distance={distances[idx]}, score={scores[idx]:.2f}")

    apache18_answer = {"flagged": flagged18, "scores": {str(k): v for k, v in scores.items() if k in flagged18}}
    print("ANSWER APACHE-18", json.dumps(apache18_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skill 04: Adaptive project assessment and historical update.

Usage:
    python skill_04.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
from collections import defaultdict

# Only numpy and scikit‑learn are allowed
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def format_number(val, decimals=2):
    """Format number with a given precision, stripping trailing zeros."""
    if val is None or math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """
    Read the NASA data set.

    Returns:
        completed: list of dicts with Kloc, Effort, Defects (all present)
        proposals: list of dicts with Kloc and id (missing Effort or Defects)
    """
    completed = []
    proposals = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = float(row["Kloc+"].strip())
            except (ValueError, KeyError):
                continue
            if k <= 0:
                continue

            # Check whether effort and defects are present
            eff_str = row.get("Effort-", "").strip()
            def_str = row.get("Defects-", "").strip()

            id_ = row.get("idX", "").strip()

            if eff_str and def_str:
                # Completed project
                try:
                    eff = float(eff_str)
                    defs = float(def_str)
                except ValueError:
                    continue
                completed.append({"id": id_, "Kloc": k, "Effort": eff, "Defects": defs})
            else:
                # Proposal – we may or may not have Kloc
                proposals.append({"id": id_, "Kloc": k})

    return completed, proposals

# ---------------------------------------------------------------------------

def fit_linear(x, y):
    """
    Fit a simple linear regression model y = a*x + b.

    Returns the fitted model and the predicted y for the training points.
    """
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return model, pred

# ---------------------------------------------------------------------------

def compute_stats(values):
    """Return mean and std of a numeric list."""
    if not values:
        return None, None
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)  # population std
    return mean, std

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    completed, proposals = read_csv(csv_path)

    # ---------------------------------------------------------------
    # NASA-07: assess proposals
    print("== NASA-07 ==")
    if not completed:
        print("No completed project data available to build models.")
    elif not proposals:
        print("No proposals to assess.")
    else:
        # Fit models on completed projects
        # Use Kloc as the sole predictor
        k_values = [p["Kloc"] for p in completed]
        effort_values = [p["Effort"] for p in completed]
        defect_values = [p["Defects"] for p in completed]

        effort_model, effort_pred_train = fit_linear(k_values, effort_values)
        defect_model, defect_pred_train = fit_linear(k_values, defect_values)

        # Statistics of predictions on the training data
        eff_mean, eff_std = compute_stats(effort_pred_train)
        def_mean, def_std = compute_stats(defect_pred_train)

        # Thresholds for "unusually high" predictions
        eff_threshold = eff_mean + 2 * eff_std if eff_std is not None else None
        def_threshold = def_mean + 2 * def_std if def_std is not None else None

        flagged_effort = 0
        flagged_defects = 0

        for prop in proposals:
            k = prop["Kloc"]
            pid = prop["id"] or "UNKNOWN"

            eff_pred = effort_model.predict(np.array([[k]]))[0]
            def_pred = defect_model.predict(np.array([[k]]))[0]

            eff_alert = eff_threshold is not None and eff_pred > eff_threshold
            def_alert = def_threshold is not None and def_pred > def_threshold

            if eff_alert:
                flagged_effort += 1
            if def_alert:
                flagged_defects += 1

            print(f"  Proposal {pid}:")
            print(f"    Predicted effort   : {format_number(eff_pred)} man-months"
                  f"{'  ⚠ High effort!' if eff_alert else ''}")
            print(f"    Predicted defects : {format_number(def_pred)}"
                  f"{'  ⚠ High defects!' if def_alert else ''}")

        print(f"\n  Total proposals evaluated: {len(proposals)}")
        print(f"  Proposals with high predicted effort: {flagged_effort}")
        print(f"  Proposals with high predicted defects: {flagged_defects}")

    # ---------------------------------------------------------------
    # NASA-08: update historical summaries and estimates
    print("\n== NASA-08 ==")
    if not completed:
        print("No completed projects to update summaries.")
    else:
        # Compute overall statistics
        kloc_vals = [p["Kloc"] for p in completed]
        effort_vals = [p["Effort"] for p in completed]
        defect_vals = [p["Defects"] for p in completed]

        kloc_mean, kloc_std = compute_stats(kloc_vals)
        eff_mean, eff_std = compute_stats(effort_vals)
        def_mean, def_std = compute_stats(defect_vals)

        print(f"  Total completed projects: {len(completed)}")
        print(f"  Kloc+    mean: {format_number(kloc_mean)} (±{format_number(kloc_std)})")
        print(f"  Effort   mean: {format_number(eff_mean)} man-months (±{format_number(eff_std)})")
        print(f"  Defects  mean: {format_number(def_mean)} (±{format_number(def_std)})")

        # Re‑estimate using current data (just for demonstration)
        # (Here we reuse the models from NASA‑07 if they exist)
        if completed:
            # Use the same linear models (already built)
            eff_pred_mean, eff_pred_std = compute_stats(effort_pred_train)
            def_pred_mean, def_pred_std = compute_stats(defect_pred_train)

            print(f"\n  Estimated effort based on current data: "
                  f"{format_number(eff_pred_mean)} ± {format_number(eff_pred_std)}")
            print(f"  Estimated defects based on current data: "
                  f"{format_number(def_pred_mean)} ± {format_number(def_pred_std)}")

    # ---------------------------------------------------------------

if __name__ == "__main__":
    main()

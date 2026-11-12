#!/usr/bin/env python3
"""
Skill 10: Comparative evaluation of estimation and selection methods.

NASA-19: Compare the linear‑regression based effort/defect estimation system against
        a simple historical‑average estimate and show whether it makes
        meaningfully better predictions on unseen projects.

NASA-20: Compare our project‑selection approach with randomly selecting proposals
        under the same evaluation budget, and show whether our system
        identifies more desirable project outcomes.

Usage:
    python skill_10.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
import random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """
    Read the NASA data set and split into:
        - completed: projects with both Effort- and Defects- present
        - proposals: projects missing at least one of those fields
    """
    completed = []
    proposals = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip all fields
            clean = {k: v.strip() for k, v in row.items()}

            # Kloc+ is mandatory
            k_val = clean.get("Kloc+", "")
            if not k_val:
                continue
            try:
                k = float(k_val)
                if k <= 0:
                    continue
            except ValueError:
                continue

            id_ = clean.get("idX", "")

            eff_str = clean.get("Effort-", "")
            def_str = clean.get("Defects-", "")

            if eff_str and def_str:
                try:
                    eff = float(eff_str)
                    defs = float(def_str)
                except ValueError:
                    continue
                completed.append({"id": id_, "Kloc": k, "Effort": eff, "Defects": defs})
            else:
                proposals.append({"id": id_, "Kloc": k, "Effort": None, "Defects": None, **clean})

    return completed, proposals

# ---------------------------------------------------------------------------

def fit_regression(x, y):
    """Simple linear regression: y = a*x + b."""
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)
    model = LinearRegression()
    model.fit(X, Y)
    return model

# ---------------------------------------------------------------------------

def mae(pred, true):
    """Mean absolute error."""
    if len(pred) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - true)))

def format_number(val, decimals=3):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')

# ---------------------------------------------------------------------------

def nasa_19(completed):
    """
    Evaluate linear regression vs historical average on unseen projects.
    """
    # Split into training and test sets (80/20)
    train, test = train_test_split(completed, test_size=0.2, random_state=42)

    # Train models
    eff_train = [p["Effort"] for p in train]
    def_train = [p["Defects"] for p in train]
    k_train = [p["Kloc"] for p in train]

    eff_model = fit_regression(k_train, eff_train)
    def_model = fit_regression(k_train, def_train)

    # Historical averages
    eff_avg = np.mean(eff_train)
    def_avg = np.mean(def_train)

    # Test predictions
    k_test = [p["Kloc"] for p in test]
    eff_true = np.array([p["Effort"] for p in test])
    def_true = np.array([p["Defects"] for p in test])

    eff_pred_model = eff_model.predict(np.array(k_test).reshape(-1, 1))
    def_pred_model = def_model.predict(np.array(k_test).reshape(-1, 1))

    eff_pred_avg = np.full_like(eff_true, eff_avg)
    def_pred_avg = np.full_like(def_true, def_avg)

    # Errors
    eff_mae_model = mae(eff_pred_model, eff_true)
    def_mae_model = mae(def_pred_model, def_true)
    eff_mae_avg = mae(eff_pred_avg, eff_true)
    def_mae_avg = mae(def_pred_avg, def_true)

    # Print results
    print("== NASA-19 ==")
    print(f"Number of completed projects: {len(completed)}")
    print(f"Training set size: {len(train)}; Test set size: {len(test)}")
    print("\nEffort prediction error (MAE):")
    print(f"  Linear regression: {format_number(eff_mae_model)} man-months")
    print(f"  Historical average: {format_number(eff_mae_avg)} man-months")
    print("\nDefect prediction error (MAE):")
    print(f"  Linear regression: {format_number(def_mae_model)} defects")
    print(f"  Historical average: {format_number(def_mae_avg)} defects")
    print("\nInterpretation:")
    if eff_mae_model < eff_mae_avg and def_mae_model < def_mae_avg:
        print("  The regression model outperforms the historical average for both effort and defects.")
    elif eff_mae_model < eff_mae_avg:
        print("  The regression model outperforms the historical average for effort only.")
    elif def_mae_model < def_mae_avg:
        print("  The regression model outperforms the historical average for defects only.")
    else:
        print("  The regression model does not outperform the historical average.")

# ---------------------------------------------------------------------------

def nasa_20(completed, proposals):
    """
    Compare our selection strategy (lowest predicted effort+defect density)
    with random selection under the same evaluation budget.
    """
    if not proposals:
        print("\n== NASA-20 ==")
        print("No proposals available for selection.")
        return

    # Build models from all completed data
    k_all = [p["Kloc"] for p in completed]
    eff_all = [p["Effort"] for p in completed]
    def_all = [p["Defects"] for p in completed]

    eff_model = fit_regression(k_all, eff_all)
    def_model = fit_regression(k_all, def_all)

    # Predict for each proposal
    pred_eff = []
    pred_def = []
    for p in proposals:
        k = p["Kloc"]
        eff_pred = eff_model.predict([[k]])[0]
        def_pred = def_model.predict([[k]])[0]
        pred_eff.append(eff_pred)
        pred_def.append(def_pred)

    # Compute combined metric: effort density + defect density
    combined = [ (e + d) / p["Kloc"] for e, d, p in zip(pred_eff, pred_def, proposals) ]

    # Selection size: either 5 proposals or all if fewer
    budget = min(5, len(proposals))

    # Our selection: lowest combined metric
    indices_sorted = sorted(range(len(proposals)), key=lambda i: combined[i])
    selected_idx = indices_sorted[:budget]
    selected_eff = [pred_eff[i] for i in selected_idx]
    selected_def = [pred_def[i] for i in selected_idx]
    selected_kloc = [proposals[i]["Kloc"] for i in selected_idx]

    # Random selection
    random.seed(0)
    all_indices = list(range(len(proposals)))
    random.shuffle(all_indices)
    random_idx = all_indices[:budget]
    random_eff = [pred_eff[i] for i in random_idx]
    random_def = [pred_def[i] for i in random_idx]

    # Compute averages
    sel_eff_avg = np.mean(selected_eff) if selected_eff else float("nan")
    sel_def_avg = np.mean(selected_def) if selected_def else float("nan")
    rand_eff_avg = np.mean(random_eff) if random_eff else float("nan")
    rand_def_avg = np.mean(random_def) if random_def else float("nan")

    print("\n== NASA-20 ==")
    print(f"Evaluation budget: {budget} proposals")
    print("\nPredicted averages for selected proposals:")
    print(f"  Effort: {format_number(sel_eff_avg)} man-months")
    print(f"  Defects: {format_number(sel_def_avg)}")
    print("\nPredicted averages for random proposals:")
    print(f"  Effort: {format_number(rand_eff_avg)} man-months")
    print(f"  Defects: {format_number(rand_def_avg)}")
    print("\nInterpretation:")
    if sel_eff_avg < rand_eff_avg and sel_def_avg < rand_def_avg:
        print("  Our selection identifies projects with lower predicted effort and defects than random.")
    elif sel_eff_avg < rand_eff_avg:
        print("  Our selection identifies projects with lower predicted effort than random.")
    elif sel_def_avg < rand_def_avg:
        print("  Our selection identifies projects with lower predicted defects than random.")
    else:
        print("  Our selection does not outperform random selection.")

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_nasa93dem.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    completed, proposals = read_csv(csv_path)

    if not completed:
        print("No completed projects available for analysis.")
        sys.exit(1)

    nasa_19(completed)
    nasa_20(completed, proposals)

if __name__ == "__main__":
    main()

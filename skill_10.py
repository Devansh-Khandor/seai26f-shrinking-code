#!/usr/bin/env python3
"""
Skill 10: Evaluation of the effort estimation system and proposal selection strategy.

NASA-19: Compare the learned effort model against a simple historical‑average estimator
on unseen projects and report error metrics.

NASA-20: Compare the NASA‑07 proposal prioritisation against random selection
within the same evaluation budget, reporting average predicted effort and defects.

The script reuses the data loading from skill_01 and the regression helper
from skill_02.  It performs its own train/test splits and predictions
so that the results are reproducible for a single run.
"""

import sys
import random
import numpy as np
from sklearn.linear_model import LinearRegression

# Reuse data loading from skill_01
try:
    import skill_01 as s01
except ImportError:
    # Minimal fallback if skill_01 is missing
    import csv

    def load_data(csv_path: str):
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r['Kloc+'] = float(r['Kloc+'])
                    r['Effort-'] = float(r['Effort-'])
                    r['Defects-'] = float(r['Defects-'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
        return rows

    s01 = type('s01', (), {'load_data': staticmethod(load_data)})


# ----------------------------------------------------------------------
# Helper: train a linear regression model for a single target
# ----------------------------------------------------------------------
def train_regression(rows, target_key: str):
    """Return a sklearn LinearRegression model trained on Kloc+ vs target."""
    X = np.array([r['Kloc+'] for r in rows]).reshape(-1, 1)
    y = np.array([r[target_key] for r in rows])
    model = LinearRegression().fit(X, y)
    return model


# ----------------------------------------------------------------------
# NASA-19: Evaluation of effort model vs historical average
# ----------------------------------------------------------------------
def nasa_19(rows):
    """Print comparison of regression vs simple average on held‑out projects."""
    if len(rows) < 5:
        print("Not enough data for robust evaluation.")
        return

    # Random train/test split
    random.seed(0)
    indices = list(range(len(rows)))
    random.shuffle(indices)
    split = int(0.7 * len(rows))
    train_idx, test_idx = indices[:split], indices[split:]

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    # Train regression for effort
    eff_model = train_regression(train_rows, 'Effort-')

    # Predictions on test set
    pred_eff = eff_model.predict([[r['Kloc+']] for r in test_rows]).flatten()
    actual_eff = np.array([r['Effort-'] for r in test_rows])

    # Simple average estimator
    avg_eff = np.mean([r['Effort-'] for r in train_rows])
    pred_avg = np.full_like(actual_eff, avg_eff)

    # Error metrics
    mae_eff = np.mean(np.abs(pred_eff - actual_eff))
    mae_avg = np.mean(np.abs(pred_avg - actual_eff))
    mse_eff = np.mean((pred_eff - actual_eff) ** 2)
    mse_avg = np.mean((pred_avg - actual_eff) ** 2)

    # Report
    print("== NASA-19 ==")
    print(f"Test set size: {len(test_rows)} projects")
    print(f"Mean absolute error – Regression: {mae_eff:.2f} PM")
    print(f"Mean absolute error – Historical average: {mae_avg:.2f} PM")
    print(f"Mean squared error   – Regression: {mse_eff:.2f} PM²")
    print(f"Mean squared error   – Historical average: {mse_avg:.2f} PM²")
    improvement = mae_avg / mae_eff if mae_eff else float('inf')
    print(f"MAE improvement factor: {improvement:.2f}×")


# ----------------------------------------------------------------------
# NASA-20: Proposal selection vs random selection
# ----------------------------------------------------------------------
def nasa_20(rows):
    """Print comparison of prioritized selection vs random within same budget."""
    # Identify proposals: rows missing either Effort- or Defects- (historical data)
    # For this synthetic exercise we treat all rows as proposals and
    # randomly split them into historical and proposal sets.
    if len(rows) < 10:
        print("Not enough data for proposal evaluation.")
        return

    random.seed(0)
    shuffled = rows.copy()
    random.shuffle(shuffled)

    # 50% historical, 50% proposals
    split = len(shuffled) // 2
    hist_rows = shuffled[:split]
    prop_rows = shuffled[split:]

    if not prop_rows:
        print("No proposals generated for evaluation.")
        return

    # Train models on historical data
    eff_model = train_regression(hist_rows, 'Effort-')
    def_model = train_regression(hist_rows, 'Defects-')

    # For each proposal, compute predicted effort & defects
    preds = []
    for p in prop_rows:
        kloc = float(p['Kloc+'])
        eff_pred = eff_model.predict([[kloc]])[0]
        def_pred = def_model.predict([[kloc]])[0]
        preds.append({'row': p, 'eff': eff_pred, 'def': def_pred})

    # Selection strategy: choose proposals with lowest predicted effort
    preds_sorted = sorted(preds, key=lambda x: x['eff'])
    selection_size = min(10, len(preds))  # budget: evaluate up to 10 proposals
    selected = preds_sorted[:selection_size]
    random_selected = random.sample(preds, selection_size)

    # Compute average predicted effort & defects for both sets
    def avg_stats(proposals):
        effs = [p['eff'] for p in proposals]
        defs = [p['def'] for p in proposals]
        return np.mean(effs), np.mean(defs)

    sel_eff, sel_def = avg_stats(selected)
    rand_eff, rand_def = avg_stats(random_selected)

    # Report
    print("\n== NASA-20 ==")
    print(f"Evaluation budget: {selection_size} proposals")
    print(f"Selected strategy – Avg predicted effort: {sel_eff:.2f} PM")
    print(f"Selected strategy – Avg predicted defects: {sel_def:.2f} cases")
    print(f"Random strategy   – Avg predicted effort: {rand_eff:.2f} PM")
    print(f"Random strategy   – Avg predicted defects: {rand_def:.2f} cases")
    eff_improve = (rand_eff - sel_eff) / rand_eff if rand_eff else float('inf')
    def_improve = (rand_def - sel_def) / rand_def if rand_def else float('inf')
    print(f"Effort reduction vs random: {eff_improve:.2%}")
    print(f"Defect reduction vs random: {def_improve:.2%}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = s01.load_data(csv_path)

    if not rows:
        print("No valid data found.")
        sys.exit(1)

    nasa_19(rows)
    nasa_20(rows)


if __name__ == "__main__":
    main()

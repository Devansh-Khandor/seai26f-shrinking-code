#!/usr/bin/env python3
"""
Skill 02: Estimate effort and defects for a proposed software project
using historical NASA project data.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

# Reuse the data loading helper from skill_01
try:
    import skill_01 as s01
except ImportError:
    # Fallback if skill_01 is not importable for some reason
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

def estimate_effort(rows):
    """
    Train a simple linear regression model Effort- = a * Kloc+ + b
    Return predicted effort for a typical project (mean KLOC),
    its standard error, 95% CI bounds, and R^2.
    """
    X = np.array([r['Kloc+'] for r in rows]).reshape(-1, 1)
    y = np.array([r['Effort-'] for r in rows])

    model = LinearRegression().fit(X, y)

    # Typical project size
    mean_kloc = X.mean()

    # Prediction at mean size
    pred = model.predict([[mean_kloc]])[0]

    # Residuals and MSE
    residuals = y - model.predict(X)
    mse = np.mean(residuals ** 2)
    n = len(rows)

    # Sum of squares of X deviations
    sxx = np.sum((X - X.mean()) ** 2)

    # Standard error of prediction at mean size
    se = np.sqrt(mse * (1 + 1 / n + ((mean_kloc - X.mean()) ** 2) / sxx))

    # 95% prediction interval (normal approx)
    ci_low = pred - 1.96 * se
    ci_high = pred + 1.96 * se

    r2 = model.score(X, y)
    return pred, se, ci_low, ci_high, r2

def estimate_defects(rows):
    """
    Train a simple linear regression model Defects- = a * Kloc+ + b
    Return predicted defects for a typical project (mean KLOC),
    its standard error, 95% CI bounds, and R^2.
    """
    X = np.array([r['Kloc+'] for r in rows]).reshape(-1, 1)
    y = np.array([r['Defects-'] for r in rows])

    model = LinearRegression().fit(X, y)

    mean_kloc = X.mean()
    pred = model.predict([[mean_kloc]])[0]

    residuals = y - model.predict(X)
    mse = np.mean(residuals ** 2)
    n = len(rows)

    sxx = np.sum((X - X.mean()) ** 2)

    se = np.sqrt(mse * (1 + 1 / n + ((mean_kloc - X.mean()) ** 2) / sxx))

    ci_low = pred - 1.96 * se
    ci_high = pred + 1.96 * se

    r2 = model.score(X, y)
    return pred, se, ci_low, ci_high, r2

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = s01.load_data(csv_path)

    if not rows:
        print("No valid data found.")
        sys.exit(1)

    # NASA-03: Effort estimation
    effort_pred, effort_se, effort_ci_low, effort_ci_high, effort_r2 = estimate_effort(rows)

    # NASA-04: Defect estimation
    defect_pred, defect_se, defect_ci_low, defect_ci_high, defect_r2 = estimate_defects(rows)

    # Print results
    print("== NASA-03 ==")
    print(f"Estimated effort for a typical project (size {rows[0]['Kloc+']:.2f} KLOC): "
          f"{effort_pred:.2f} PM ± {effort_se:.2f} PM "
          f"(95% CI: {effort_ci_low:.2f} – {effort_ci_high:.2f})")
    print(f"Model R²: {effort_r2:.2f}")

    print("\n== NASA-04 ==")
    print(f"Estimated defects for a typical project (size {rows[0]['Kloc+']:.2f} KLOC): "
          f"{defect_pred:.2f} ± {defect_se:.2f} defects "
          f"(95% CI: {defect_ci_low:.2f} – {defect_ci_high:.2f})")
    print(f"Model R²: {defect_r2:.2f}")

if __name__ == "__main__":
    main()

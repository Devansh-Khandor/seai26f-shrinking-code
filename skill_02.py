#!/usr/bin/env python3
"""
skill_02.py

This script reads the NASA software project CSV file and answers two
requests:

NASA-03: Estimate the development effort a new project is likely to
require based on its known characteristics (primarily software size,
measured in KLOC).

NASA-04: Estimate the likely number of software defects for a
proposed project and provide a simple measure of uncertainty (the
standard deviation of the residuals of the fitted defect model).

The script reuses helper functions from skill_01.py (read_csv,
extract_numeric_columns, compute_stats) to keep the code DRY.
"""

import sys
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import helpers from skill_01
from skill_01 import read_csv, extract_numeric_columns, compute_stats


def main(csv_path: str):
    rows = read_csv(csv_path)

    # Extract numeric columns of interest
    size = extract_numeric_columns(rows, ["Kloc+"])           # project size (KLOC)
    effort = extract_numeric_columns(rows, ["Effort-"])      # development effort (man-months)
    defects = extract_numeric_columns(rows, ["Defects-"])    # defect count
    year = extract_numeric_columns(rows, ["YearX"])          # project start year

    # Filter out rows with missing critical data
    mask = ~np.isnan(size) & ~np.isnan(effort) & ~np.isnan(defects)
    size = size[mask]
    effort = effort[mask]
    defects = defects[mask]
    year = year[mask]

    # ------------------------------------------------------------------
    # NASA-03: Estimate Development Effort
    # ------------------------------------------------------------------
    print("== NASA-03 ==")

    # Use KLOC (and optionally Year) as predictors.  Here we keep it
    # simple: just KLOC.
    X_effort = size.reshape(-1, 1)
    y_effort = effort

    # Train linear regression model
    reg_effort = LinearRegression()
    reg_effort.fit(X_effort, y_effort)
    effort_pred = reg_effort.predict(X_effort)

    # Compute R² and residual statistics
    r2_effort = r2_score(y_effort, effort_pred)
    std_err_effort = np.sqrt(np.mean((y_effort - effort_pred) ** 2))

    # Output model details
    print(f"Linear regression model: Effort = {reg_effort.coef_[0]:.4f} * KLOC + {reg_effort.intercept_:.4f}")
    print(f"R² of model: {r2_effort:.4f}")
    print(f"Residual standard deviation (man-months): {std_err_effort:.4f}")

    # Predict effort for a project of median KLOC
    median_k = np.nanmedian(size)
    pred_effort_median = reg_effort.predict([[median_k]])[0]
    print(f"Predicted effort for a project of {median_k:.1f} KLOC: {pred_effort_median:.2f} months")

    # ------------------------------------------------------------------
    # NASA-04: Estimate Number of Defects
    # ------------------------------------------------------------------
    print("\n== NASA-04 ==")

    X_defect = size.reshape(-1, 1)
    y_defect = defects

    reg_defect = LinearRegression()
    reg_defect.fit(X_defect, y_defect)
    defect_pred = reg_defect.predict(X_defect)

    r2_defect = r2_score(y_defect, defect_pred)
    std_err_defect = np.sqrt(np.mean((y_defect - defect_pred) ** 2))

    print(f"Linear regression model: Defects = {reg_defect.coef_[0]:.4f} * KLOC + {reg_defect.intercept_:.4f}")
    print(f"R² of model: {r2_defect:.4f}")
    print(f"Residual standard deviation (defects): {std_err_defect:.4f}")

    # Predict defects for a project of median KLOC
    pred_defect_median = reg_defect.predict([[median_k]])[0]
    print(f"Predicted defects for a project of {median_k:.1f} KLOC: {pred_defect_median:.2f} defects")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])

#!/usr/bin/env python3
"""
skill_04.py

This script performs the following NASA requests:

NASA-07: As new project proposals arrive, assess each proposal
        using its available characteristics and alert me when its
        predicted effort or defects appear unusually high.

NASA-08: As additional completed-project records become available,
        update our historical summaries and project estimates so
        that future assessments reflect the latest information.

The script re-uses helper functions from skill_01 and trains
simple linear regression models (size → effort/defects) to
predict missing values.  It then prints out summaries and
high‑alert indications for proposals.
"""

import sys
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import helpers from skill_01
from skill_01 import read_csv, extract_numeric_columns, compute_stats


# ----------------------------------------------------------------------
# Helper: Train models and return predictions for missing values
# ----------------------------------------------------------------------
def train_and_predict(rows):
    """
    Train two linear regression models (size → effort, size → defects)
    using all records that have valid effort and defect values.

    Returns:
        reg_effort: fitted LinearRegression for effort
        reg_defect: fitted LinearRegression for defects
        hist_effort_stats: dict with mean, median, std of historical effort
        hist_defect_stats: dict with mean, median, std of historical defects
    """
    # Extract numeric columns
    size = extract_numeric_columns(rows, ["Kloc+"])
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])

    # Mask for rows with complete data
    mask = ~np.isnan(size) & ~np.isnan(effort) & ~np.isnan(defects)

    size_h = size[mask].reshape(-1, 1)
    effort_h = effort[mask]
    defect_h = defects[mask]

    # Historical statistics
    hist_effort_stats = {
        'mean': np.nanmean(effort_h),
        'median': np.nanmedian(effort_h),
        'std': np.nanstd(effort_h, ddof=0)
    }
    hist_defect_stats = {
        'mean': np.nanmean(defect_h),
        'median': np.nanmedian(defect_h),
        'std': np.nanstd(defect_h, ddof=0)
    }

    # Train effort model
    reg_effort = LinearRegression()
    reg_effort.fit(size_h, effort_h)

    # Train defect model
    reg_defect = LinearRegression()
    reg_defect.fit(size_h, defect_h)

    return reg_effort, reg_defect, hist_effort_stats, hist_defect_stats


# ----------------------------------------------------------------------
# NASA-07: Assess proposals
# ----------------------------------------------------------------------
def assess_proposals(rows, reg_effort, reg_defect,
                     hist_effort_stats, hist_defect_stats):
    """
    Identify proposals (rows with missing Effort- or Defects-),
    predict missing values, and flag unusually high predictions.
    Returns a list of strings describing each proposal.
    """
    size = extract_numeric_columns(rows, ["Kloc+"])
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])
    ids = np.array([int(row['idX']) for row in rows], dtype=int)

    # Identify proposals
    proposal_mask = np.isnan(effort) | np.isnan(defects)

    if not np.any(proposal_mask):
        return ["No proposals found in the data."]

    proposals = []
    high_effort_count = 0
    high_defect_count = 0

    for idx, is_prop in enumerate(proposal_mask):
        if not is_prop:
            continue
        proj_id = ids[idx]
        size_val = size[idx]
        if np.isnan(size_val):
            # Skip proposals lacking size
            continue

        # Predict values
        pred_effort = reg_effort.predict([[size_val]])[0]
        pred_defects = reg_defect.predict([[size_val]])[0]

        # Thresholds: mean + 2*std
        effort_thr = hist_effort_stats['mean'] + 2 * hist_effort_stats['std']
        defect_thr = hist_defect_stats['mean'] + 2 * hist_defect_stats['std']

        eff_alert = pred_effort > effort_thr
        def_alert = pred_defects > defect_thr

        if eff_alert:
            high_effort_count += 1
        if def_alert:
            high_defect_count += 1

        line = (
            f"Proposal ID={proj_id}: Pred Effort={pred_effort:.2f} "
            f"months, Pred Defects={pred_defects:.2f}, "
            f"Alert={'E' if eff_alert else ''}{'D' if def_alert else ''}"
        )
        proposals.append(line)

    # Summary
    summary = (
        f"Total proposals: {len(proposals)}; "
        f"High Effort alerts: {high_effort_count}; "
        f"High Defect alerts: {high_defect_count}"
    )
    return [summary] + proposals


# ----------------------------------------------------------------------
# NASA-08: Update historical summaries and models
# ----------------------------------------------------------------------
def update_historical_summaries(rows):
    """
    Compute updated statistics and retrain models on the full dataset.
    Returns a list of formatted strings.
    """
    # Reuse the training function
    reg_effort, reg_defect, eff_stats, def_stats = train_and_predict(rows)

    # Compute R² and residual std for reporting
    size = extract_numeric_columns(rows, ["Kloc+"])
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])

    mask = ~np.isnan(size) & ~np.isnan(effort) & ~np.isnan(defects)
    size_h = size[mask].reshape(-1, 1)
    effort_h = effort[mask]
    defect_h = defects[mask]

    eff_pred = reg_effort.predict(size_h)
    def_pred = reg_defect.predict(size_h)

    r2_effort = r2_score(effort_h, eff_pred)
    r2_defect = r2_score(defect_h, def_pred)
    std_err_effort = np.sqrt(np.mean((effort_h - eff_pred) ** 2))
    std_err_defect = np.sqrt(np.mean((defect_h - def_pred) ** 2))

    lines = [
        f"Updated typical project size: mean={np.nanmean(size):.2f}, median={np.nanmedian(size):.2f}",
        f"Updated effort model: Effort = {reg_effort.coef_[0]:.4f} * KLOC + {reg_effort.intercept_:.4f}",
        f"R² (effort): {r2_effort:.4f}, residual std: {std_err_effort:.4f}",
        f"Updated defect model: Defects = {reg_defect.coef_[0]:.4f} * KLOC + {reg_defect.intercept_:.4f}",
        f"R² (defects): {r2_defect:.4f}, residual std: {std_err_defect:.4f}",
    ]
    return lines


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str):
    rows = read_csv(csv_path)

    # Train models once
    reg_effort, reg_defect, hist_effort_stats, hist_defect_stats = train_and_predict(rows)

    # ------------------- NASA-07 -------------------
    print("== NASA-07 ==")
    for line in assess_proposals(rows, reg_effort, reg_defect,
                                 hist_effort_stats, hist_defect_stats):
        print(line)

    # ------------------- NASA-08 -------------------
    print("\n== NASA-08 ==")
    for line in update_historical_summaries(rows):
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

#!/usr/bin/env python3
"""
skill_06.py

This script implements the NASA-11 and NASA-12 requests.

NASA-11: When the system estimates the effort required for a proposed project, explain the estimate
          using relevant historical projects and their recorded characteristics.

NASA-12: When two project proposals receive different risk assessments, explain the main
          differences that contributed to those assessments and identify the important tradeoffs.

The implementation re‑uses the helper functions from earlier modules (skill_01) and the
model‑training logic from skill_04.  Results are printed with the required header
format and at least three lines of output for each request.
"""

import sys
import numpy as np

# Import helpers from the earlier skills
from skill_01 import read_csv, extract_numeric_columns, compute_stats


# ----------------------------------------------------------------------
# Helper: Train two simple linear models (size → effort, size → defects)
# ----------------------------------------------------------------------
def train_models(rows):
    """
    Train linear regression models using historical records with
    complete effort and defect data.

    Returns
    -------
    reg_effort : sklearn.linear_model.LinearRegression
    reg_defect : sklearn.linear_model.LinearRegression
    hist_effort_stats : dict with mean, std of historical effort
    hist_defect_stats : dict with mean, std of historical defects
    """
    from sklearn.linear_model import LinearRegression

    size = extract_numeric_columns(rows, ["Kloc+"])
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])

    mask = ~np.isnan(size) & ~np.isnan(effort) & ~np.isnan(defects)
    size_h = size[mask].reshape(-1, 1)
    effort_h = effort[mask]
    defect_h = defects[mask]

    hist_effort_stats = {
        'mean': np.nanmean(effort_h),
        'std': np.nanstd(effort_h, ddof=0)
    }
    hist_defect_stats = {
        'mean': np.nanmean(defect_h),
        'std': np.nanstd(defect_h, ddof=0)
    }

    reg_effort = LinearRegression()
    reg_effort.fit(size_h, effort_h)

    reg_defect = LinearRegression()
    reg_defect.fit(size_h, defect_h)

    return reg_effort, reg_defect, hist_effort_stats, hist_defect_stats


# ----------------------------------------------------------------------
# Helper: Identify proposals (rows with missing Effort- or Defects-)
# ----------------------------------------------------------------------
def get_proposals(rows):
    size = extract_numeric_columns(rows, ["Kloc+"])
    effort = extract_numeric_columns(rows, ["Effort-"])
    defects = extract_numeric_columns(rows, ["Defects-"])
    ids = np.array([int(row["idX"]) for row in rows], dtype=int)

    proposal_mask = np.isnan(effort) | np.isnan(defects)
    proposal_indices = np.where(proposal_mask)[0]
    return proposal_indices, ids, size, effort, defects


# ----------------------------------------------------------------------
# NASA-11: Explain effort estimate using similar historical projects
# ----------------------------------------------------------------------
def explain_estimate(idx, proposal_id, size_val, reg_effort, reg_defect,
                     rows, size_h, effort_h, defect_h):
    """
    Find the three nearest historical projects by size and
    print an explanation of the predicted effort.
    """
    # Predict effort and defects for the proposal
    pred_effort = reg_effort.predict([[size_val]])[0]
    pred_defects = reg_defect.predict([[size_val]])[0]

    # Find distances to all historical projects
    distances = np.abs(size_h - size_val)
    nearest_idx = np.argsort(distances)[:3]

    print(f"  Proposal ID={proposal_id} predicted effort={pred_effort:.2f} months, "
          f"predicted defects={pred_defects:.2f}.  Similar projects:")

    for n, hi in enumerate(nearest_idx, 1):
        hid = int(rows[hi]["idX"])
        hsize = size_h[hi]
        heffort = effort_h[hi]
        hdefects = defect_h[hi]
        hapex = rows[hi]["apex"]
        hcplx = rows[hi]["cplx"]
        print(f"    {n}. ID={hid} Size={hsize:.1f} KLOC, Effort={heffort:.1f} months, "
              f"Defects={hdefects:.1f}, apex={hapex}, cplx={hcplx}")


# ----------------------------------------------------------------------
# NASA-12: Compare two proposals with different risk assessments
# ----------------------------------------------------------------------
def compare_risk(idx1, idx2, ids, size, pred_effort1, pred_defect1,
                 pred_effort2, pred_defect2, hist_effort_stats, hist_defect_stats,
                 rows):
    """
    Print a comparison between two proposals: one with high risk, one with low risk.
    """
    # Compute risk scores: positive means above average, negative below
    eff_z1 = (pred_effort1 - hist_effort_stats['mean']) / hist_effort_stats['std']
    def_z1 = (pred_defect1 - hist_defect_stats['mean']) / hist_defect_stats['std']
    risk1 = eff_z1 + def_z1

    eff_z2 = (pred_effort2 - hist_effort_stats['mean']) / hist_effort_stats['std']
    def_z2 = (pred_defect2 - hist_defect_stats['mean']) / hist_defect_stats['std']
    risk2 = eff_z2 + def_z2

    print("  High‑risk proposal:")
    print(f"    ID={ids[idx1]} Size={size[idx1]:.1f} KLOC, "
          f"Pred Effort={pred_effort1:.1f} months, "
          f"Pred Defects={pred_defect1:.1f}, "
          f"Risk={risk1:.2f}")

    print("  Low‑risk proposal:")
    print(f"    ID={ids[idx2]} Size={size[idx2]:.1f} KLOC, "
          f"Pred Effort={pred_effort2:.1f} months, "
          f"Pred Defects={pred_defect2:.1f}, "
          f"Risk={risk2:.2f}")

    # Highlight main differences
    size_diff = size[idx1] - size[idx2]
    eff_diff = pred_effort1 - pred_effort2
    def_diff = pred_defect1 - pred_defect2

    print("  Main differences:")
    print(f"    Size difference: {size_diff:+.1f} KLOC")
    print(f"    Predicted effort difference: {eff_diff:+.1f} months")
    print(f"    Predicted defects difference: {def_diff:+.1f} defects")
    print("  Trade‑off summary:")
    print(f"    Larger projects tend to increase effort and defect risk; "
          f"higher programmer capability (apex) or lower complexity (cplx) can mitigate risk.")
    # Note: for brevity we do not compute apex/cplx differences here, but the logic can be expanded as needed.


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)

    # Train models on historical data
    reg_effort, reg_defect, hist_effort_stats, hist_defect_stats = train_models(rows)

    # Extract historical data (for similarity search)
    size_h = extract_numeric_columns(rows, ["Kloc+"])
    effort_h = extract_numeric_columns(rows, ["Effort-"])
    defect_h = extract_numeric_columns(rows, ["Defects-"])

    # Identify proposals
    proposal_indices, ids, size, effort, defects = get_proposals(rows)

    if len(proposal_indices) == 0:
        print("== NASA-11 ==")
        print("No proposals with missing effort or defect data.")
        print("\n== NASA-12 ==")
        print("No proposals with missing effort or defect data.")
        return

    # ------------------------------------------------------------------
    # NASA-11: Explain estimate for the first proposal
    # ------------------------------------------------------------------
    print("== NASA-11 ==")
    first_idx = proposal_indices[0]
    propose_id = ids[first_idx]
    size_val = size[first_idx]
    explain_estimate(first_idx, propose_id, size_val,
                     reg_effort, reg_defect, rows,
                     size_h, effort_h, defect_h)

    # ------------------------------------------------------------------
    # NASA-12: Compare two proposals with different risk assessments
    # ------------------------------------------------------------------
    print("\n== NASA-12 ==")

    # Predict effort and defects for all proposals
    pred_effort = np.array([reg_effort.predict([[s]])[0] if not np.isnan(s) else np.nan
                            for s in size])
    pred_defect = np.array([reg_defect.predict([[s]])[0] if not np.isnan(s) else np.nan
                            for s in size])

    # Compute risk scores for each proposal
    risk_scores = []
    for idx in proposal_indices:
        # only compute if predictions are available
        if np.isnan(pred_effort[idx]) or np.isnan(pred_defect[idx]):
            risk_scores.append(-np.inf)
            continue
        eff_z = (pred_effort[idx] - hist_effort_stats['mean']) / hist_effort_stats['std']
        def_z = (pred_defect[idx] - hist_defect_stats['mean']) / hist_defect_stats['std']
        risk_scores.append(eff_z + def_z)
    risk_scores = np.array(risk_scores)

    # Find indices of highest and lowest risk among proposals
    high_risk_idx = proposal_indices[np.argmax(risk_scores)]
    low_risk_idx = proposal_indices[np.argmin(risk_scores)]

    # Ensure we have at least two distinct proposals
    if high_risk_idx == low_risk_idx:
        print("Only one proposal with risk assessment; cannot compare.")
    else:
        compare_risk(high_risk_idx, low_risk_idx,
                     ids, size,
                     pred_effort[high_risk_idx], pred_defect[high_risk_idx],
                     pred_effort[low_risk_idx], pred_defect[low_risk_idx],
                     hist_effort_stats, hist_defect_stats,
                     rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

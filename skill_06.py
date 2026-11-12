#!/usr/bin/env python3
"""
skill_06.py

Provides explanatory analyses for the NASA project dataset.

- NASA-11: When the system estimates the effort required for a proposed project,
           explain the estimate using relevant historical projects and their
           recorded characteristics.
- NASA-12: When two project proposals receive different risk assessments,
           explain the main differences that contributed to those assessments
           and identify the important tradeoffs.

Usage:
    python skill_06.py path/to/nasa93dem.csv
"""

import csv
import sys
import math
import statistics
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """Read CSV and return list of dicts with numeric conversion."""
    data = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = None
                    continue
                val = v.strip()
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data

def _numeric_values(data, column):
    """Return a list of numeric values for a column, excluding None."""
    return [row[column] for row in data if isinstance(row.get(column), (int, float))]

def _linear_regression(x_vals, y_vals):
    """Return (intercept, slope, SE, mean_x, ssx)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx

def _prediction_interval(x, intercept, slope, se, mean_x, ssx, n, conf_factor=2.0):
    """Compute prediction interval for a new observation at x."""
    if ssx == 0 or n <= 2:
        return None, None
    se_pred = se * math.sqrt(1 + 1/n + ((x - mean_x) ** 2) / ssx)
    margin = conf_factor * se_pred
    y_hat = intercept + slope * x
    return y_hat, margin

def _nearest_projects(k_val, data, n=3, exclude_id=None):
    """Return n rows with Kloc+ closest to k_val (excluding exclude_id)."""
    candidates = [row for row in data
                  if isinstance(row.get("Kloc+"), (int, float)) and row.get("idX") != exclude_id]
    sorted_candidates = sorted(candidates,
                               key=lambda r: abs(r["Kloc+"] - k_val))
    return sorted_candidates[:n]

def _risk_category(pred, mean, std):
    """Return 'High' if pred > mean + 2*std, else 'Low'."""
    if mean is None or std is None:
        return 'Unknown'
    return 'High' if pred > mean + 2 * std else 'Low'

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    # Numeric columns needed for regression
    kloc_vals = _numeric_values(data, "Kloc+")
    effort_vals = _numeric_values(data, "Effort-")
    defect_vals = _numeric_values(data, "Defects-")

    # Regression models
    try:
        e_int, e_slope, e_se, e_mean_x, e_ssx = _linear_regression(kloc_vals, effort_vals)
    except ValueError:
        e_int = e_slope = e_se = e_mean_x = e_ssx = None

    try:
        d_int, d_slope, d_se, d_mean_x, d_ssx = _linear_regression(effort_vals, defect_vals)
    except ValueError:
        d_int = d_slope = d_se = d_mean_x = d_ssx = None

    # Statistics for risk thresholds
    effort_mean = statistics.mean(effort_vals) if effort_vals else None
    effort_std = statistics.stdev(effort_vals) if len(effort_vals) > 1 else None
    defect_mean = statistics.mean(defect_vals) if defect_vals else None
    defect_std = statistics.stdev(defect_vals) if len(defect_vals) > 1 else None

    # ----------------------------------------------------------------------- #
    # NASA-11: Explain effort estimate using historical projects
    # ----------------------------------------------------------------------- #
    print("== NASA-11 ==")

    if e_int is None:
        print("Insufficient data to compute effort regression.")
    else:
        # Pick the first project in the dataset as the proposed project
        proposed = data[0]
        proj_id = proposed.get("idX", "Unknown")
        kloc = proposed.get("Kloc+")
        if not isinstance(kloc, (int, float)):
            print(f"Project {proj_id} has no numeric Kloc+; cannot estimate effort.")
        else:
            y_hat, margin = _prediction_interval(
                kloc, e_int, e_slope, e_se, e_mean_x, e_ssx, len(kloc_vals)
            )
            if margin is None:
                lower = upper = y_hat
            else:
                lower, upper = y_hat - margin, y_hat + margin
            print(f"Project ID: {proj_id}")
            print(f"Size (Kloc+): {kloc:.1f}")
            print(f"Estimated Effort = {y_hat:.1f} (+/- {margin:.1f})")
            print(f"95% prediction interval: {lower:.1f} – {upper:.1f}")

            # Find the three most similar historical projects by size
            similar = _nearest_projects(kloc, data, n=3, exclude_id=proj_id)
            if similar:
                print("\nHistorical projects with closest Kloc+:")
                for s in similar:
                    sid = s.get("idX", "N/A")
                    sk = s.get("Kloc+")
                    se = s.get("Effort-")
                    print(f"  ID {sid}: Kloc+={sk:.1f}, Effort-={se:.1f}")
            else:
                print("No historical projects found for comparison.")

    print()

    # ----------------------------------------------------------------------- #
    # NASA-12: Compare two proposals with different risk assessments
    # ----------------------------------------------------------------------- #
    print("== NASA-12 ==")

    if e_int is None or d_int is None:
        print("Insufficient data to evaluate risk differences.")
    else:
        # Choose two projects: one with smallest size, one with largest size
        valid_projects = [r for r in data if isinstance(r.get("Kloc+"), (int, float))]
        if len(valid_projects) < 2:
            print("Not enough projects to compare.")
        else:
            # Sort by Kloc+ size
            sorted_by_size = sorted(valid_projects, key=lambda r: r["Kloc+"])
            proj_a = sorted_by_size[0]      # smallest
            proj_b = sorted_by_size[-1]     # largest

            def analyze(row):
                proj_id = row.get("idX", "N/A")
                k = row.get("Kloc+")
                effort_pred = e_int + e_slope * k
                defect_pred = d_int + d_slope * effort_pred
                eff_risk = _risk_category(effort_pred, effort_mean, effort_std)
                def_risk = _risk_category(defect_pred, defect_mean, defect_std)
                return {
                    "id": proj_id,
                    "kloc": k,
                    "eff_pred": effort_pred,
                    "def_pred": defect_pred,
                    "eff_risk": eff_risk,
                    "def_risk": def_risk
                }

            a = analyze(proj_a)
            b = analyze(proj_b)

            # Print summary for each
            print(f"Proposal A (ID {a['id']}):")
            print(f"  Kloc+ = {a['kloc']:.1f}")
            print(f"  Predicted Effort = {a['eff_pred']:.1f} ({a['eff_risk']} risk)")
            print(f"  Predicted Defects = {a['def_pred']:.1f} ({a['def_risk']} risk)")
            print()
            print(f"Proposal B (ID {b['id']}):")
            print(f"  Kloc+ = {b['kloc']:.1f}")
            print(f"  Predicted Effort = {b['eff_pred']:.1f} ({b['eff_risk']} risk)")
            print(f"  Predicted Defects = {b['def_pred']:.1f} ({b['def_risk']} risk)")
            print()

            # Highlight main differences
            kloc_diff = b['kloc'] - a['kloc']
            effort_diff = b['eff_pred'] - a['eff_pred']
            defect_diff = b['def_pred'] - a['def_pred']

            print("Main differences contributing to risk assessments:")
            print(f"  Size difference (Kloc+): {kloc_diff:.1f} lines")
            print(f"  Effort difference (predicted): {effort_diff:.1f} person‑months")
            print(f"  Defects difference (predicted): {defect_diff:.1f} defects")
            print("\nTrade‑offs:")
            print("  Larger projects require more effort and may incur more defects.")
            print("  However, if the larger project has a highly experienced team or")
            print("  uses advanced tooling, the defect risk can be mitigated.")
            print("  The smaller project has lower effort but may still suffer from defects")
            print("  if the team is inexperienced or the product complexity is high.")
    # End
    sys.exit(0)

if __name__ == "__main__":
    main()

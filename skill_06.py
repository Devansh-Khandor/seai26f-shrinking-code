#!/usr/bin/env python3
"""
skill_06.py - NASA-11 and NASA-12 explanations.

NASA-11: After estimating effort for a hypothetical project (using the median KLOC)
        the script lists the five historical projects that most closely match
        that size and shows their characteristics and actual outcomes.  These
        projects provide context for the estimate.

NASA-12: The user enters two proposal KLOC values (one per line).  The script
        predicts effort and defects for each, compares them, and explains why
        their risk assessments differ, highlighting the main trade‑offs.
"""

import csv
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

# --------------------------------------------------------------------
# Import helpers from existing modules
# --------------------------------------------------------------------
try:
    from skill_01 import load_data
    from skill_02 import linear_regression, estimate_for_median
except Exception as exc:
    raise RuntimeError("Could not import helpers from skill_01/skill_02") from exc

# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def parse_float(value: str) -> float | None:
    """Convert string to float, None on failure."""
    try:
        return float(value.strip())
    except Exception:
        return None

def nearest_projects(projects: List[Dict], target: float, n: int = 5) -> List[Dict]:
    """Return n projects whose KLOC are closest to the target value."""
    return sorted(projects, key=lambda p: abs(float(p["Kloc+"]) - target))[:n]

# --------------------------------------------------------------------
# NASA-11: Explain effort estimate
# --------------------------------------------------------------------
def nasa_11(projects: List[Dict]) -> str:
    # Build regression for effort
    slope_eff, intercept_eff, std_err_eff = linear_regression(projects, "Kloc+", "Effort-")

    # Median KLOC for the estimate
    kloc_vals = [float(p["Kloc+"]) for p in projects]
    kloc_median = statistics.median(kloc_vals)
    effort_est = slope_eff * kloc_median + intercept_eff

    # Find nearest historical projects
    close = nearest_projects(projects, kloc_median, n=5)

    lines = ["== NASA-11 =="]
    lines.append(f"Median project size (used for estimate): {kloc_median:.1f} KLOC")
    lines.append(f"Estimated effort: {effort_est:.1f} person‑months (slope={slope_eff:.2f}, intercept={intercept_eff:.2f}, σ={std_err_eff:.2f})")
    lines.append("\nRelevant historical projects (closest in size):")
    header = f"{'idX':>4}  {'KLOC':>6}  {'Effort':>7}  {'Defects':>8}  {'acap':>4}  {'cplx':>4}"
    lines.append(header)
    for p in close:
        line = f"{p['idX']:>4}  {float(p['Kloc+']):>6.1f}  {float(p['Effort-']):>7.1f}  {float(p['Defects-']):>8.1f}  {p.get('acap',''):>4}  {p.get('cplx',''):>4}"
        lines.append(line)
    return "\n".join(lines)

# --------------------------------------------------------------------
# NASA-12: Compare two proposals
# --------------------------------------------------------------------
def read_two_proposals() -> Tuple[float, float]:
    """Read two KLOC values from stdin."""
    proposals: List[float] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            proposals.append(float(line))
            if len(proposals) == 2:
                break
        except ValueError:
            continue
    if len(proposals) != 2:
        raise ValueError("Exactly two numeric KLOC values must be supplied via stdin.")
    return proposals[0], proposals[1]

def nasa_12(projects: List[Dict], k1: float, k2: float) -> str:
    # Build regression models
    slope_eff, intercept_eff, std_err_eff = linear_regression(projects, "Kloc+", "Effort-")
    slope_def, intercept_def, std_err_def = linear_regression(projects, "Kloc+", "Defects-")

    # Predicted values
    eff1 = slope_eff * k1 + intercept_eff
    def1 = slope_def * k1 + intercept_def
    eff2 = slope_eff * k2 + intercept_eff
    def2 = slope_def * k2 + intercept_def

    # Risk thresholds (median + 2σ)
    median_effort = statistics.median([float(p["Effort-"]) for p in projects])
    median_defects = statistics.median([float(p["Defects-"]) for p in projects])
    eff_risk_thresh = median_effort + 2 * std_err_eff
    def_risk_thresh = median_defects + 2 * std_err_def

    risk1 = "HIGH" if eff1 > eff_risk_thresh or def1 > def_risk_thresh else "NORMAL"
    risk2 = "HIGH" if eff2 > eff_risk_thresh or def2 > def_risk_thresh else "NORMAL"

    # Percentage differences
    eff_diff_pct = ((eff2 - eff1) / eff1 * 100) if eff1 != 0 else 0.0
    def_diff_pct = ((def2 - def1) / def1 * 100) if def1 != 0 else 0.0

    lines = ["== NASA-12 =="]
    lines.append(f"Proposal 1: KLOC={k1:.1f} → Effort={eff1:.1f} ({risk1} risk), Defects={def1:.1f}")
    lines.append(f"Proposal 2: KLOC={k2:.1f} → Effort={eff2:.1f} ({risk2} risk), Defects={def2:.1f}")
    lines.append("\nKey differences:")
    if eff_diff_pct > 0:
        lines.append(f"  Effort of Proposal 2 is {eff_diff_pct:.1f}% higher than Proposal 1.")
    elif eff_diff_pct < 0:
        lines.append(f"  Effort of Proposal 2 is {abs(eff_diff_pct):.1f}% lower than Proposal 1.")
    else:
        lines.append("  Effort estimates are identical.")

    if def_diff_pct > 0:
        lines.append(f"  Defects of Proposal 2 is {def_diff_pct:.1f}% higher than Proposal 1.")
    elif def_diff_pct < 0:
        lines.append(f"  Defects of Proposal 2 is {abs(def_diff_pct):.1f}% lower than Proposal 1.")
    else:
        lines.append("  Defect estimates are identical.")

    lines.append("\nTrade‑offs:")
    if eff_diff_pct > 0 and def_diff_pct < 0:
        lines.append("  Proposal 2 has higher effort but fewer defects – a trade‑off between resource usage and quality.")
    elif eff_diff_pct < 0 and def_diff_pct > 0:
        lines.append("  Proposal 2 has lower effort but more defects – a trade‑off between cost and defect risk.")
    elif eff_diff_pct > 0 and def_diff_pct > 0:
        lines.append("  Proposal 2 carries higher risk on both effort and defects.")
    elif eff_diff_pct < 0 and def_diff_pct < 0:
        lines.append("  Proposal 2 is lower risk across both dimensions.")
    else:
        lines.append("  Both proposals show similar trade‑offs.")
    return "\n".join(lines)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)
    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    # NASA-11
    print(nasa_11(projects))

    # NASA-12
    try:
        k1, k2 = read_two_proposals()
        print("\n" + nasa_12(projects, k1, k2))
    except Exception as exc:
        print(f"\nError reading proposals: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()

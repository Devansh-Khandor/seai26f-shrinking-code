#!/usr/bin/env python3
"""
skill_04.py - NASA-07 and NASA-08 for real‑time assessment and incremental updates.

Usage:
    python skill_04.py <historical-csv-path> [<updates-csv-path>]

If a second argument is provided, the program treats it as a CSV file containing
additional completed project records that should be incorporated into the
historical model before new proposals are evaluated.

After loading the data, the program reads proposal specifications from
standard input, one per line.  Each line should contain at least a numeric
value for KLOC (the primary predictor).  For each proposal it prints the
predicted effort and defect counts and signals if either estimate is
unusually high relative to the historical median plus two standard
errors of the regression model.

The program then prints updated historical summaries that incorporate
the new records (if any).  The output is grouped into two sections:

    == NASA-07 ==   – real‑time proposal assessment
    == NASA-08 ==   – updated historical summaries
"""

import csv
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

# Import helper functions from skill_01 and skill_02
try:
    from skill_01 import load_data, compute_statistics, find_best_worst, compute_medians
    from skill_02 import linear_regression, estimate_for_median
except Exception as exc:
    raise RuntimeError("Could not import helper functions.") from exc


# --------------------------------------------------------------------
# Utility functions for proposal handling
# --------------------------------------------------------------------
def read_proposals() -> List[float]:
    """
    Read proposal KLOC values from standard input.
    Each non‑empty line is interpreted as a float.
    """
    proposals = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            kloc = float(line)
            proposals.append(kloc)
        except ValueError:
            # Skip lines that cannot be parsed
            continue
    return proposals


def predict(projects: List[Dict], proposals: List[float]) -> List[Tuple[float, float, float]]:
    """
    For each proposal KLOC, compute predicted effort and defects
    using regression models derived from the given projects.
    Returns a list of tuples (kloc, effort_est, defects_est).
    """
    # Build regression models
    slope_eff, intercept_eff, std_err_eff = linear_regression(projects, "Kloc+", "Effort-")
    slope_def, intercept_def, std_err_def = linear_regression(projects, "Kloc+", "Defects-")

    # Compute median values for threshold calculation
    median_effort = statistics.median([p["Effort-"] for p in projects])
    median_defects = statistics.median([p["Defects-"] for p in projects])

    predictions = []
    for kloc in proposals:
        effort = slope_eff * kloc + intercept_eff
        defects = slope_def * kloc + intercept_def
        predictions.append((kloc, effort, defects, std_err_eff, std_err_def,
                            median_effort, median_defects))
    return predictions


def format_nasa_07(predictions: List[Tuple[float, float, float, float, float, float, float]]) -> str:
    """
    Generate the NASA‑07 output string.
    """
    lines = ["== NASA-07 =="]
    if not predictions:
        lines.append("No proposals were provided.")
        return "\n".join(lines)

    for idx, (kloc, effort, defects, std_err_eff, std_err_def, med_eff, med_def) in enumerate(predictions, start=1):
        lines.append(f"Proposal {idx}: KLOC={kloc:.1f}")
        lines.append(f"  Predicted effort = {effort:.1f} person‑months")
        lines.append(f"  Predicted defects = {defects:.1f}")
        # Determine if unusually high
        high_eff = effort > med_eff + 2 * std_err_eff
        high_def = defects > med_def + 2 * std_err_def
        if high_eff and high_def:
            lines.append("  ⚠️  ALERT: Both effort and defects unusually high.")
        elif high_eff:
            lines.append("  ⚠️  ALERT: Effort unusually high.")
        elif high_def:
            lines.append("  ⚠️  ALERT: Defects unusually high.")
        else:
            lines.append("  ✅ OK.")
        lines.append("")  # Blank line between proposals
    return "\n".join(lines)


def format_nasa_08(projects: List[Dict], updates_path: Path | None) -> str:
    """
    Generate the NASA‑08 output string, showing updated summaries.
    """
    stats = compute_statistics(projects)
    best, worst = find_best_worst(projects)

    lines = ["== NASA-08 =="]
    if updates_path is not None:
        lines.append(f"Updated historical data with {updates_path.name}.")
    else:
        lines.append("No updates applied; using original historical data.")
    lines.append("")

    lines.append(f"Typical project size (KLOC): mean={stats['Kloc+'][0]:.1f}, "
                 f"min={stats['Kloc+'][1]:.1f}, max={stats['Kloc+'][2]:.1f}")
    lines.append(f"Typical development effort: mean={stats['Effort-'][0]:.1f} person‑months, "
                 f"min={stats['Effort-'][1]:.1f}, max={stats['Effort-'][2]:.1f}")
    lines.append(f"Typical defect counts: mean={stats['Defects-'][0]:.1f}, "
                 f"min={stats['Defects-'][1]:.1f}, max={stats['Defects-'][2]:.1f}")
    lines.append(f"Best outcome (lowest effort): id={best['idX']}, effort={best['Effort-']:.1f}, "
                 f"defects={best['Defects-']:.1f}")
    lines.append(f"Worst outcome (highest effort): id={worst['idX']}, effort={worst['Effort-']:.1f}, "
                 f"defects={worst['Defects-']:.1f}")
    lines.append("")

    # Updated regression models
    slope_eff, intercept_eff, std_err_eff = linear_regression(projects, "Kloc+", "Effort-")
    slope_def, intercept_def, std_err_def = linear_regression(projects, "Kloc+", "Defects-")
    kloc_med, effort_est = estimate_for_median(projects, slope_eff, intercept_eff)
    _, defect_est = estimate_for_median(projects, slope_def, intercept_def)

    lines.append("Updated regression models:")
    lines.append(f"  Effort ~ KLOC: slope={slope_eff:.4f}, intercept={intercept_eff:.2f}, "
                 f"σ={std_err_eff:.2f}")
    lines.append(f"  Defects ~ KLOC: slope={slope_def:.4f}, intercept={intercept_def:.2f}, "
                 f"σ={std_err_def:.2f}")
    lines.append(f"Median project size used for estimates: {kloc_med:.1f} KLOC")
    lines.append(f"Estimated effort at median size: {effort_est:.1f} person‑months ±{std_err_eff:.1f}")
    lines.append(f"Estimated defects at median size: {defect_est:.1f} ±{std_err_def:.1f}")

    return "\n".join(lines)


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python skill_04.py <historical-csv-path> [<updates-csv-path>]")
        sys.exit(1)

    historical_path = Path(sys.argv[1])
    updates_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None

    if not historical_path.is_file():
        print(f"Error: File not found: {historical_path}")
        sys.exit(1)

    # Load base historical data
    projects = load_data(historical_path)

    # If an updates file is provided, load and append it
    if updates_path:
        if not updates_path.is_file():
            print(f"Error: Updates file not found: {updates_path}")
            sys.exit(1)
        updated_projects = load_data(updates_path)
        projects.extend(updated_projects)

    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    # Read proposals from stdin
    proposals = read_proposals()

    # Generate NASA-07 output
    predictions = predict(projects, proposals)
    print(format_nasa_07(predictions))

    # Generate NASA-08 output
    print("\n" + format_nasa_08(projects, updates_path))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>

This script implements the new NASA-07 and NASA-08 queries:

- NASA-07:  As new project proposals arrive, the script
            predicts effort and defects for each proposal
            using the regression models derived from the
            historical data and alerts when the predictions
            are unusually high.

- NASA-08:  After new completed‑project records are added
            to the dataset, the script re‑computes the
            historical summaries and regression models so
            that future predictions reflect the latest data.

The implementation re‑uses the parsing, regression and
statistical helpers from the previous skill modules.
"""

import csv
import sys
import statistics
from typing import List, Tuple

# --------------------------------------------------------------------
# Import helpers from the previous skill modules
# --------------------------------------------------------------------

# The read_csv, linear_regression, predict and compute_basic_stats
# helpers are defined in skill_01.py and skill_02.py.  Import them
# directly to keep this module self‑contained.
try:
    # These imports work when the skill files are in the same directory.
    from skill_01 import compute_basic_stats  # noqa: F401
    from skill_02 import read_csv, linear_regression, predict  # noqa: F401
except Exception as e:
    # Fallback to re‑implement minimal versions if imports fail.
    # This should never happen in normal usage.
    raise RuntimeError(f"Failed to import helpers from skill_01/02: {e}")

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std_dev). If len(values)<=1, std_dev=0."""
    if not values:
        return 0.0, 0.0
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_val, std_val

# --------------------------------------------------------------------
# NASA-07 – Proposal assessment
# --------------------------------------------------------------------

def nasa_07(rows: List[Tuple]) -> None:
    """
    For each new proposal entered by the user, predict effort and defect
    counts using the regression models built from the historical data.
    Flag proposals whose predictions exceed the mean + 2*std of the
    historical predicted distributions.
    """
    print("\n== NASA-07 ==")

    # Build regression models for effort and defects
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    defects = [r.defects for r in rows]

    a_eff, b_eff = linear_regression(klocs, efforts)
    a_def, b_def = linear_regression(klocs, defects)

    # Compute historical predicted distributions
    pred_efforts = [predict(k, a_eff, b_eff) for k in klocs]
    pred_defects = [predict(k, a_def, b_def) for k in klocs]

    mean_eff, std_eff = compute_mean_std(pred_efforts)
    mean_def, std_def = compute_mean_std(pred_defects)

    # Define high‑risk thresholds (mean + 2 * std)
    eff_threshold = mean_eff + 2 * std_eff
    def_threshold = mean_def + 2 * std_def

    # Ask the user how many proposals to evaluate
    try:
        n_str = input("\nEnter number of new proposals to evaluate: ").strip()
        n_props = int(n_str)
    except Exception:
        print("  (Invalid input – defaulting to 1 proposal.)")
        n_props = 1

    proposals: List[Tuple[float, float, float]] = []  # (size, eff, def)

    for i in range(1, n_props + 1):
        try:
            sz_str = input(f"  Proposal #{i} – enter size in KLOC: ").strip()
            size = float(sz_str)
        except Exception:
            print("  (Invalid input – using median KLOC from data.)")
            size = statistics.median(klocs)

        pred_eff = predict(size, a_eff, b_eff)
        pred_def = predict(size, a_def, b_def)

        proposals.append((size, pred_eff, pred_def))

    # Output results
    header = f"{'Prop':>5} {'Size(KLOC)':>10} {'Effort':>8} {'Defects':>9} {'Alert':>6}"
    print("\n" + header)
    for idx, (size, eff, defe) in enumerate(proposals, start=1):
        alert = ""
        if eff > eff_threshold or defe > def_threshold:
            alert = "YES"
        line = f"{idx:5d} {size:10.1f} {eff:8.1f} {defe:9.1f} {alert:>6}"
        print(line)

# --------------------------------------------------------------------
# NASA-08 – Update historical summaries
# --------------------------------------------------------------------

def nasa_08(rows: List[Tuple]) -> None:
    """
    Re‑compute and display updated summary statistics for the
    historical dataset.  This demonstrates that the analysis
    automatically incorporates any new completed‑project records
    that have been added to the CSV file before the script runs.
    """
    print("\n== NASA-08 ==")

    # Summary statistics
    size_stats = compute_basic_stats([r.kloc for r in rows])
    effort_stats = compute_basic_stats([r.effort for r in rows])
    defect_stats = compute_basic_stats([r.defects for r in rows])

    print(f"Number of projects in history: {len(rows)}")
    print(f"Typical size (KLOC): mean={size_stats['mean']:.2f}, median={size_stats['median']:.2f}")
    print(f"Typical effort (person‑months): mean={effort_stats['mean']:.2f}, median={effort_stats['median']:.2f}")
    print(f"Typical defect count: mean={defect_stats['mean']:.2f}, median={defect_stats['median']:.2f}")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main(csv_path: str) -> None:
    # Load the dataset
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not rows:
        print("No valid data found.")
        return

    # Run the NASA-07 query
    nasa_07(rows)

    # Run the NASA-08 query
    nasa_08(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

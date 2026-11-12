#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv_path>

This script implements the new NASA-11 and NASA-12 queries:

- NASA-11 – Explain the effort estimate for a proposed project by citing
            the most similar historical projects and their recorded
            characteristics.

- NASA-12 – When two project proposals receive different risk assessments,
            explain the main differences that contributed to those assessments
            and identify the important trade‑offs.

The implementation re‑uses parsing, regression, and statistical helpers
from the previous skill modules.
"""

import csv
import sys
import statistics
from typing import List, Tuple
from collections import namedtuple

# --------------------------------------------------------------------
# Import helpers from the previous skill modules
# --------------------------------------------------------------------
try:
    # Parsing and regression helpers
    from skill_02 import read_csv, linear_regression, predict  # noqa: F401
    # Basic stats helper
    from skill_04 import compute_basic_stats  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import helpers from skill_02/04: {e}")

# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------
# The dataset contains categorical ratings for programmer capability
# (pmat) and complexity (cplx).  We keep these as strings in the
# Row, but create numeric encodings for distance calculations.
Row = namedtuple(
    "Row",
    [
        "id",
        "center",
        "year",
        "pmat",
        "cplx",
        "kloc",
        "effort",
        "defects",
    ],
)

# --------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------
def parse_row(raw: dict) -> Row | None:
    try:
        id_ = int(raw["idX"])
        center = int(raw["centerX"])
        year = int(raw["YearX"])
        pmat = raw["pmat"].strip()
        cplx = raw["cplx"].strip()
        kloc = float(raw["Kloc+"])
        effort = float(raw["Effort-"])
        defects = float(raw["Defects-"])
    except (ValueError, KeyError):
        return None
    return Row(id_, center, year, pmat, cplx, kloc, effort, defects)


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
# Mapping of categorical ratings to numeric values for distance
PMAT_MAP = {"l": 1.0, "m": 2.0, "h": 3.0, "vh": 4.0}
CPLX_MAP = {"l": 1.0, "m": 2.0, "h": 3.0, "vh": 4.0}

def encode_pmat(p: str) -> float:
    return PMAT_MAP.get(p.lower(), 0.0)

def encode_cplx(c: str) -> float:
    return CPLX_MAP.get(c.lower(), 0.0)

def distance(proposed: Tuple[float, float, float], r: Row) -> float:
    """Euclidean distance in (kloc, pmat, cplx) space."""
    return ((proposed[0] - r.kloc) ** 2 + (proposed[1] - encode_pmat(r.pmat)) ** 2 + (proposed[2] - encode_cplx(r.cplx)) ** 2) ** 0.5

def top_similar_projects(proposed: Tuple[float, float, float], rows: List[Row], n: int = 3) -> List[Tuple[int, Row, float]]:
    """Return the top `n` most similar historical projects."""
    scored = [(r, distance(proposed, r)) for r in rows]
    scored.sort(key=lambda x: x[1])
    return [(i, r, d) for i, (r, d) in enumerate(scored[:n], start=1)]

# --------------------------------------------------------------------
# NASA-11 – Explain effort estimate
# --------------------------------------------------------------------
def nasa_11(rows: List[Row]) -> None:
    """Explain the effort estimate for a proposed project."""
    print("\n== NASA-11 ==")

    # Prompt user for proposal characteristics
    try:
        sz_str = input("Enter proposed project size (KLOC): ").strip()
        proj_kloc = float(sz_str)
    except Exception:
        print("  (Invalid input – using median KLOC from data)")
        proj_kloc = statistics.median([r.kloc for r in rows])

    # For simplicity, use the most common pmat and cplx values from the data
    # to fill missing ratings.  Prompt user but allow defaults.
    pmats = [r.pmat for r in rows]
    cplxs = [r.cplx for r in rows]
    default_pmat = max(set(pmats), key=pmats.count)
    default_cplx = max(set(cplxs), key=cplxs.count)

    pmat_str = input(f"Enter programmer capability rating [{default_pmat}]: ").strip()
    if not pmat_str:
        pmat_str = default_pmat
    cplx_str = input(f"Enter complexity rating [{default_cplx}]: ").strip()
    if not cplx_str:
        cplx_str = default_cplx

    # Regression model for effort vs KLOC
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    a, b = linear_regression(klocs, efforts)
    predicted_effort = predict(proj_kloc, a, b)

    # Similar projects
    top3 = top_similar_projects((proj_kloc, encode_pmat(pmat_str), encode_cplx(cplx_str)), rows)

    # Output
    print(f"\nProject size: {proj_kloc:.1f} KLOC")
    print(f"Programmer capability: {pmat_str}")
    print(f"Complexity: {cplx_str}")
    print(f"\nEstimated effort: {predicted_effort:.1f} person‑months")

    print("\nMost similar historical projects:")
    header = f"{'Rank':>4} {'idX':>4} {'KLOC':>7} {'PMat':>5} {'Cplx':>5} {'Effort':>8}"
    print(header)
    for rank, r, dist in top3:
        print(f"{rank:4d} {r.id:4d} {r.kloc:7.1f} {r.pmat:>5} {r.cplx:>5} {r.effort:8.1f}")

# --------------------------------------------------------------------
# NASA-12 – Compare two proposals
# --------------------------------------------------------------------
def nasa_12(rows: List[Row]) -> None:
    """Explain differences between two proposals and their risk assessments."""
    print("\n== NASA-12 ==")

    # Build regression models once
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    defects = [r.defects for r in rows]
    a_eff, b_eff = linear_regression(klocs, efforts)
    a_def, b_def = linear_regression(klocs, defects)

    # Historical predicted distributions for thresholds
    pred_efforts = [predict(k, a_eff, b_eff) for k in klocs]
    pred_defects = [predict(k, a_def, b_def) for k in klocs]
    eff_stats = compute_basic_stats(pred_efforts)
    def_stats = compute_basic_stats(pred_defects)
    eff_thresh = eff_stats["mean"] + 2 * (eff_stats["max"] - eff_stats["mean"])
    def_thresh = def_stats["mean"] + 2 * (def_stats["max"] - def_stats["mean"])

    # Collect two proposals
    proposals = []
    for i in range(1, 3):
        print(f"\nProposal #{i}")
        try:
            sz_str = input("  Enter size (KLOC): ").strip()
            size = float(sz_str)
        except Exception:
            size = statistics.median(klocs)
        pmat_str = input("  Enter programmer capability rating (l/m/h/vh): ").strip() or "m"
        cplx_str = input("  Enter complexity rating (l/m/h/vh): ").strip() or "m"

        # Predictions
        eff = predict(size, a_eff, b_eff)
        defe = predict(size, a_def, b_def)
        risk = "HIGH" if eff > eff_thresh or defe > def_thresh else "LOW"
        proposals.append((size, pmat_str, cplx_str, eff, defe, risk))

    # Explanation
    print("\nRisk assessments:")
    for idx, (_, _, _, eff, defe, risk) in enumerate(proposals, start=1):
        print(f"  Proposal {idx}: {risk} (Effort={eff:.1f} PM, Defects={defe:.1f})")

    # Differences
    size_diff = abs(proposals[0][0] - proposals[1][0])
    pmat_diff = proposals[0][1] != proposals[1][1]
    cplx_diff = proposals[0][2] != proposals[1][2]
    eff_diff = proposals[0][3] - proposals[1][3]
    defe_diff = proposals[0][4] - proposals[1][4]

    print("\nKey differences contributing to risk assessment:")
    print(f"  Size difference: {size_diff:.1f} KLOC")
    print(f"  Programmer capability rating difference: {'Yes' if pmat_diff else 'No'}")
    print(f"  Complexity rating difference: {'Yes' if cplx_diff else 'No'}")
    print(f"  Effort difference: {eff_diff:.1f} person‑months")
    print(f"  Defect difference: {defe_diff:.1f}")

    print("\nTrade‑offs:")
    print("  Larger size typically increases both effort and defect expectations.")
    print("  Higher programmer capability (vh) can offset effort per KLOC but may still lead to higher absolute effort.")
    print("  Lower complexity (l) tends to reduce both effort and defect counts.")

# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.")
        return
    nasa_11(rows)
    nasa_12(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

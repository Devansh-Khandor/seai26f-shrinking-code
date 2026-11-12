#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <csv_path>

This script implements the new NASA-13 and NASA-14 queries:

  NASA-13 – Identify historically promising combinations of project
            characteristics that deliver more software with lower
            development effort and fewer defects.

  NASA-14 – Recommend which new proposals should be evaluated first
            to discover promising alternatives within a limited
            evaluation budget.
"""

import csv
import sys
import statistics
from typing import List, Tuple
from collections import namedtuple

# --------------------------------------------------------------------
# Import helpers from earlier skills
# --------------------------------------------------------------------
try:
    # Parsing, regression helpers
    from skill_02 import read_csv, linear_regression, predict  # noqa: F401
    # Row type & parser (pmat, cplx)
    from skill_03 import Row as Row3, parse_row as parse_row_03  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import helpers from previous modules: {e}")

# --------------------------------------------------------------------
# Re‑use the Row type from skill_03 (pmat, cplx)
# --------------------------------------------------------------------
Row = Row3

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def group_by_combo(rows: List[Row]) -> dict:
    """
    Group rows by the (pmat, cplx) combination and compute aggregated
    statistics: total KLOC, total effort, total defects, count.
    """
    groups: dict[Tuple[str, str], dict] = {}
    for r in rows:
        key = (r.pmat, r.cplx)
        rec = groups.setdefault(key, {"kloc": 0.0, "effort": 0.0, "defects": 0.0, "count": 0})
        rec["kloc"] += r.kloc
        rec["effort"] += r.effort
        rec["defects"] += r.defects
        rec["count"] += 1
    return groups


def nasa_13(rows: List[Row]) -> None:
    """Identify historically promising (pmat, cplx) combinations."""
    print("\n== NASA-13 ==")

    groups = group_by_combo(rows)
    if not groups:
        print("  No data available.")
        return

    combo_stats = []
    for (pmat, cplx), rec in groups.items():
        if rec["kloc"] == 0:
            continue
        eff_per_kloc = rec["effort"] / rec["kloc"]
        def_per_kloc = rec["defects"] / rec["kloc"]
        combo_stats.append(
            (pmat, cplx, rec["count"], rec["kloc"], eff_per_kloc, def_per_kloc)
        )

    # Sort by effort per KLOC then defect per KLOC
    combo_stats.sort(key=lambda x: (x[4], x[5]))

    print(f"Top 3 promising (pmat, cplx) combinations (lower values are better):")
    print(f"{'PMat':>4} {'Cplx':>5} {'Count':>6} {'Total KLOC':>10} {'Eff/kloc':>9} {'Def/kloc':>9}")
    for idx, (pmat, cplx, count, kloc, eff_k, def_k) in enumerate(combo_stats[:3], start=1):
        print(
            f"{pmat:>4} {cplx:>5} {count:6d} {kloc:10.1f} {eff_k:9.3f} {def_k:9.3f}"
        )
    print("\nInterpretation:")
    print(
        "  These combinations have historically required the least effort per unit "
        "size and produced the fewest defects per unit size."
    )


def nasa_14(rows: List[Row]) -> None:
    """Recommend proposals to evaluate first given a limited budget."""
    print("\n== NASA-14 ==")

    # Build regression models from historical data (effort vs KLOC, defects vs KLOC)
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    defects = [r.defects for r in rows]
    try:
        a_eff, b_eff = linear_regression(klocs, efforts)
        a_def, b_def = linear_regression(klocs, defects)
    except ValueError as e:
        print(f"  (Regression failed: {e})")
        return

    # Ask user for evaluation budget (# proposals)
    try:
        budget_str = input("\nEnter the number of proposals you can evaluate: ").strip()
        budget = int(budget_str)
    except Exception:
        print("  (Invalid input – defaulting to 3 proposals.)")
        budget = 3

    proposals: List[Tuple[int, float, str, str, float, float, float]] = []
    print("\nEnter proposal details:")
    for i in range(1, budget + 1):
        print(f"\nProposal #{i}")
        try:
            sz_str = input("  Enter size (KLOC): ").strip()
            size = float(sz_str)
        except Exception:
            size = statistics.median(klocs)
            print(f"  (Using median size {size:.1f} KLOC)")

        pmat = input("  Enter programmer capability rating (l/m/h/vh): ").strip() or "m"
        cplx = input("  Enter complexity rating (l/m/h/vh): ").strip() or "m"

        # Predictions
        eff = predict(size, a_eff, b_eff)
        defe = predict(size, a_def, b_def)

        # Simple promise score: lower effort/kloc + defect/kloc
        score = (eff / size) + (defe / size)

        proposals.append((i, size, pmat, cplx, eff, defe, score))

    # Sort proposals by promise score
    proposals.sort(key=lambda x: x[6])

    print("\nProposals sorted by predicted promise (lower is better):")
    header = (
        f"{'Rank':>4} {'KLOC':>6} {'PMat':>5} {'Cplx':>5} "
        f"{'Effort':>8} {'Defects':>8} {'Score':>8}"
    )
    print(header)
    for rank, (idx, size, pmat, cplx, eff, defe, score) in enumerate(proposals, start=1):
        print(
            f"{rank:4d} {size:6.1f} {pmat:>5} {cplx:>5} "
            f"{eff:8.1f} {defe:8.1f} {score:8.3f}"
        )

    print("\nRecommendation:")
    print(
        "  Evaluate the proposals at the top of the list first. "
        "They are predicted to deliver lower effort per size and fewer defects per size "
        "for the same amount of effort invested."
    )


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
def main(csv_path: str) -> None:
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not rows:
        print("No valid data found.")
        return

    nasa_13(rows)
    nasa_14(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

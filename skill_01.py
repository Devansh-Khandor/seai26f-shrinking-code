#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>

This script implements the NASA-01 and NASA-02 queries on the
NASA93DEM dataset.
"""

import csv
import sys
import statistics
from collections import namedtuple

# --------------------------------------------------------------------
# Data structures and parsing helpers
# --------------------------------------------------------------------

# Define the fields we care about
NUMERIC_FIELDS = ["idX", "centerX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"]

Row = namedtuple(
    "Row",
    [
        "id",
        "center",
        "year",
        "kloc",
        "effort",
        "defects",
        "months",
    ],
)


def parse_row(raw: dict) -> Row | None:
    """Parse a raw dict from DictReader into a Row. Return None if any field
    fails to parse as a number."""
    try:
        id_ = int(raw["idX"])
        center = int(raw["centerX"])
        year = int(raw["YearX"])
        kloc = float(raw["Kloc+"])
        effort = float(raw["Effort-"])
        defects = float(raw["Defects-"])
        months = float(raw["MonthsX"])
    except (ValueError, KeyError):
        # Skip rows that don't have the required numeric columns
        return None
    return Row(id_, center, year, kloc, effort, defects, months)


def read_csv(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = parse_row(raw)
            if row:
                rows.append(row)
    return rows


# --------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------

def compute_basic_stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def quantile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0 <= q <= 1)."""
    # statistics.quantiles gives quartiles; for arbitrary q use linear interpolation
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    pos = q * (n - 1)
    lower = int(pos)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[lower]
    weight = pos - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------

def main(csv_path: str):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.")
        return

    # Gather raw metrics
    klocs = [r.kloc for r in rows]
    efforts = [r.effort for r in rows]
    defects = [r.defects for r in rows]

    # Compute basic statistics
    size_stats = compute_basic_stats(klocs)
    effort_stats = compute_basic_stats(efforts)
    defect_stats = compute_basic_stats(defects)

    # Compute defect and effort rates per KLOC
    rates = []
    for r in rows:
        if r.kloc <= 0:
            continue
        def_rate = r.defects / r.kloc
        eff_rate = r.effort / r.kloc
        rates.append(
            {
                "row": r,
                "def_rate": def_rate,
                "eff_rate": eff_rate,
            }
        )

    # Determine best/worst outcomes
    best_defect = min(rates, key=lambda x: x["def_rate"])
    worst_defect = max(rates, key=lambda x: x["def_rate"])
    best_effort = min(rates, key=lambda x: x["eff_rate"])
    worst_effort = max(rates, key=lambda x: x["eff_rate"])

    # --- NASA-01 ---
    print("== NASA-01 ==")
    print(
        f"Typical project size (KLOC): mean={size_stats['mean']:.2f}, median={size_stats['median']:.2f}"
    )
    print(
        f"Typical development effort (person-months): mean={effort_stats['mean']:.2f}, median={effort_stats['median']:.2f}"
    )
    print(
        f"Typical defect count: mean={defect_stats['mean']:.2f}, median={defect_stats['median']:.2f}"
    )
    print(
        f"Best outcome – lowest defects/KLOC: Project {best_defect['row'].id} with {best_defect['def_rate']:.2f} defects per KLOC"
    )
    print(
        f"Best outcome – lowest effort/KLOC: Project {best_effort['row'].id} with {best_effort['eff_rate']:.2f} effort per KLOC"
    )
    print(
        f"Worst outcome – highest defects/KLOC: Project {worst_defect['row'].id} with {worst_defect['def_rate']:.2f} defects per KLOC"
    )
    print(
        f"Worst outcome – highest effort/KLOC: Project {worst_effort['row'].id} with {worst_effort['eff_rate']:.2f} effort per KLOC"
    )

    # --- NASA-02 ---
    print("\n== NASA-02 ==")
    print(
        "Projects with unusually favorable size-effort-defect combination (below 25th percentile for both metrics):"
    )
    # Determine thresholds
    def_rate_values = [r["def_rate"] for r in rates]
    eff_rate_values = [r["eff_rate"] for r in rates]
    def_thresh = quantile(def_rate_values, 0.25)
    eff_thresh = quantile(eff_rate_values, 0.25)

    favorable = [
        r
        for r in rates
        if r["def_rate"] <= def_thresh and r["eff_rate"] <= eff_thresh
    ]

    # Header
    header = f"{'idX':>4} {'Size(KLOC)':>10} {'Effort':>8} {'Defects':>8} {'Def/KLOC':>9} {'Eff/KLOC':>9}"
    print(header)
    for r in favorable:
        row = r["row"]
        print(
            f"{row.id:4d} {row.kloc:10.1f} {row.effort:8.1f} {row.defects:8.1f} {r['def_rate']:9.2f} {r['eff_rate']:9.2f}"
        )
    if not favorable:
        print("  (No projects meet the criteria.)")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

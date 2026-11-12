#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>

This script performs NASA-05 and NASA-06 queries on the
NASA93DEM dataset.

- NASA-05 compares projects with different programmer capability
  ratings (pmat) and reports whether their development‑effort
  differences are meaningful.

- NASA-06 compares projects with different complexity ratings
  (cplx) and reports whether their defect outcomes differ
  reliably or whether the available evidence is insufficient.
"""

import csv
import sys
import statistics
from collections import namedtuple
from typing import List, Dict, Tuple

# --------------------------------------------------------------------
# Data structures and CSV parsing
# --------------------------------------------------------------------

# We need to capture additional attributes (pmat, cplx) not used in
# earlier skills, so we define a new Row type.
Row = namedtuple(
    "Row",
    [
        "id",
        "center",
        "year",
        "pmat",   # Programmer capability rating
        "cplx",   # Complexity rating
        "kloc",
        "effort",
        "defects",
    ],
)

def parse_row(raw: dict) -> Row | None:
    """Parse a raw dict from DictReader into a Row."""
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
        # Skip rows that don't have the required columns
        return None
    return Row(id_, center, year, pmat, cplx, kloc, effort, defects)

def read_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = parse_row(raw)
            if row:
                rows.append(row)
    return rows

# --------------------------------------------------------------------
# Helper statistical functions
# --------------------------------------------------------------------

def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std_dev). If len(values)<=1, std_dev=0."""
    if not values:
        return (0.0, 0.0)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_val, std_val

def t_statistic(mean1: float, mean2: float, s1: float, n1: int,
                s2: float, n2: int) -> float:
    """Compute two‑sample t‑statistic assuming unequal variances."""
    se = (s1 ** 2 / n1) + (s2 ** 2 / n2)
    if se == 0:
        return 0.0
    return abs(mean1 - mean2) / (se ** 0.5)

# --------------------------------------------------------------------
# NASA-05 – Programmer capability comparison
# --------------------------------------------------------------------

def nasa_05(rows: List[Row]) -> None:
    """Compare effort across programmer capability ratings."""
    # Group rows by pmat rating
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(r.pmat, []).append(r.effort)

    # Compute statistics per group
    stats: Dict[str, Tuple[float, float, int]] = {}
    for rating, efforts in groups.items():
        mean, std = compute_mean_std(efforts)
        stats[rating] = (mean, std, len(efforts))

    # Prepare output
    print("\n== NASA-05 ==")
    if len(stats) < 2:
        print("  (Insufficient data: only one programmer capability rating present.)")
        return

    # Print header
    print(f"{'Rating':>6} {'Count':>6} {'Mean Effort':>12} {'Std Dev':>10}")
    for rating, (mean, std, count) in sorted(stats.items()):
        print(f"{rating:>6} {count:6d} {mean:12.2f} {std:10.2f}")

    # Compare pairwise differences
    ratings = list(stats.keys())
    print("\nPairwise effort differences:")
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            r1, r2 = ratings[i], ratings[j]
            mean1, std1, n1 = stats[r1]
            mean2, std2, n2 = stats[r2]
            t_val = t_statistic(mean1, mean2, std1, n1, std2, n2)
            diff = mean1 - mean2
            print(f"  {r1} vs {r2}: Δ={diff: .2f} person‑months, t={t_val:.2f}")

    # Interpretation
    print("\nInterpretation:")
    print("  A t‑statistic above ~2.0 (rough threshold for 95% confidence with moderate df)")
    print("  suggests that the mean effort difference is statistically meaningful.")
    print("  The pairs listed above show the magnitude of the differences and their t‑values.")

# --------------------------------------------------------------------
# NASA-06 – Complexity rating comparison
# --------------------------------------------------------------------

def nasa_06(rows: List[Row]) -> None:
    """Compare defect outcomes across complexity ratings."""
    # Group rows by cplx rating
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(r.cplx, []).append(r.defects)

    # Compute statistics per group
    stats: Dict[str, Tuple[float, float, int]] = {}
    for rating, defects in groups.items():
        mean, std = compute_mean_std(defects)
        stats[rating] = (mean, std, len(defects))

    # Prepare output
    print("\n== NASA-06 ==")
    if len(stats) < 2:
        print("  (Insufficient data: only one complexity rating present.)")
        return

    # Print header
    print(f"{'Rating':>6} {'Count':>6} {'Mean Defects':>13} {'Std Dev':>10}")
    for rating, (mean, std, count) in sorted(stats.items()):
        print(f"{rating:>6} {count:6d} {mean:13.2f} {std:10.2f}")

    # Compare pairwise differences
    ratings = list(stats.keys())
    print("\nPairwise defect differences:")
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            r1, r2 = ratings[i], ratings[j]
            mean1, std1, n1 = stats[r1]
            mean2, std2, n2 = stats[r2]
            t_val = t_statistic(mean1, mean2, std1, n1, std2, n2)
            diff = mean1 - mean2
            print(f"  {r1} vs {r2}: Δ={diff: .2f} defects, t={t_val:.2f}")

    # Interpretation
    print("\nInterpretation:")
    print("  A t‑statistic above ~2.0 indicates a reliable difference in defect outcomes.")
    print("  If all pairwise t‑values are below that threshold, the evidence is insufficient")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main(csv_path: str) -> None:
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.")
        return

    nasa_05(rows)
    nasa_06(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

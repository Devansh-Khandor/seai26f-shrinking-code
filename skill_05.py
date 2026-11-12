#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv_path>

This script implements the new NASA-09 and NASA-10 queries:

  NASA-09 – For projects that required unusually high development effort,
            identify which recorded project characteristics are most
            strongly associated with those outcomes.

  NASA-10 – Examine projects with unusually high defect counts and
            identify the characteristics that distinguish them from
            projects with better quality outcomes.
"""

import csv
import sys
import statistics
from typing import List, Tuple, Dict, Any

# --------------------------------------------------------------------
# Helpers from earlier skills
# --------------------------------------------------------------------

try:
    # Compute mean and standard deviation helpers
    from skill_04 import compute_mean_std  # noqa: F401
    # Quantile function for thresholding
    from skill_01 import quantile  # noqa: F401
except Exception as e:
    raise RuntimeError(f"Failed to import helpers from skill_01/04: {e}")

# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------

Row = tuple[
    int,      # id
    int,      # center
    int,      # year
    str,      # pmat
    str,      # cplx
    float,    # kloc
    float,    # effort
    float,    # defects
]

# --------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------

def parse_row(raw: Dict[str, Any]) -> Row | None:
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
        return None
    return (id_, center, year, pmat, cplx, kloc, effort, defects)

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

def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0

def get_numeric_columns(row: Row) -> Dict[str, float]:
    """Return a mapping of numeric field names to values for a single row."""
    _, _, _, _, _, kloc, effort, defects = row
    return {
        "Kloc+": kloc,
        "Effort-": effort,
        "Defects-": defects,
    }

def compare_groups(
    all_rows: List[Row],
    group_rows: List[Row],
    label: str,
) -> List[Tuple[str, float, float, float]]:
    """
    For each numeric characteristic, compute:
        overall_mean, group_mean, diff
    Return a list of tuples sorted by |diff| descending.
    """
    # Overall means
    overall = {k: [] for k in get_numeric_columns(all_rows[0]).keys()}
    for r in all_rows:
        for k, v in get_numeric_columns(r).items():
            overall[k].append(v)
    overall_means = {k: mean(vs) for k, vs in overall.items()}

    # Group means
    group = {k: [] for k in overall_means.keys()}
    for r in group_rows:
        for k, v in get_numeric_columns(r).items():
            group[k].append(v)
    group_means = {k: mean(vs) for k, vs in group.items()}

    # Compute differences
    diffs = []
    for k in overall_means.keys():
        diff = group_means[k] - overall_means[k]
        diffs.append((k, overall_means[k], group_means[k], diff))

    # Sort by absolute difference descending
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    return diffs

# --------------------------------------------------------------------
# NASA-09 – High effort analysis
# --------------------------------------------------------------------

def nasa_09(rows: List[Row]) -> None:
    """Identify characteristics associated with unusually high effort."""
    if not rows:
        print("\n== NASA-09 ==")
        print("  No data available.")
        return

    # Determine 75th percentile threshold for effort
    effort_values = [r[6] for r in rows]
    effort_thresh = quantile(effort_values, 0.75)

    # High effort group
    high_effort = [r for r in rows if r[6] >= effort_thresh]
    if not high_effort:
        print("\n== NASA-09 ==")
        print("  No projects exceed the 75th percentile for effort.")
        return

    diffs = compare_groups(rows, high_effort, "Effort")

    print("\n== NASA-09 ==")
    print(f"High effort threshold: {effort_thresh:.1f} person‑months")
    print(f"Number of high‑effort projects: {len(high_effort)}")
    print("\nTop 3 characteristics most associated with high effort:")
    for idx, (name, overall, group, diff) in enumerate(diffs[:3], start=1):
        print(
            f"  {idx}. {name}: overall={overall:.1f}, "
            f"high-effort={group:.1f} (Δ={diff:+.1f})"
        )

# --------------------------------------------------------------------
# NASA-10 – High defect analysis
# --------------------------------------------------------------------

def nasa_10(rows: List[Row]) -> None:
    """Identify characteristics associated with unusually high defect counts."""
    if not rows:
        print("\n== NASA-10 ==")
        print("  No data available.")
        return

    # Determine 75th percentile threshold for defects
    defect_values = [r[7] for r in rows]
    defect_thresh = quantile(defect_values, 0.75)

    # High defect group
    high_defect = [r for r in rows if r[7] >= defect_thresh]
    if not high_defect:
        print("\n== NASA-10 ==")
        print("  No projects exceed the 75th percentile for defects.")
        return

    diffs = compare_groups(rows, high_defect, "Defects")

    print("\n== NASA-10 ==")
    print(f"High defect threshold: {defect_thresh:.1f} defects")
    print(f"Number of high‑defect projects: {len(high_defect)}")
    print("\nTop 3 characteristics most associated with high defects:")
    for idx, (name, overall, group, diff) in enumerate(diffs[:3], start=1):
        print(
            f"  {idx}. {name}: overall={overall:.1f}, "
            f"high-defect={group:.1f} (Δ={diff:+.1f})"
        )

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main(csv_path: str) -> None:
    rows = read_csv(csv_path)
    nasa_09(rows)
    nasa_10(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)

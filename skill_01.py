#!/usr/bin/env python3
"""
skill_01.py - Summary and identification of favorable software projects.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import math
from statistics import mean, median
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Load the CSV and return a list of row dicts and the header.
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return rows, header

# ----------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------
def get_numeric_column(rows: List[Dict[str, str]], col: str) -> List[float]:
    """Extract numeric values from a column, ignoring missing or non-numeric values."""
    values = []
    for r in rows:
        try:
            val = float(r[col])
            values.append(val)
        except (ValueError, TypeError):
            pass
    return values

def best_and_worst_by_score(rows: List[Dict[str, str]],
                            effort_col: str,
                            defect_col: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Compute a simple score = effort + defects.
    Return the rows with minimal and maximal score.
    """
    best = None
    worst = None
    best_score = math.inf
    worst_score = -math.inf

    for r in rows:
        try:
            eff = float(r[effort_col])
            defects = float(r[defect_col])
        except (ValueError, TypeError):
            continue
        score = eff + defects
        if score < best_score:
            best_score = score
            best = r
        if score > worst_score:
            worst_score = score
            worst = r
    return best, worst

# ----------------------------------------------------------------------
# Favorable projects identification
# ----------------------------------------------------------------------
def favorable_projects(rows: List[Dict[str, str]],
                       size_col: str,
                       effort_col: str,
                       defect_col: str,
                       effort_quantile: float = 0.25,
                       defect_quantile: float = 0.25) -> List[Dict[str, str]]:
    """
    Return projects that are in the lower `effort_quantile` of effort
    and lower `defect_quantile` of defects.  Projects are sorted by
    combined score (effort + defects) ascending.
    """
    efforts = get_numeric_column(rows, effort_col)
    defects = get_numeric_column(rows, defect_col)

    if not efforts or not defects:
        return []

    effort_thresh = median([x for x in efforts if x is not None]) \
                    if effort_quantile == 0.5 else sorted(efforts)[int(len(efforts)*effort_quantile)]
    defect_thresh = median([x for x in defects if x is not None]) \
                    if defect_quantile == 0.5 else sorted(defects)[int(len(defects)*defect_quantile)]

    fav = []
    for r in rows:
        try:
            eff = float(r[effort_col])
            defc = float(r[defect_col])
        except (ValueError, TypeError):
            continue
        if eff <= effort_thresh and defc <= defect_thresh:
            fav.append((eff + defc, r))
    # Sort by score ascending
    fav.sort(key=lambda x: x[0])
    return [r for _, r in fav]

# ----------------------------------------------------------------------
# Main reporting
# ----------------------------------------------------------------------
def report_nasa_01(rows: List[Dict[str, str]]) -> None:
    """Print summary for NASA-01."""
    size_col = 'Kloc+'
    effort_col = 'Effort-'
    defect_col = 'Defects-'

    sizes = get_numeric_column(rows, size_col)
    efforts = get_numeric_column(rows, effort_col)
    defects = get_numeric_column(rows, defect_col)

    print("== NASA-01 ==")
    if sizes:
        print(f"Mean project size (KLOC): {mean(sizes):.2f}")
    if efforts:
        print(f"Mean development effort: {mean(efforts):.2f}")
    if defects:
        print(f"Mean defects: {mean(defects):.2f}")

    best, worst = best_and_worst_by_score(rows, effort_col, defect_col)

    if best:
        print(f"Best outcome: project {best['idX']} "
              f"(size: {best.get('Kloc+', 'N/A')}, effort: {best.get('Effort-', 'N/A')}, defects: {best.get('Defects-', 'N/A')})")
    if worst:
        print(f"Worst outcome: project {worst['idX']} "
              f"(size: {worst.get('Kloc+', 'N/A')}, effort: {worst.get('Effort-', 'N/A')}, defects: {worst.get('Defects-', 'N/A')})")

def report_nasa_02(rows: List[Dict[str, str]]) -> None:
    """Print favorable projects for NASA-02."""
    print("\n== NASA-02 ==")
    fav = favorable_projects(rows, 'Kloc+', 'Effort-', 'Defects-')
    if not fav:
        print("No projects meet the favorable criteria.")
        return
    print("Projects with favorable size, effort, defects:")
    for r in fav[:10]:  # limit to top 10
        print(f"- id {r['idX']}: size {r.get('Kloc+', 'N/A')}, "
              f"effort {r.get('Effort-', 'N/A')}, defects {r.get('Defects-', 'N/A')}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, _ = load_csv(csv_path)

    report_nasa_01(rows)
    report_nasa_02(rows)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skill_01.py

A self‑contained script that processes the NASA project CSV and answers:
- NASA-01: A statistical summary of historical projects.
- NASA-02: Projects that performed unusually well in size, effort, and defects.

Run as:
    python skill_01.py path/to/nasa93dem.csv
"""

import csv
import sys
import statistics
import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """
    Reads the CSV and returns a list of dicts with numeric fields converted.
    Non‑numeric fields are kept as strings. Missing numeric values are
    treated as None and excluded from calculations.
    """
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
                # try to convert to float
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data

def _column_values(data, column):
    """Return a list of numeric values for the given column, excluding None."""
    return [row[column] for row in data if isinstance(row.get(column), (int, float))]

def _mean_std(values):
    """Return mean and standard deviation of a list of numbers."""
    if not values:
        return (None, None)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return (mean_val, std_val)

def _best_worst_defects(data):
    """Return the rows with lowest and highest Defects- values."""
    def get_defects(row):
        return row.get("Defects-")
    numeric_rows = [row for row in data if isinstance(get_defects(row), (int, float))]
    best = min(numeric_rows, key=get_defects, default=None)
    worst = max(numeric_rows, key=get_defects, default=None)
    return best, worst

def _favorable_projects(data, kloc_std, effort_std, defect_std):
    """
    Return projects where Kloc+, Effort-, and Defects- are each
    at least two standard deviations below their respective means.
    """
    favorable = []
    for row in data:
        kloc = row.get("Kloc+")
        effort = row.get("Effort-")
        defect = row.get("Defects-")
        if not all(isinstance(v, (int, float)) for v in (kloc, effort, defect)):
            continue
        if (kloc <= (data["kloc_mean"] - 2 * kloc_std) and
            effort <= (data["effort_mean"] - 2 * effort_std) and
            defect <= (data["defect_mean"] - 2 * defect_std)):
            favorable.append(row)
    return favorable

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    # Extract numeric columns
    kloc_values = _column_values(data, "Kloc+")
    effort_values = _column_values(data, "Effort-")
    defect_values = _column_values(data, "Defects-")

    # Compute statistics
    kloc_mean, kloc_std = _mean_std(kloc_values)
    effort_mean, effort_std = _mean_std(effort_values)
    defect_mean, defect_std = _mean_std(defect_values)

    # Store means for later use
    for row in data:
        row.update({"kloc_mean": kloc_mean, "effort_mean": effort_mean, "defect_mean": defect_mean})

    # Best and worst
    best, worst = _best_worst_defects(data)

    # --- NASA-01 ------------------------------------------------------------ #
    print("== NASA-01 ==")
    print(f"Avg Kloc+: {kloc_mean:.2f}  Avg Effort: {effort_mean:.2f}  Avg Defects: {defect_mean:.2f}")
    if best:
        print(f"Best: id {best.get('idX')} (Defects- = {best.get('Defects-')})")
    if worst:
        print(f"Worst: id {worst.get('idX')} (Defects- = {worst.get('Defects-')})")
    print()

    # --- NASA-02 ------------------------------------------------------------ #
    favorable = _favorable_projects(data, kloc_std, effort_std, defect_std)

    print("== NASA-02 ==")
    if favorable:
        print("Projects with unusually favorable combinations (size, effort, defects):")
        for proj in favorable[:10]:  # limit to 10 for brevity
            print(f"  id {proj.get('idX')}: Kloc+={proj.get('Kloc+')}, Effort={proj.get('Effort-')}, Defects={proj.get('Defects-')}")
    else:
        print("No projects meet the stringent criteria for unusually favorable outcomes.")
    print()

if __name__ == "__main__":
    main()

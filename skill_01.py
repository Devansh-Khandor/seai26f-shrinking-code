#!/usr/bin/env python3
"""
skill_01.py - Historical project summary and identification of favorable projects.

Usage:
    python skill_01.py <path-to-nasa93dem.csv>
"""

import csv
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

# -------------------- Utility functions --------------------

def parse_float(value: str) -> float | None:
    """Convert a string to a float, returning None if conversion fails."""
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None

def load_data(csv_path: Path) -> List[Dict]:
    """Read the CSV and return a list of project dicts with numeric fields parsed."""
    projects = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proj = {}
            # Copy all original fields
            for k, v in row.items():
                proj[k] = v.strip()
            # Parse numeric fields of interest
            for field in ("Kloc+", "Effort-", "Defects-"):
                proj[field] = parse_float(proj.get(field, ""))
            # Skip if any critical numeric is missing
            if proj["Kloc+"] is None or proj["Effort-"] is None or proj["Defects-"] is None:
                continue
            projects.append(proj)
    return projects

def compute_statistics(projects: List[Dict]) -> Dict[str, Tuple[float, float, float]]:
    """Return mean, min, max for Kloc+, Effort- and Defects-."""
    stats = {}
    for field in ("Kloc+", "Effort-", "Defects-"):
        values = [p[field] for p in projects]
        stats[field] = (
            statistics.mean(values),
            min(values),
            max(values)
        )
    return stats

def find_best_worst(projects: List[Dict]) -> Tuple[Dict, Dict]:
    """Return the project with minimum Effort- and the one with maximum Effort-."""
    best = min(projects, key=lambda p: p["Effort-"])
    worst = max(projects, key=lambda p: p["Effort-"])
    return best, worst

def compute_medians(projects: List[Dict]) -> Tuple[float, float]:
    """Return median of effort per Kloc and defects per Kloc."""
    eff_per_kloc = [p["Effort-"] / p["Kloc+"] for p in projects]
    defect_per_kloc = [p["Defects-"] / p["Kloc+"] for p in projects]
    return statistics.median(eff_per_kloc), statistics.median(defect_per_kloc)

def find_favorable(projects: List[Dict], eff_median: float, defect_median: float,
                   top_n: int = 5) -> List[Dict]:
    """Return projects whose effort/Kloc and defects/Kloc are below median."""
    candidates = [
        p for p in projects
        if (p["Effort-"] / p["Kloc+"] <= eff_median) and (p["Defects-"] / p["Kloc+"] <= defect_median)
    ]
    # Sort by combined metric (sum of ratios)
    sorted_candidates = sorted(
        candidates,
        key=lambda p: (p["Effort-"] / p["Kloc+"] + p["Defects-"] / p["Kloc+"])
    )
    return sorted_candidates[:top_n]

# -------------------- Main --------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path-to-nasa93dem.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    projects = load_data(csv_path)

    if not projects:
        print("Error: No valid project data found.")
        sys.exit(1)

    # NASA-01: Summary
    stats = compute_statistics(projects)
    best, worst = find_best_worst(projects)

    print("== NASA-01 ==")
    print(f"Typical project size (KLOC): mean={stats['Kloc+'][0]:.1f}, min={stats['Kloc+'][1]:.1f}, max={stats['Kloc+'][2]:.1f}")
    print(f"Typical development effort: mean={stats['Effort-'][0]:.1f} person‑months, min={stats['Effort-'][1]:.1f}, max={stats['Effort-'][2]:.1f}")
    print(f"Typical defect counts: mean={stats['Defects-'][0]:.1f}, min={stats['Defects-'][1]:.1f}, max={stats['Defects-'][2]:.1f}")
    print(f"Best outcome (lowest effort): id={best['idX']}, effort={best['Effort-']:.1f}, defects={best['Defects-']:.1f}")
    print(f"Worst outcome (highest effort): id={worst['idX']}, effort={worst['Effort-']:.1f}, defects={worst['Defects-']:.1f}")

    # NASA-02: Favorable projects
    eff_med, defect_med = compute_medians(projects)
    favorable = find_favorable(projects, eff_med, defect_med, top_n=5)

    print("\n== NASA-02 ==")
    if not favorable:
        print("No projects found that satisfy the favorable criteria.")
    else:
        print(f"{'idX':>4}  {'KLOC':>6}  {'Effort':>7}  {'Defects':>8}")
        for p in favorable:
            print(f"{p['idX']:>4}  {p['Kloc+']:>6.1f}  {p['Effort-']:>7.1f}  {p['Defects-']:>8.1f}")

if __name__ == "__main__":
    main()

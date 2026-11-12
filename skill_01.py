#!/usr/bin/env python3
"""
Skill 01: Summarise historical NASA software projects and identify unusually favourable projects.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import numpy as np
from typing import List, Dict, Any


def load_data(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load the CSV data and return a list of rows as dictionaries.
    Only rows with valid numeric Kloc+, Effort- and Defects- are kept.
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r['Kloc+'] = float(r['Kloc+'])
                r['Effort-'] = float(r['Effort-'])
                r['Defects-'] = float(r['Defects-'])
            except (ValueError, KeyError):
                # Skip rows with missing or non-numeric values
                continue
            rows.append(r)
    return rows


def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean, std, min and max for Kloc+, Effort- and Defects-.
    Returns a dictionary keyed by the metric name.
    """
    data = {
        'Kloc+': np.array([r['Kloc+'] for r in rows]),
        'Effort-': np.array([r['Effort-'] for r in rows]),
        'Defects-': np.array([r['Defects-'] for r in rows]),
    }
    stats = {}
    for key, arr in data.items():
        stats[key] = {
            'mean': arr.mean(),
            'std': arr.std(ddof=1),
            'min': arr.min(),
            'max': arr.max(),
        }
    return stats


def best_and_worst(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Identify best (min effort, min defects) and worst (max effort, max defects) projects.
    Returns a dict with two entries: 'best' and 'worst'.
    """
    # Best: lowest Effort-, then lowest Defects-
    best = min(rows, key=lambda r: (r['Effort-'], r['Defects-']))
    # Worst: highest Effort-, then highest Defects-
    worst = max(rows, key=lambda r: (r['Effort-'], r['Defects-']))
    return {'best': best, 'worst': worst}


def identify_favorable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return rows that fall below the 25th percentile for size, effort and defects.
    """
    size_arr = np.array([r['Kloc+'] for r in rows])
    effort_arr = np.array([r['Effort-'] for r in rows])
    defect_arr = np.array([r['Defects-'] for r in rows])

    size_thr = np.percentile(size_arr, 25)
    effort_thr = np.percentile(effort_arr, 25)
    defect_thr = np.percentile(defect_arr, 25)

    favorable = [
        r for r in rows
        if r['Kloc+'] <= size_thr
        and r['Effort-'] <= effort_thr
        and r['Defects-'] <= defect_thr
    ]
    return favorable


def print_summary(stats: Dict[str, Dict[str, float]], bw: Dict[str, Dict[str, Any]]):
    print("== NASA-01 ==")
    # At least three lines of results
    print(f"Typical size (KLOC): {stats['Kloc+']['mean']:.2f} ± {stats['Kloc+']['std']:.2f}")
    print(f"Typical effort (PM): {stats['Effort-']['mean']:.2f} ± {stats['Effort-']['std']:.2f}")
    print(f"Typical defects: {stats['Defects-']['mean']:.0f} ± {stats['Defects-']['std']:.0f}")
    # Best and worst
    best = bw['best']
    worst = bw['worst']
    print(f"Best project (ID {best['idX']}): size {best['Kloc+']:.2f}, effort {best['Effort-']:.2f}, defects {best['Defects-']}")
    print(f"Worst project (ID {worst['idX']}): size {worst['Kloc+']:.2f}, effort {worst['Effort-']:.2f}, defects {worst['Defects-']}")


def print_favorable(fav: List[Dict[str, Any]]):
    print("\n== NASA-02 ==")
    if not fav:
        print("No projects meet the favorable criteria.")
        return
    # Print header
    print(f"{'ID':>3} {'Size(KLOC)':>10} {'Effort(PM)':>10} {'Defects':>8}")
    for r in fav:
        print(f"{int(r['idX']):>3} {r['Kloc+']:10.2f} {r['Effort-']:10.2f} {int(r['Defects-']):8d}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_data(csv_path)
    if not rows:
        print("No valid data found.")
        sys.exit(1)

    stats = compute_stats(rows)
    bw = best_and_worst(rows)
    favorable = identify_favorable(rows)

    print_summary(stats, bw)
    print_favorable(favorable)


if __name__ == "__main__":
    main()
